from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .feature_io import SequenceData, load_egemaps_pooled, load_sequence

log = logging.getLogger(__name__)


SESSIONS = ["A01", "B01", "B02", "B03"]
SESSION_TO_IDX = {s: i for i, s in enumerate(SESSIONS)}
ITEM_COLS = [f"d{i:02d}" for i in range(1, 22)]
A1_COLS = ["y_D", "y_A", "y_S"]
POOLED_AUDIO_FEATURES = {"egemaps"}


@dataclass
class FeatureConfig:
    feature_root: str = "/media/k3nwong/Data1/test/outputs/pipeline/anonymized"
    audio_features: list[str] = field(
        default_factory=lambda: ["mel_mfcc", "vad", "egemaps", "ssl_embed"]
    )
    video_features: list[str] = field(
        default_factory=lambda: [
            "headpose_geom", "face_behavior", "qc_stats", "vad_agg",
            "body_pose", "global_motion", "vision_ssl_embed",
        ]
    )
    audio_ssl_model_tag: str = "chinese-hubert-base"
    video_ssl_model_tag: str = "dinov2-base"
    grid_step_ms: float = 40.0
    tolerance_ms: float = 25.0


    mask_policy: str = "and_core"
    core_audio: list[str] = field(default_factory=lambda: ["mel_mfcc", "vad"])
    core_video: list[str] = field(default_factory=lambda: ["face_behavior", "headpose_geom"])

    @property
    def audio_sequence_features(self) -> list[str]:
        return [name for name in self.audio_features if name not in POOLED_AUDIO_FEATURES]

    @property
    def audio_pooled_features(self) -> list[str]:
        return [name for name in self.audio_features if name in POOLED_AUDIO_FEATURES]



def _nearest_indices(grid: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(timestamps, grid, side="left")
    idx = np.clip(idx, 0, len(timestamps) - 1)

    idx_left = np.clip(idx - 1, 0, len(timestamps) - 1)
    dist_right = np.abs(grid - timestamps[idx])
    dist_left = np.abs(grid - timestamps[idx_left])
    use_left = dist_left < dist_right
    best_idx = np.where(use_left, idx_left, idx)
    best_dist = np.where(use_left, dist_left, dist_right)
    return best_idx, best_dist


def align_to_grid(
    groups: dict[str, SequenceData],
    grid_step_ms: float = 40.0,
    tolerance_ms: float = 25.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    if not groups:
        raise ValueError("No feature groups supplied for alignment")

    t_min = min(seq.timestamps_ms[0] for seq in groups.values())
    t_max = max(seq.timestamps_ms[-1] for seq in groups.values())
    grid = np.arange(t_min, t_max + grid_step_ms * 0.5, grid_step_ms)
    T = len(grid)

    aligned_feats: dict[str, np.ndarray] = {}
    aligned_masks: dict[str, np.ndarray] = {}

    for name, seq in groups.items():
        best_idx, best_dist = _nearest_indices(grid, seq.timestamps_ms)
        within = best_dist <= tolerance_ms
        aligned_feats[name] = seq.features[best_idx]  # (T, D)
        aligned_masks[name] = seq.valid_mask[best_idx] & within

    return aligned_feats, aligned_masks, grid, T


class MultimodalDataset(Dataset):

    def __init__(
        self,
        manifest_path: str | Path,
        cfg: FeatureConfig,
        split: str,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.feature_root)

        self.manifest = pd.read_csv(manifest_path)
        required = {"anon_school", "anon_class", "anon_pid", "session"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise KeyError(f"Manifest missing columns: {missing}")

        self._feature_dims: dict[str, int] | None = None

        self._cache: list[dict[str, Any] | None] | None = None

    @property
    def feature_dims(self) -> dict[str, int]:
        """Lazy-compute feature dims from the first sample."""
        if self._feature_dims is None:
            self._feature_dims = self._probe_dims()
        return self._feature_dims

    def _probe_dims(self) -> dict[str, int]:
        row = self.manifest.iloc[0]
        dims: dict[str, int] = {}
        for name, seq in self._load_raw_groups(row, "audio").items():
            dims[name] = seq.features.shape[1]
        for name, seq in self._load_raw_groups(row, "video").items():
            dims[name] = seq.features.shape[1]
        if "egemaps" in self.cfg.audio_pooled_features:
            eg = load_egemaps_pooled(
                self.root, self.split,
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            if eg is not None:
                dims["egemaps"] = len(eg)
        return dims

    @staticmethod
    def _compute_modality_mask(
        mask_parts: list[np.ndarray],
        mask_names: list[str],
        core_names: list[str],
        policy: str,
        T: int,
    ) -> np.ndarray:
        if not mask_parts:
            return np.zeros(T, dtype=bool)

        if policy == "or":
            return np.any(np.stack(mask_parts), axis=0)

        if policy == "and_core":
            core_masks = [
                m for m, n in zip(mask_parts, mask_names) if n in core_names
            ]
            if core_masks:
                return np.all(np.stack(core_masks), axis=0)
            return np.any(np.stack(mask_parts), axis=0)

        if policy == "require_k":
            core_masks = [
                m for m, n in zip(mask_parts, mask_names) if n in core_names
            ]
            k = max(1, len(core_names))
            if core_masks:
                return np.sum(np.stack(core_masks), axis=0) >= k
            return np.any(np.stack(mask_parts), axis=0)

        raise ValueError(f"Unknown mask_policy: {policy!r}")


    def preload(self, desc: str | None = None) -> float:
        n = len(self)
        if desc is None:
            desc = f"Preload {self.split}"
        self._cache = [None] * n
        errors = 0
        for i in tqdm(range(n), desc=desc, dynamic_ncols=True):
            try:
                self._cache[i] = self._load_sample(i)
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    log.warning(f"Preload: sample {i} failed: {exc}")
        if errors > 0:
            log.warning(f"Preload: {errors}/{n} samples failed and will be skipped")
        gb = self._estimate_cache_bytes() / 1024**3
        log.info(f"Preloaded {n - errors}/{n} samples ({gb:.1f} GB in RAM)")
        return gb

    def _estimate_cache_bytes(self) -> int:
        total = 0
        if self._cache is None:
            return 0
        for sample in self._cache:
            if sample is None:
                continue
            for v in sample.values():
                if isinstance(v, torch.Tensor):
                    total += v.nelement() * v.element_size()
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, torch.Tensor):
                            total += vv.nelement() * vv.element_size()
        return total

    @property
    def is_preloaded(self) -> bool:
        return self._cache is not None

    def _load_raw_groups(
        self, row: pd.Series, modality: str
    ) -> dict[str, SequenceData]:
        cfg = self.cfg
        feat_list = cfg.audio_sequence_features if modality == "audio" else cfg.video_features
        groups: dict[str, SequenceData] = {}
        for feat_name in feat_list:
            tag: str | None = None
            if feat_name == "ssl_embed":
                tag = cfg.audio_ssl_model_tag
            elif feat_name == "vision_ssl_embed":
                tag = cfg.video_ssl_model_tag
            try:
                seq = load_sequence(
                    self.root, self.split,
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]),
                    modality, feat_name, str(row["session"]),
                    model_tag=tag,
                )
                groups[feat_name] = seq
            except FileNotFoundError:
                pass 
        return groups

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx]
        return self._load_sample(idx)

    def _load_sample(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[idx]
        cfg = self.cfg

        audio_raw = self._load_raw_groups(row, "audio")
        video_raw = self._load_raw_groups(row, "video")

        all_groups = {}
        for k, v in audio_raw.items():
            all_groups[f"audio/{k}"] = v
        for k, v in video_raw.items():
            all_groups[f"video/{k}"] = v

        if not all_groups:
            raise RuntimeError(
                f"No features loaded for {row['anon_pid']} session {row['session']}"
            )

        aligned_feats, aligned_masks, grid_ms, T = align_to_grid(
            all_groups, cfg.grid_step_ms, cfg.tolerance_ms
        )

        audio_groups: dict[str, torch.Tensor] = {}
        video_groups: dict[str, torch.Tensor] = {}
        audio_mask_parts: list[np.ndarray] = []
        audio_mask_names: list[str] = []
        video_mask_parts: list[np.ndarray] = []
        video_mask_names: list[str] = []

        for key, feat in aligned_feats.items():
            modality, name = key.split("/", 1)
            mask = aligned_masks[key]
            t = torch.from_numpy(feat.astype(np.float32))
            if modality == "audio":
                audio_groups[name] = t
                audio_mask_parts.append(mask)
                audio_mask_names.append(name)
            else:
                video_groups[name] = t
                video_mask_parts.append(mask)
                video_mask_names.append(name)

        mask_audio = self._compute_modality_mask(
            audio_mask_parts, audio_mask_names, cfg.core_audio, cfg.mask_policy, T
        )
        mask_video = self._compute_modality_mask(
            video_mask_parts, video_mask_names, cfg.core_video, cfg.mask_policy, T
        )

        vad_signal = np.zeros(T, dtype=np.float32)
        if "audio/vad" in aligned_feats:
            v = aligned_feats["audio/vad"]
            vad_signal = v[:, 0].astype(np.float32) * aligned_masks["audio/vad"].astype(np.float32)
        elif "video/vad_agg" in aligned_feats:
            v = aligned_feats["video/vad_agg"]
            vad_signal = v[:, 0].astype(np.float32) * aligned_masks["video/vad_agg"].astype(np.float32)

        qc_quality = np.zeros(T, dtype=np.float32)
        if "video/qc_stats" in aligned_feats:
            v = aligned_feats["video/qc_stats"]
            qc_quality = v[:, 0].astype(np.float32) * aligned_masks["video/qc_stats"].astype(np.float32)

        audio_pooled_groups: dict[str, torch.Tensor] = {}
        pooled_presence: dict[str, bool] = {}
        if "egemaps" in cfg.audio_pooled_features:
            egemaps = load_egemaps_pooled(
                self.root, self.split,
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            dims = self.feature_dims
            audio_pooled_groups["egemaps"] = (
                torch.from_numpy(egemaps)
                if egemaps is not None
                else torch.zeros(dims.get("egemaps", 88))
            )
            pooled_presence["egemaps"] = egemaps is not None

        session_idx = SESSION_TO_IDX.get(str(row["session"]), 0)

        y_a1 = np.array(
            [float(row.get(c, -1)) for c in A1_COLS], dtype=np.float32
        )
        y_a2 = np.array(
            [float(row.get(c, -1)) for c in ITEM_COLS], dtype=np.float32
        )

        dims = self.feature_dims
        for name in cfg.audio_features:
            if name not in audio_groups and name not in cfg.audio_pooled_features and name in dims:
                audio_groups[name] = torch.zeros(T, dims[name])
        for name in cfg.video_features:
            if name not in video_groups and name in dims:
                video_groups[name] = torch.zeros(T, dims[name])

        return {
            "audio_groups": audio_groups,
            "audio_pooled_groups": audio_pooled_groups,
            "video_groups": video_groups,
            "mask_audio": torch.from_numpy(mask_audio),
            "mask_video": torch.from_numpy(mask_video),
            "vad_signal": torch.from_numpy(vad_signal),
            "qc_quality": torch.from_numpy(qc_quality),
            "audio_pooled_present": pooled_presence,
            "session_idx": session_idx,
            "y_a1": torch.from_numpy(y_a1),
            "y_a2": torch.from_numpy(y_a2),
            "seq_len": T,
            "anon_pid": str(row["anon_pid"]),
            "session": str(row["session"]),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    T_max = max(b["seq_len"] for b in batch)


    audio_names = list(batch[0]["audio_groups"].keys())
    pooled_audio_names = list(batch[0]["audio_pooled_groups"].keys())
    video_names = list(batch[0]["video_groups"].keys())


    def _pad_groups(names: list[str], key: str) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for n in names:
            D = batch[0][key][n].shape[-1]
            t = torch.zeros(B, T_max, D)
            for i, b in enumerate(batch):
                L = b["seq_len"]
                t[i, :L] = b[key][n]
            result[n] = t
        return result

    def _pad_1d(key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        t = torch.zeros(B, T_max, dtype=dtype)
        for i, b in enumerate(batch):
            L = b["seq_len"]
            t[i, :L] = b[key]
        return t

    pad_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        pad_mask[i, : b["seq_len"]] = False

    return {
        "audio_groups": _pad_groups(audio_names, "audio_groups"),
        "audio_pooled_groups": {
            name: torch.stack([b["audio_pooled_groups"][name] for b in batch])
            for name in pooled_audio_names
        },
        "video_groups": _pad_groups(video_names, "video_groups"),
        "mask_audio": _pad_1d("mask_audio", torch.bool),
        "mask_video": _pad_1d("mask_video", torch.bool),
        "pad_mask": pad_mask,
        "vad_signal": _pad_1d("vad_signal"),
        "qc_quality": _pad_1d("qc_quality"),
        "audio_pooled_present": {
            name: torch.tensor(
                [b["audio_pooled_present"].get(name, False) for b in batch],
                dtype=torch.bool,
            )
            for name in pooled_audio_names
        },
        "session_idx": torch.tensor([b["session_idx"] for b in batch], dtype=torch.long),
        "y_a1": torch.stack([b["y_a1"] for b in batch]),
        "y_a2": torch.stack([b["y_a2"] for b in batch]),
        "seq_len": torch.tensor([b["seq_len"] for b in batch], dtype=torch.long),
        "anon_pid": [b["anon_pid"] for b in batch],
        "session": [b["session"] for b in batch],
    }
