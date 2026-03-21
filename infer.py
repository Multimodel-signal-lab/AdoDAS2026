#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from common.data.dataset import FeatureConfig
from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.models.grouped_model import CORALHead, GroupedModel
from common.models.heads import A1Head, A2OrdinalHead
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone
from common.runner import (
    _normalize_decode_method,
    generate_submission_grouped,
    setup_logging,
)
from common.utils.ckpt import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["a1", "a2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test_hidden")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_config(config_path: str | None, checkpoint_path: Path) -> dict:
    if config_path is None:
        candidate = checkpoint_path.parent.parent / "config_used.yaml"
        config_path = str(candidate)
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping in the config YAML")
    cfg.update(feature_selection)
    return cfg


def load_calibration(run_dir: Path, task: str) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    calibration_dir = run_dir / "calibration"
    if task == "a1":
        path = calibration_dir / "a1_bias_grouped.json"
        if not path.exists():
            return None, None, "participant"
        with open(path) as f:
            data = json.load(f)
        biases = torch.tensor(data.get("biases", []), dtype=torch.float32) if data.get("biases") else None
        return biases, None, "participant"

    path = calibration_dir / "a2_threshold_offsets_grouped.json"
    if not path.exists():
        return None, None, _normalize_decode_method("expectation")
    with open(path) as f:
        data = json.load(f)
    selected_method = _normalize_decode_method(data.get("selected_decode_method", "expectation"))
    strategies = data.get("strategies", {})
    selected_strategy = data.get("selected_strategy", "")
    offsets = None
    if selected_strategy in strategies and "offsets" in strategies[selected_strategy]:
        offsets = torch.tensor(strategies[selected_strategy]["offsets"], dtype=torch.float32)
    return None, offsets, selected_method


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(args.config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = checkpoint_path.parent.parent
    setup_logging(run_dir / "logs", f"infer_{args.task}")

    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))
    manifest_path = Path(args.manifest) if args.manifest else manifest_dir / f"{args.split}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", defaults.feature_root),
        audio_features=cfg.get("audio_features", defaults.audio_features),
        video_features=cfg.get("video_features", defaults.video_features),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", defaults.video_ssl_model_tag),
        mask_policy=cfg.get("mask_policy", defaults.mask_policy),
        core_audio=cfg.get("core_audio", defaults.core_audio),
        core_video=cfg.get("core_video", defaults.core_video),
    )

    ds = GroupedParticipantDataset(manifest_path, feat_cfg, split=args.split)
    preload = bool(cfg.get("preload", True))
    num_workers = int(cfg.get("num_workers", 8))
    if preload:
        ds.preload(desc=f"Preload {args.split}")
        num_workers = 0

    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=grouped_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    dims = ds.feature_dims
    bb_cfg = BackboneConfig(
        audio_group_dims={n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims},
        audio_pooled_group_dims={n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims},
        video_group_dims={n: dims[n] for n in feat_cfg.video_features if n in dims},
        d_adapter=cfg.get("d_adapter", 64),
        d_model=cfg.get("d_model", 256),
        tcn_layers=cfg.get("tcn_layers", 6),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
        asp_alpha=cfg.get("asp_alpha", 0.5),
        asp_beta=cfg.get("asp_beta", 0.5),
        dropout=cfg.get("dropout", 0.2),
        d_shared=cfg.get("d_shared", 256),
    )
    grouped_model = GroupedModel(
        backbone=MTCNBackbone(bb_cfg),
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    if args.task == "a1":
        task_head = A1Head(bb_cfg.d_shared).to(device)
    else:
        if bool(cfg.get("use_coral", False)):
            task_head = CORALHead(bb_cfg.d_shared).to(device)
        else:
            task_head = A2OrdinalHead(bb_cfg.d_shared).to(device)

    state = load_checkpoint(checkpoint_path, grouped_model, optimizer=None)
    task_head.load_state_dict(state["head_state_dict"])
    grouped_model.eval()
    task_head.eval()

    a1_biases, a2_offsets, selected_decode_method = load_calibration(run_dir, args.task)
    use_amp = bool(cfg.get("amp", True))
    submission_level = cfg.get("submission_level", "participant")

    pids, sessions, preds = generate_submission_grouped(
        grouped_model=grouped_model,
        task_head=task_head,
        loader=loader,
        device=device,
        task=args.task,
        use_amp=use_amp,
        desc=f"Infer {args.split}",
        submission_level=submission_level,
        a1_biases=None if a1_biases is None else a1_biases.to(device),
        decode_method=selected_decode_method,
        a2_threshold_offsets=None if a2_offsets is None else a2_offsets.to(device),
    )

    manifest_df = pd.read_csv(manifest_path)
    file_ids = []
    filtered_preds = []
    if submission_level == "participant":
        pid_to_info = {}
        for _, row in manifest_df.iterrows():
            pid = str(row["anon_pid"])
            pid_to_info.setdefault(pid, (str(row["anon_school"]), str(row["anon_class"])))

        for pid, pred in zip(pids, preds):
            pid_str = str(pid)
            info = pid_to_info.get(pid_str)
            if info is None:
                continue
            school, cls = info
            file_ids.append(f"{school}_{cls}_{pid_str}")
            filtered_preds.append(pred)
    else:
        pid_to_info = {
            (str(row["anon_pid"]), str(row["session"])): (
                str(row["anon_school"]),
                str(row["anon_class"]),
            )
            for _, row in manifest_df.iterrows()
        }

        for pid, sess, pred in zip(pids, sessions, preds):
            key = (str(pid), str(sess))
            info = pid_to_info.get(key)
            if info is None:
                continue
            school, cls = info
            file_ids.append(f"{school}_{cls}_{key[0]}_{key[1]}")
            filtered_preds.append(pred)

    if args.task == "a1":
        sub = pd.DataFrame(
            {
                "file_id": file_ids,
                "p_D": [float(pred[0]) for pred in filtered_preds],
                "p_A": [float(pred[1]) for pred in filtered_preds],
                "p_S": [float(pred[2]) for pred in filtered_preds],
            }
        )
    else:
        sub = pd.DataFrame({"file_id": file_ids})
        for idx, col in enumerate([f"d{i:02d}" for i in range(1, 22)]):
            sub[col] = [int(pred[idx]) for pred in filtered_preds]

    output_path = Path(args.output) if args.output else run_dir / "submissions" / f"submission_{args.task}_{args.split}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
