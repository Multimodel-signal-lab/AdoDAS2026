from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


class SequenceData(NamedTuple):
    features: np.ndarray     
    timestamps_ms: np.ndarray  
    valid_mask: np.ndarray     


_MEL_MFCC_KEYS = ("mel_features", "mfcc_features")
_GENERIC_KEY = "features"


def load_sequence(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    modality: str,
    feature_set: str,
    session: str,
    model_tag: str | None = None,
) -> SequenceData:
    parts = [root, split, anon_school, anon_class, anon_pid, modality, feature_set]
    if model_tag is not None:
        parts.append(model_tag)
    parts.append(session)
    seq_path = Path(*[str(p) for p in parts]) / "sequence.npz"

    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequence file: {seq_path}")

    data = np.load(str(seq_path), allow_pickle=True)

    if feature_set == "mel_mfcc":
        arrays = []
        for k in _MEL_MFCC_KEYS:
            if k not in data:
                raise KeyError(f"Expected key '{k}' in {seq_path}, found {list(data.keys())}")
            arrays.append(data[k].astype(np.float32))
        features = np.concatenate(arrays, axis=-1)
    elif _GENERIC_KEY in data:
        features = data[_GENERIC_KEY].astype(np.float32)
    else:
        raise KeyError(
            f"No known feature key in {seq_path}. Keys: {list(data.keys())}"
        )

    if features.ndim == 1:
        features = features[:, np.newaxis]

    if "timestamps_ms" not in data:
        raise KeyError(f"Missing 'timestamps_ms' in {seq_path}")
    timestamps_ms = data["timestamps_ms"].astype(np.float64)

    if "valid_mask" in data:
        valid_mask = data["valid_mask"].astype(bool)
    else:
        valid_mask = np.ones(len(timestamps_ms), dtype=bool)

    T = len(timestamps_ms)
    if features.shape[0] != T:
        raise ValueError(
            f"Shape mismatch in {seq_path}: features {features.shape[0]} vs timestamps {T}"
        )
    if valid_mask.shape[0] != T:
        raise ValueError(
            f"Shape mismatch in {seq_path}: valid_mask {valid_mask.shape[0]} vs timestamps {T}"
        )

    return SequenceData(features=features, timestamps_ms=timestamps_ms, valid_mask=valid_mask)


def load_egemaps_pooled(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    session: str,
) -> np.ndarray | None:
    base = root / split / anon_school / anon_class / anon_pid / "audio" / "egemaps" / session

    parquet_path = base / "pooled.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            return df.iloc[0].values.astype(np.float32)
        except Exception:
            pass
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path, engine="fastparquet")
            return df.iloc[0].values.astype(np.float32)
        except Exception:
            pass

    json_path = base / "pooled.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
            if "features" in meta and isinstance(meta["features"], dict):
                vals = np.array(list(meta["features"].values()), dtype=np.float32)
                return vals
        except Exception:
            pass

    return None


def discover_feature_sets(
    root: Path, split: str, modality: str, limit: int = 5
) -> dict[str, list[str]]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    result: dict[str, list[str]] = {}
    count = 0
    for sch in sorted(split_dir.iterdir()):
        if not sch.is_dir():
            continue
        for cls_ in sorted(sch.iterdir()):
            if not cls_.is_dir():
                continue
            for pid in sorted(cls_.iterdir()):
                if not pid.is_dir():
                    continue
                mod_dir = pid / modality
                if not mod_dir.exists():
                    continue
                for feat in sorted(mod_dir.iterdir()):
                    if not feat.is_dir():
                        continue
                    name = feat.name
                    if name not in result:
                        sub_dirs = [d.name for d in sorted(feat.iterdir()) if d.is_dir()]
                        sessions = {"A01", "B01", "B02", "B03"}
                        model_tags = [s for s in sub_dirs if s not in sessions]
                        if model_tags:
                            result[name] = sorted(model_tags)
                        else:
                            result[name] = []
                count += 1
                if count >= limit:
                    return result
    return result


def list_file_ids(root: Path, split: str, limit: int = 0) -> list[tuple[str, str, str]]:
    split_dir = root / split
    results: list[tuple[str, str, str]] = []
    for sch in sorted(split_dir.iterdir()):
        if not sch.is_dir():
            continue
        for cls_ in sorted(sch.iterdir()):
            if not cls_.is_dir():
                continue
            for pid in sorted(cls_.iterdir()):
                if not pid.is_dir():
                    continue
                results.append((sch.name, cls_.name, pid.name))
                if limit > 0 and len(results) >= limit:
                    return results
    return results
