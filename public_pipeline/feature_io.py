from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio_features import SequenceFeature


def save_sequence_npz(
    output_dir: Path,
    sequence: SequenceFeature,
    feature_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "features": sequence.features.astype(np.float32),
        "timestamps_ms": sequence.timestamps_ms.astype(np.float64),
        "valid_mask": sequence.valid_mask.astype(bool),
        "feature_version": "public-reference-1.0",
    }
    if feature_names is not None:
        payload["feature_names"] = np.asarray(feature_names)
    if metadata:
        payload.update(metadata)
    path = output_dir / "sequence.npz"
    np.savez_compressed(path, **payload)
    return path


def save_mel_mfcc_npz(
    output_dir: Path,
    mel: SequenceFeature,
    mfcc: SequenceFeature,
    metadata: dict[str, Any] | None = None,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mel_features": mel.features.astype(np.float32),
        "mfcc_features": mfcc.features.astype(np.float32),
        "timestamps_ms": mel.timestamps_ms.astype(np.float64),
        "valid_mask": mel.valid_mask.astype(bool) & mfcc.valid_mask.astype(bool),
        "feature_version": "public-reference-1.0",
    }
    if metadata:
        payload.update(metadata)
    path = output_dir / "sequence.npz"
    np.savez_compressed(path, **payload)
    return path


def save_pooled_json(output_dir: Path, stats: dict[str, Any], filename: str = "pooled.json") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(stats), handle, indent=2, ensure_ascii=False)
    return path


def save_pooled_table(output_dir: Path, stats: dict[str, Any], filename: str = "pooled.parquet") -> Path:

    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    flat = flatten_stats(stats)
    path = output_dir / filename
    pd.DataFrame([flat]).to_parquet(path, index=False)
    return path


def flatten_stats(stats: dict[str, Any], prefix: str = "") -> dict[str, float | str | int | bool]:

    out: dict[str, float | str | int | bool] = {}
    for key, value in stats.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_stats(value, name))
        elif isinstance(value, np.ndarray):
            for idx, scalar in enumerate(value.reshape(-1)):
                out[f"{name}_{idx:04d}"] = float(scalar)
        elif isinstance(value, (list, tuple)) and value and all(_is_number(x) for x in value):
            for idx, scalar in enumerate(value):
                out[f"{name}_{idx:04d}"] = float(scalar)
        elif _is_number(value):
            out[name] = float(value)
        elif isinstance(value, (str, bool, int)):
            out[name] = value
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
