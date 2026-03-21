from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

_VIDEO_BASE_SHORT = {
    "headpose_geom": "headpose",
    "face_behavior": "facebeh",
    "qc_stats": "qc",
    "vad_agg": "vadagg",
    "body_pose": "bodypose",
    "global_motion": "globmot",
}


def _shorten_video_base(name: str) -> str:
    return _VIDEO_BASE_SHORT.get(name, name)


def build_run_name(
    cfg: dict[str, Any],
    task: str,
    timestamp: str | None = None,
    training_mode: str = "single_session",
) -> str:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts: list[str] = []

    parts.append(task)

    mode_short = "grouped" if "grouped" in training_mode else "single"
    parts.append(mode_short)

    use_coral = cfg.get("use_coral", False)
    if task == "a2" and use_coral:
        parts.append("coral")
    else:
        parts.append("mtcn")

    audio_feats = cfg.get("audio_features", [])
    if not isinstance(audio_feats, list):
        audio_feats = []
    audio_base = [f for f in audio_feats if f not in ("ssl_embed",)]
    if audio_base:
        parts.append(f"a-base-{'+'.join(audio_base)}")
    else:
        parts.append("a-base-none")

    audio_ssl_tag = cfg.get("audio_ssl_model_tag", "")
    if "ssl_embed" in audio_feats and audio_ssl_tag:
        parts.append(f"a-ssl-{audio_ssl_tag}")
    else:
        parts.append("a-ssl-none")

    video_feats = cfg.get("video_features", [])
    if not isinstance(video_feats, list):
        video_feats = []
    video_base = [f for f in video_feats if f not in ("vision_ssl_embed",)]
    if video_base:
        short_names = [_shorten_video_base(f) for f in video_base]
        parts.append(f"v-base-{'+'.join(short_names)}")
    else:
        parts.append("v-base-none")

    video_ssl_tag = cfg.get("video_ssl_model_tag", "")
    if "vision_ssl_embed" in video_feats and video_ssl_tag:
        parts.append(f"v-ssl-{video_ssl_tag}")
    else:
        parts.append("v-ssl-none")

    mask = cfg.get("mask_policy", "or")
    mask_short = mask.replace("_", "")
    parts.append(f"mask-{mask_short}")

    settings: list[str] = []
    if cfg.get("use_pos_weight", False):
        settings.append("pw")
    if task == "a1":
        settings.append("biascalib")
    if task == "a2":
        if use_coral:
            settings.append("pwthr")
        decode = cfg.get("decode_method", "default")
        if decode == "auto":
            settings.append("autodecode")
        elif decode != "default":
            settings.append(f"{decode}decode")
        else:
            settings.append("argmaxdecode")
        settings.append("thrcalib")
    if settings:
        parts.append("_".join(settings))

    seed = cfg.get("seed", 42)
    parts.append(f"seed{seed}")

    parts.append(timestamp)

    return "__".join(parts)


def setup_run_dirs(output_root: Path, run_name: str) -> dict[str, Path]:
    run_dir = output_root / "runs" / run_name
    subdirs = {
        "root": run_dir,
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "submissions": run_dir / "submissions",
        "calibration": run_dir / "calibration",
    }
    for key in ("root", "logs", "checkpoints", "calibration"):
        subdirs[key].mkdir(parents=True, exist_ok=True)
    return subdirs
