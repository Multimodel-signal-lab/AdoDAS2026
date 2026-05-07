from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AudioDenoiseConfig:
    clearvoice_root: Path | str | None = "ClearerVoice-Studio"
    model_name: str = "FRCRN_SE_16K"
    task: str = "speech_enhancement"
    gpu: int | None = 0


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    target_fps: int = 25
    mel_bins: int = 80
    mfcc_bins: int = 13
    n_fft: int = 400
    hop_length: int = 640
    win_length: int = 400
    vad_aggressiveness: int = 2
    vad_frame_duration_ms: int = 30
    long_pause_threshold: float = 0.5
    egemaps_set: str = "eGeMAPSv02"
    egemaps_level: str = "functionals"
    audio_ssl_models: tuple[str, ...] = (
        "microsoft/wavlm-base",
        "TencentGameMate/chinese-hubert-base",
        "TencentGameMate/chinese-hubert-large",
        "TencentGameMate/chinese-wav2vec2-base",
        "TencentGameMate/chinese-wav2vec2-large",
        "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    )


@dataclass(frozen=True)
class VideoFeatureConfig:
    target_fps: int = 25
    face_detector: str = "insightface/buffalo_l"
    face_det_thresh: float = 0.5
    face_det_size: tuple[int, int] = (640, 640)
    min_face_size: int = 40
    aligned_face_size: int = 112
    pose_models: tuple[str, ...] = (
        "pose_landmarker_full.task",
        "pose_landmarker_lite.task",
    )
    vision_ssl_models: tuple[str, ...] = (
        "facebook/dinov2-small",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "facebook/vit-mae-base",
        "google/vit-base-patch16-224",
        "google/siglip-base-patch16-224",
        "google/siglip-so400m-patch14-384",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
    )
    motion_downscale_width: int = 320


@dataclass(frozen=True)
class PublicPipelineReleaseConfig:
    """Public parameters contestants need to generate compatible features."""

    denoise: AudioDenoiseConfig = field(default_factory=AudioDenoiseConfig)
    audio: AudioFeatureConfig = field(default_factory=AudioFeatureConfig)
    video: VideoFeatureConfig = field(default_factory=VideoFeatureConfig)
    sequence_file: str = "sequence.npz"
    pooled_json_file: str = "pooled.json"
    pooled_table_file: str = "pooled.parquet"
    grid_step_ms: float = 40.0
    grid_tolerance_ms: float = 25.0


def default_release_config() -> PublicPipelineReleaseConfig:
    return PublicPipelineReleaseConfig()
