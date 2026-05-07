from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:

    duration: float
    video_duration: float
    audio_duration: float | None
    has_audio: bool
    width: int
    height: int
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None

    @property
    def has_valid_video(self) -> bool:
        return self.video_duration > 0 and self.width > 0 and self.height > 0


@dataclass(frozen=True)
class StandardizationConfig:

    target_fps: int = 25
    target_width: int | None = 1280
    target_height: int | None = 720
    video_codec: str = "libx264"
    crf: int = 18
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    wav_codec: str = "pcm_s16le"
    min_duration: float = 1.0
    command_timeout: int = 300


def _run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def probe_media(video_path: Path, timeout: int = 30) -> MediaInfo | None:

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,duration,avg_frame_rate,codec_name",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = _run_command(command, timeout=timeout)
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None

    video_stream = None
    audio_stream = None
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        return None

    format_duration = float(payload.get("format", {}).get("duration", 0) or 0)
    video_duration = float(video_stream.get("duration", 0) or 0) or format_duration
    audio_duration = None
    if audio_stream is not None:
        audio_duration = float(audio_stream.get("duration", 0) or 0) or format_duration

    fps = None
    avg_frame_rate = video_stream.get("avg_frame_rate")
    if avg_frame_rate and "/" in avg_frame_rate:
        num_s, den_s = avg_frame_rate.split("/", 1)
        try:
            den = float(den_s)
            fps = float(num_s) / den if den else None
        except ValueError:
            fps = None

    return MediaInfo(
        duration=video_duration,
        video_duration=video_duration,
        audio_duration=audio_duration,
        has_audio=audio_stream is not None,
        width=int(video_stream.get("width", 0) or 0),
        height=int(video_stream.get("height", 0) or 0),
        fps=fps,
        video_codec=video_stream.get("codec_name"),
        audio_codec=None if audio_stream is None else audio_stream.get("codec_name"),
    )


def standardize_video(
    input_video: Path,
    output_video: Path,
    config: StandardizationConfig | None = None,
) -> None:

    config = config or StandardizationConfig()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    vf_parts: list[str] = [f"fps={config.target_fps}"]
    if config.target_width is not None and config.target_height is not None:
        vf_parts.append(f"scale={config.target_width}:{config.target_height}")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        ",".join(vf_parts),
        "-c:v",
        config.video_codec,
        "-crf",
        str(config.crf),
        "-preset",
        "medium",
        "-c:a",
        config.audio_codec,
        "-b:a",
        config.audio_bitrate,
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    result = _run_command(command, timeout=config.command_timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:] or "ffmpeg_video_standardization_failed")

    output_info = probe_media(output_video)
    if output_info is None or not output_info.has_valid_video:
        raise RuntimeError("standardized_video_failed_validation")


def extract_audio_wav(
    input_video: Path,
    output_wav: Path,
    config: StandardizationConfig | None = None,
) -> None:

    config = config or StandardizationConfig()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        config.wav_codec,
        "-ar",
        str(config.audio_sample_rate),
        "-ac",
        str(config.audio_channels),
        str(output_wav),
    ]
    result = _run_command(command, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:] or "ffmpeg_audio_extraction_failed")
