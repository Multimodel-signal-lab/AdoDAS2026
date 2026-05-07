from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .configs import AudioDenoiseConfig
from .media_preprocessing import (
    StandardizationConfig,
    extract_audio_wav,
    probe_media,
    standardize_video,
)
from .audio_features import extract_egemaps, extract_log_mel_mfcc, extract_vad
from .audio_enhancement import denoise_audio_file
from .feature_io import save_mel_mfcc_npz, save_pooled_json, save_pooled_table, save_sequence_npz


@dataclass(frozen=True)
class ParticipantInput:

    participant_id: str
    clips: Mapping[str, Path]


@dataclass(frozen=True)
class ClipInput:
    clip_id: str
    video_path: Path


@dataclass(frozen=True)
class PipelineConfig:
    media: StandardizationConfig = field(default_factory=StandardizationConfig)
    denoise: AudioDenoiseConfig = field(default_factory=AudioDenoiseConfig)
    denoise_audio: bool = True


@dataclass(frozen=True)
class ParticipantPipelineResult:
    participant_id: str
    succeeded: bool
    standardized_videos: dict[str, Path]
    extracted_audio: dict[str, Path]
    enhanced_audio: dict[str, Path]
    errors: dict[str, str]


@dataclass(frozen=True)
class ClipPipelineResult:
    clip_id: str
    media_stats: dict[str, float | int | str | bool | None]
    standardized_video: Path | None
    extracted_audio: Path | None
    enhanced_audio: Path | None
    error: str | None


def run_clip_reference_pipeline(
    clips: list[ClipInput],
    config: PipelineConfig,
    standardized_video_dir: Path,
    extracted_audio_dir: Path,
    enhanced_audio_dir: Path | None = None,
) -> list[ClipPipelineResult]:

    results: list[ClipPipelineResult] = []
    for clip in clips:
        out_video = standardized_video_dir / f"{clip.clip_id}.mp4"
        out_audio = extracted_audio_dir / f"{clip.clip_id}.wav"
        out_enhanced = (
            (enhanced_audio_dir or extracted_audio_dir / "denoised") / f"{clip.clip_id}.wav"
        )
        media_stats, standardized, audio, enhanced, error = _process_clip_media(
            clip.video_path,
            out_video,
            out_audio,
            out_enhanced,
            config,
        )
        results.append(
            ClipPipelineResult(
                clip_id=clip.clip_id,
                media_stats=media_stats,
                standardized_video=standardized,
                extracted_audio=audio,
                enhanced_audio=enhanced,
                error=error,
            )
        )
    return results


def run_public_reference_pipeline(
    participants: list[ParticipantInput],
    config: PipelineConfig,
    standardized_video_dir: Path,
    extracted_audio_dir: Path,
    enhanced_audio_dir: Path | None = None,
) -> list[ParticipantPipelineResult]:

    results: list[ParticipantPipelineResult] = []
    for participant in participants:
        standardized: dict[str, Path] = {}
        audio: dict[str, Path] = {}
        enhanced: dict[str, Path] = {}
        errors: dict[str, str] = {}

        for clip_id, input_video in participant.clips.items():
            out_video = standardized_video_dir / participant.participant_id / f"{clip_id}.mp4"
            out_audio = extracted_audio_dir / participant.participant_id / f"{clip_id}.wav"
            out_enhanced = (
                (enhanced_audio_dir or extracted_audio_dir / "denoised")
                / participant.participant_id
                / f"{clip_id}.wav"
            )
            _, standardized_path, audio_path, enhanced_path, error = _process_clip_media(
                input_video,
                out_video,
                out_audio,
                out_enhanced,
                config,
            )
            if error is not None:
                errors[clip_id] = error
                continue
            if standardized_path is not None:
                standardized[clip_id] = standardized_path
            if audio_path is not None:
                audio[clip_id] = audio_path
            if enhanced_path is not None:
                enhanced[clip_id] = enhanced_path

        results.append(
            ParticipantPipelineResult(
                participant_id=participant.participant_id,
                succeeded=not errors,
                standardized_videos=standardized,
                extracted_audio=audio,
                enhanced_audio=enhanced,
                errors=errors,
            )
        )
    return results


def extract_public_audio_feature_bundle(
    audio_path: Path,
    output_dir: Path,
    *,
    include_egemaps: bool = True,
    denoise_before_features: bool = False,
    denoise_config: AudioDenoiseConfig | None = None,
) -> dict[str, Path]:

    outputs: dict[str, Path] = {}
    feature_audio = audio_path
    if denoise_before_features:
        feature_audio = denoise_audio_file(
            audio_path,
            output_dir / "enhanced_audio" / audio_path.name,
            denoise_config,
        )
        outputs["enhanced_audio"] = feature_audio

    mel_mfcc = extract_log_mel_mfcc(feature_audio)
    mel_dir = output_dir / "mel_mfcc"
    outputs["mel_mfcc_sequence"] = save_mel_mfcc_npz(mel_dir, mel_mfcc.mel, mel_mfcc.mfcc)
    outputs["mel_mfcc_pooled"] = save_pooled_json(mel_dir, mel_mfcc.pooled)

    vad = extract_vad(feature_audio)
    vad_dir = output_dir / "vad"
    outputs["vad_sequence"] = save_sequence_npz(
        vad_dir,
        vad.sequence,
        feature_names=["vad_decision"],
    )
    outputs["vad_pooled"] = save_pooled_json(
        vad_dir,
        {
            "speech_ratio": vad.speech_ratio,
            "total_speech_duration": vad.total_speech_duration,
            "total_silence_duration": vad.total_silence_duration,
            "pause_count": vad.pause_count,
            "mean_pause_duration": vad.mean_pause_duration,
            "max_pause_duration": vad.max_pause_duration,
            "long_pause_count": vad.long_pause_count,
            "speech_segments": vad.speech_segments,
            "silence_segments": vad.silence_segments,
        },
    )

    if include_egemaps:
        egemaps = extract_egemaps(feature_audio)
        egemaps_dir = output_dir / "egemaps"
        stats = {
            "features": egemaps.values,
            "feature_names": egemaps.names,
            "metadata": egemaps.metadata,
        }
        outputs["egemaps_json"] = save_pooled_json(egemaps_dir, stats)
        outputs["egemaps_table"] = save_pooled_table(
            egemaps_dir,
            {name: float(value) for name, value in zip(egemaps.names, egemaps.values)},
        )
    return outputs


def _process_clip_media(
    input_video: Path,
    output_video: Path,
    output_audio: Path,
    output_enhanced_audio: Path,
    config: PipelineConfig,
) -> tuple[
    dict[str, float | int | str | bool | None],
    Path | None,
    Path | None,
    Path | None,
    str | None,
]:
    media_info = probe_media(input_video)
    media_stats = _media_stats_dict(media_info)
    try:
        standardize_video(input_video, output_video, config.media)
        extract_audio_wav(output_video, output_audio, config.media)
        enhanced_audio = (
            denoise_audio_file(output_audio, output_enhanced_audio, config.denoise)
            if config.denoise_audio
            else output_audio
        )
        return media_stats, output_video, output_audio, enhanced_audio, None
    except Exception as exc:
        return media_stats, None, None, None, f"{type(exc).__name__}: {exc}"


def _media_stats_dict(media_info) -> dict[str, float | int | str | bool | None]:
    if media_info is None:
        return {"readable": False}
    return {
        "readable": True,
        "duration": media_info.duration,
        "video_duration": media_info.video_duration,
        "audio_duration": media_info.audio_duration,
        "has_audio": media_info.has_audio,
        "width": media_info.width,
        "height": media_info.height,
        "fps": media_info.fps,
        "video_codec": media_info.video_codec,
        "audio_codec": media_info.audio_codec,
    }
