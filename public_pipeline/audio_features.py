from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SequenceFeature:
    features: np.ndarray
    timestamps_ms: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class MelMFCCFeature:
    mel: SequenceFeature
    mfcc: SequenceFeature
    pooled: dict[str, np.ndarray]


@dataclass(frozen=True)
class VADFeature:
    sequence: SequenceFeature
    speech_ratio: float
    total_speech_duration: float
    total_silence_duration: float
    pause_count: int
    mean_pause_duration: float
    max_pause_duration: float
    long_pause_count: int
    speech_segments: list[tuple[float, float]]
    silence_segments: list[tuple[float, float]]


@dataclass(frozen=True)
class EGeMAPSFeature:
    values: np.ndarray
    names: list[str]
    metadata: dict[str, str | int | float]


def pooled_statistics(features: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features must have shape [T, D]")
    return {
        f"{prefix}_mean": np.mean(features, axis=0),
        f"{prefix}_std": np.std(features, axis=0),
        f"{prefix}_p10": np.percentile(features, 10, axis=0),
        f"{prefix}_p50": np.percentile(features, 50, axis=0),
        f"{prefix}_p90": np.percentile(features, 90, axis=0),
    }


def extract_log_mel_mfcc(
    audio_path: Path,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 640,
    win_length: int = 400,
) -> MelMFCCFeature:

    import librosa

    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sr / 2,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max).T.astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    ).T.astype(np.float32)

    num_frames = log_mel.shape[0]
    hop_ms = hop_length / sample_rate * 1000.0
    timestamps_ms = np.arange(num_frames, dtype=np.float64) * hop_ms
    valid_mask = np.ones(num_frames, dtype=bool)

    pooled = {}
    pooled.update(pooled_statistics(log_mel, "mel"))
    pooled.update(pooled_statistics(mfcc, "mfcc"))
    return MelMFCCFeature(
        mel=SequenceFeature(log_mel, timestamps_ms, valid_mask),
        mfcc=SequenceFeature(mfcc, timestamps_ms, valid_mask.copy()),
        pooled=pooled,
    )


def extract_vad(
    audio_path: Path,
    sample_rate: int = 16000,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    target_fps: int = 25,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.1,
    long_pause_threshold: float = 0.5,
) -> VADFeature:

    import librosa
    import webrtcvad

    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(audio) / sr if sr else 0.0
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)

    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = int(sample_rate * frame_duration_ms / 1000)
    decisions: list[int] = []
    frame_times_ms: list[float] = []
    for start in range(0, max(0, len(pcm16) - frame_samples + 1), frame_samples):
        frame_bytes = pcm16[start : start + frame_samples].tobytes()
        try:
            decisions.append(1 if vad.is_speech(frame_bytes, sample_rate) else 0)
        except Exception:
            decisions.append(0)
        frame_times_ms.append(start / sample_rate * 1000.0)

    target_hop_ms = 1000.0 / target_fps
    target_frames = max(1, int(round(duration * target_fps)))
    timestamps_ms = np.arange(target_frames, dtype=np.float64) * target_hop_ms

    if decisions:
        frame_decisions = np.asarray(decisions, dtype=np.float32)
        frame_times = np.asarray(frame_times_ms, dtype=np.float64)
        nearest = np.abs(timestamps_ms[:, None] - frame_times[None, :]).argmin(axis=1)
        vad_25hz = frame_decisions[nearest]
    else:
        vad_25hz = np.zeros(target_frames, dtype=np.float32)

    speech_segments, silence_segments = _segments_from_binary_track(
        vad_25hz,
        timestamps_ms / 1000.0,
        duration,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
    )
    total_speech = sum(end - start for start, end in speech_segments)
    total_silence = sum(end - start for start, end in silence_segments)
    pauses = [end - start for start, end in silence_segments]

    return VADFeature(
        sequence=SequenceFeature(
            features=vad_25hz.reshape(-1, 1).astype(np.float32),
            timestamps_ms=timestamps_ms,
            valid_mask=np.ones(target_frames, dtype=bool),
        ),
        speech_ratio=total_speech / duration if duration > 0 else 0.0,
        total_speech_duration=total_speech,
        total_silence_duration=total_silence,
        pause_count=len(pauses),
        mean_pause_duration=float(np.mean(pauses)) if pauses else 0.0,
        max_pause_duration=float(max(pauses)) if pauses else 0.0,
        long_pause_count=sum(1 for pause in pauses if pause >= long_pause_threshold),
        speech_segments=speech_segments,
        silence_segments=silence_segments,
    )


def extract_egemaps(
    audio_path: Path,
    feature_set: str = "eGeMAPSv02",
    feature_level: str = "functionals",
) -> EGeMAPSFeature:

    import opensmile

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, feature_set),
        feature_level=getattr(opensmile.FeatureLevel, feature_level),
    )
    frame = smile.process_file(str(audio_path))
    if frame.empty:
        raise RuntimeError("opensmile_returned_empty_feature_frame")
    row = frame.iloc[0]
    values = row.to_numpy(dtype=np.float32)
    return EGeMAPSFeature(
        values=values,
        names=[str(col) for col in frame.columns],
        metadata={
            "feature_set": feature_set,
            "feature_level": feature_level,
            "num_features": int(values.shape[0]),
        },
    )


def align_sequences_to_grid(
    sequences: dict[str, SequenceFeature],
    grid_step_ms: float = 40.0,
    tolerance_ms: float = 25.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:

    if not sequences:
        raise ValueError("at least one sequence is required")

    t_min = min(float(seq.timestamps_ms[0]) for seq in sequences.values())
    t_max = max(float(seq.timestamps_ms[-1]) for seq in sequences.values())
    grid = np.arange(t_min, t_max + grid_step_ms * 0.5, grid_step_ms)

    aligned: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name, seq in sequences.items():
        nearest, distance = _nearest_indices(grid, seq.timestamps_ms)
        aligned[name] = seq.features[nearest]
        masks[name] = seq.valid_mask[nearest] & (distance <= tolerance_ms)
    return aligned, masks, grid


def _nearest_indices(grid: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(timestamps, grid, side="left")
    idx = np.clip(idx, 0, len(timestamps) - 1)
    left = np.clip(idx - 1, 0, len(timestamps) - 1)
    dist_right = np.abs(grid - timestamps[idx])
    dist_left = np.abs(grid - timestamps[left])
    use_left = dist_left < dist_right
    nearest = np.where(use_left, left, idx)
    distance = np.where(use_left, dist_left, dist_right)
    return nearest, distance


def _segments_from_binary_track(
    values: np.ndarray,
    times_s: np.ndarray,
    duration_s: float,
    min_speech_duration: float,
    min_silence_duration: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if len(values) == 0:
        return [], []

    speech: list[tuple[float, float]] = []
    silence: list[tuple[float, float]] = []
    state = int(values[0] > 0.5)
    start = float(times_s[0])

    for idx in range(1, len(values)):
        next_state = int(values[idx] > 0.5)
        if next_state == state:
            continue
        end = float(times_s[idx])
        if state:
            speech.append((start, end))
        else:
            silence.append((start, end))
        start = end
        state = next_state

    if state:
        speech.append((start, duration_s))
    else:
        silence.append((start, duration_s))

    speech = [(s, e) for s, e in speech if e - s >= min_speech_duration]
    silence = [(s, e) for s, e in silence if e - s >= min_silence_duration]
    return speech, silence


def mean_pool_embeddings(embeddings: Iterable[np.ndarray]) -> np.ndarray:

    arrays = [np.asarray(item, dtype=np.float32) for item in embeddings if len(item)]
    if not arrays:
        raise ValueError("at least one non-empty embedding array is required")
    return np.concatenate(arrays, axis=0).mean(axis=0)
