from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .audio_features import SequenceFeature


@dataclass(frozen=True)
class FaceFrame:
    timestamp_ms: float
    detected: bool
    det_score: float
    quality_score: float
    blur_score: float
    brightness: float
    landmarks5: np.ndarray | None = None
    aligned_face_path: Path | None = None


def quality_sequence(frames: Iterable[FaceFrame]) -> tuple[SequenceFeature, dict[str, float]]:

    rows = list(frames)
    if not rows:
        empty = SequenceFeature(np.zeros((0, 4), dtype=np.float32), np.zeros(0), np.zeros(0, dtype=bool))
        return empty, {"total_frames": 0.0, "detected_frames": 0.0, "detection_rate": 0.0}

    features = np.asarray(
        [[f.quality_score, f.blur_score, f.brightness, f.det_score] for f in rows],
        dtype=np.float32,
    )
    timestamps = np.asarray([f.timestamp_ms for f in rows], dtype=np.float64)
    valid = np.asarray([f.detected for f in rows], dtype=bool)

    detected = float(valid.sum())
    total = float(len(rows))
    stats = {
        "total_frames": total,
        "detected_frames": detected,
        "detection_rate": detected / total if total else 0.0,
        "quality_mean": _masked_mean(features[:, 0], valid),
        "quality_std": _masked_std(features[:, 0], valid),
        "blur_mean": _masked_mean(features[:, 1], valid),
        "brightness_mean": _masked_mean(features[:, 2], valid),
        "det_score_mean": _masked_mean(features[:, 3], valid),
    }
    return SequenceFeature(features, timestamps, valid), stats


def extract_face_frames_with_insightface(
    video_path: Path,
    detector_name: str = "buffalo_l",
    det_thresh: float = 0.5,
    det_size: tuple[int, int] = (640, 640),
    save_aligned_dir: Path | None = None,
    aligned_size: int = 112,
    device_id: int = 0,
) -> list[FaceFrame]:

    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=detector_name)
    app.prepare(ctx_id=device_id, det_size=det_size)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[FaceFrame] = []
    frame_idx = 0
    if save_aligned_dir is not None:
        save_aligned_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, image = cap.read()
        if not ok:
            break
        faces = [face for face in app.get(image) if float(face.det_score) >= det_thresh]
        timestamp_ms = frame_idx / fps * 1000.0
        if not faces:
            frames.append(
                FaceFrame(
                    timestamp_ms=timestamp_ms,
                    detected=False,
                    det_score=0.0,
                    quality_score=0.0,
                    blur_score=0.0,
                    brightness=0.0,
                    landmarks5=None,
                )
            )
            frame_idx += 1
            continue

        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        blur = _blur_score(crop)
        brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))) if crop.size else 0.0
        quality = float(face.det_score) * min(1.0, blur / 100.0)
        aligned_path = None
        if save_aligned_dir is not None and crop.size:
            resized = cv2.resize(crop, (aligned_size, aligned_size))
            aligned_path = save_aligned_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(aligned_path), resized)

        frames.append(
            FaceFrame(
                timestamp_ms=timestamp_ms,
                detected=True,
                det_score=float(face.det_score),
                quality_score=quality,
                blur_score=blur,
                brightness=brightness,
                landmarks5=np.asarray(face.kps, dtype=np.float32),
                aligned_face_path=aligned_path,
            )
        )
        frame_idx += 1
    cap.release()
    return frames


def headpose_geometry_sequence(frames: Iterable[FaceFrame]) -> tuple[SequenceFeature, dict[str, float]]:

    rows = list(frames)
    features: list[list[float]] = []
    valid: list[bool] = []
    timestamps: list[float] = []
    for frame in rows:
        timestamps.append(frame.timestamp_ms)
        if frame.landmarks5 is None or np.asarray(frame.landmarks5).shape != (5, 2):
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            valid.append(False)
            continue
        landmarks = np.asarray(frame.landmarks5, dtype=np.float32)
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
        eye_center = (left_eye + right_eye) / 2.0
        mouth_center = (left_mouth + right_mouth) / 2.0
        eye_dist = float(np.linalg.norm(right_eye - left_eye) + 1e-6)
        mouth_dist = float(np.linalg.norm(right_mouth - left_mouth) + 1e-6)

        yaw = float((nose[0] - eye_center[0]) / eye_dist)
        pitch = float((nose[1] - eye_center[1]) / eye_dist)
        roll = float(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        eye_mouth_ratio = float(np.linalg.norm(mouth_center - eye_center) / eye_dist)
        mouth_aspect_proxy = mouth_dist / eye_dist

        features.append([yaw, pitch, roll, eye_mouth_ratio, mouth_aspect_proxy])
        valid.append(frame.detected)

    feat = np.asarray(features, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    stats = {
        "yaw_mean": _masked_mean(feat[:, 0], mask) if len(feat) else 0.0,
        "pitch_mean": _masked_mean(feat[:, 1], mask) if len(feat) else 0.0,
        "roll_mean": _masked_mean(feat[:, 2], mask) if len(feat) else 0.0,
        "valid_ratio": float(mask.mean()) if len(mask) else 0.0,
    }
    return SequenceFeature(feat, np.asarray(timestamps, dtype=np.float64), mask), stats


def face_behavior_summary(
    headpose: SequenceFeature,
    quality: SequenceFeature,
    mouth_open_threshold: float = 0.5,
    yaw_change_threshold: float = 0.15,
) -> dict[str, float]:

    if len(headpose.features) == 0:
        return {
            "mouth_open_ratio": 0.0,
            "mouth_movement_std": 0.0,
            "gaze_stability_score": 0.0,
            "yaw_range": 0.0,
            "pitch_range": 0.0,
            "expression_change_count": 0.0,
            "valid_ratio": 0.0,
        }

    valid = headpose.valid_mask
    mouth = headpose.features[:, 4]
    yaw = headpose.features[:, 0]
    pitch = headpose.features[:, 1]
    q_valid = quality.valid_mask if len(quality.valid_mask) == len(valid) else valid
    valid = valid & q_valid

    if not valid.any():
        return {
            "mouth_open_ratio": 0.0,
            "mouth_movement_std": 0.0,
            "gaze_stability_score": 0.0,
            "yaw_range": 0.0,
            "pitch_range": 0.0,
            "expression_change_count": 0.0,
            "valid_ratio": 0.0,
        }

    yaw_valid = yaw[valid]
    pitch_valid = pitch[valid]
    mouth_valid = mouth[valid]
    yaw_delta = np.abs(np.diff(yaw_valid)) if len(yaw_valid) > 1 else np.zeros(0)
    return {
        "mouth_open_ratio": float(np.mean(mouth_valid > mouth_open_threshold)),
        "mouth_movement_std": float(np.std(mouth_valid)),
        "gaze_stability_score": float(1.0 / (1.0 + np.std(yaw_valid) + np.std(pitch_valid))),
        "yaw_range": float(np.max(yaw_valid) - np.min(yaw_valid)),
        "pitch_range": float(np.max(pitch_valid) - np.min(pitch_valid)),
        "expression_change_count": float(np.sum(yaw_delta > yaw_change_threshold)),
        "valid_ratio": float(valid.mean()),
    }


def aggregate_vad_to_video(
    vad: SequenceFeature,
    video_timestamps_ms: np.ndarray,
    local_window_frames: int = 12,
) -> SequenceFeature:

    if len(video_timestamps_ms) == 0:
        return SequenceFeature(np.zeros((0, 4), dtype=np.float32), video_timestamps_ms, np.zeros(0, dtype=bool))
    if len(vad.timestamps_ms) == 0:
        features = np.zeros((len(video_timestamps_ms), 4), dtype=np.float32)
        return SequenceFeature(features, video_timestamps_ms, np.zeros(len(video_timestamps_ms), dtype=bool))

    nearest = np.abs(video_timestamps_ms[:, None] - vad.timestamps_ms[None, :]).argmin(axis=1)
    speech = vad.features[nearest, 0].astype(np.float32)
    valid = vad.valid_mask[nearest]

    local_ratio = np.zeros_like(speech)
    for idx in range(len(speech)):
        lo = max(0, idx - local_window_frames)
        hi = min(len(speech), idx + local_window_frames + 1)
        local_ratio[idx] = float(np.mean(speech[lo:hi]))
    transition = np.concatenate([[0.0], np.abs(np.diff(speech))]).astype(np.float32)

    features = np.stack([speech, speech > 0.5, local_ratio, transition], axis=1).astype(np.float32)
    return SequenceFeature(features, video_timestamps_ms.astype(np.float64), valid)


def global_motion_features(video_path: Path, target_width: int = 320) -> tuple[SequenceFeature, dict[str, float]]:


    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev_gray = None
    rows: list[list[float]] = []
    times: list[float] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = target_width / frame.shape[1]
        frame = cv2.resize(frame, (target_width, int(frame.shape[0] * scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            frame_diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)).mean()
            rows.append([float(mag.mean()), float(mag.std()), float(angle.mean()), float(frame_diff)])
        times.append(frame_idx / fps * 1000.0)
        prev_gray = gray
        frame_idx += 1
    cap.release()

    features = np.asarray(rows, dtype=np.float32)
    timestamps = np.asarray(times, dtype=np.float64)
    valid = np.ones(len(features), dtype=bool)
    stats = {
        "motion_energy": float(features[:, 0].mean()) if len(features) else 0.0,
        "frame_diff_mean": float(features[:, 3].mean()) if len(features) else 0.0,
        "num_frames": float(len(features)),
    }
    return SequenceFeature(features, timestamps, valid), stats


def body_pose_sequence(video_path: Path, model_asset_path: Path, stride: int = 2) -> tuple[SequenceFeature, dict[str, float]]:

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_asset_path)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows: list[list[float]] = []
    times: list[float] = []
    valid: list[bool] = []
    frame_idx = 0
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx / fps * 1000.0)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                row = []
                for idx in indices:
                    lm = landmarks[idx]
                    row.extend([float(lm.x), float(lm.y), float(lm.visibility)])
                rows.append(row)
                valid.append(True)
            else:
                rows.append([0.0] * 27)
                valid.append(False)
            times.append(float(timestamp_ms))
            frame_idx += 1
    cap.release()

    features = np.asarray(rows, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    stats = {
        "num_frames": float(len(features)),
        "valid_ratio": float(mask.mean()) if len(mask) else 0.0,
        "landmarks_mean": np.mean(features[mask], axis=0) if mask.any() else np.zeros(27, dtype=np.float32),
        "landmarks_std": np.std(features[mask], axis=0) if mask.any() else np.zeros(27, dtype=np.float32),
    }
    return SequenceFeature(features, np.asarray(times, dtype=np.float64), mask), stats


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if mask.any() else 0.0


def _masked_std(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.std(values[mask])) if mask.any() else 0.0


def _blur_score(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
