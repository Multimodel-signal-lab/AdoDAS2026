from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_features import SequenceFeature, pooled_statistics


@dataclass(frozen=True)
class SSLSequence:
    sequence: SequenceFeature
    pooled: dict[str, np.ndarray]
    model_name: str
    embed_dim: int


def extract_audio_ssl_embeddings(
    audio_path: Path,
    model_name: str,
    sample_rate: int = 16000,
    target_fps: int = 25,
    device: str = "cuda",
) -> SSLSequence:

    import librosa
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModel

    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(audio) / sr if sr else 0.0
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    inputs = extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0).transpose(0, 1).unsqueeze(0)

    target_frames = max(1, int(round(duration * target_fps)))
    resampled = F.interpolate(hidden, size=target_frames, mode="linear", align_corners=False)
    features = resampled.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
    timestamps_ms = np.arange(target_frames, dtype=np.float64) * (1000.0 / target_fps)
    sequence = SequenceFeature(features, timestamps_ms, np.ones(target_frames, dtype=bool))
    return SSLSequence(
        sequence=sequence,
        pooled=pooled_statistics(features, "embed"),
        model_name=model_name,
        embed_dim=int(features.shape[1]),
    )


def extract_vision_ssl_embeddings(
    image_paths: list[Path],
    timestamps_ms: np.ndarray,
    model_name: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> SSLSequence:

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    batches: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs.last_hidden_state[:, 0]
        batches.append(emb.detach().cpu().numpy().astype(np.float32))

    if batches:
        features = np.concatenate(batches, axis=0)
    else:
        features = np.zeros((0, 0), dtype=np.float32)
    valid = np.ones(len(features), dtype=bool)
    sequence = SequenceFeature(features, timestamps_ms.astype(np.float64), valid)
    return SSLSequence(
        sequence=sequence,
        pooled=pooled_statistics(features, "embed") if len(features) else {},
        model_name=model_name,
        embed_dim=int(features.shape[1]) if features.ndim == 2 and features.shape[1:] else 0,
    )
