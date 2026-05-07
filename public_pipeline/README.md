# ADODAS Public Pipeline Reference

`public_pipeline` is a public reference implementation for media preprocessing and feature extraction.

## Pipeline Flow

1. **Media statistics**: use `ffprobe` to summarize duration, stream availability, resolution, FPS, and codec fields.
2. **Video standardization**: convert input videos to fixed-FPS MP4 with H.264 video, AAC audio, optional scaling, and faststart container layout.
3. **Audio extraction**: extract audio from standardized videos as 16 kHz mono PCM WAV so audio and video share the same time base.
4. **Audio enhancement**: optionally run ClearerVoice speech enhancement on the extracted WAV before computing acoustic features.
5. **Audio feature extraction**: extract Log-Mel, MFCC, VAD statistics, eGeMAPSv02 functionals, and optional speech SSL embeddings.
6. **Video feature extraction**: extract face frames, face quality, head-pose geometry, face behavior summaries, VAD-aligned video speech context, body pose, global motion, and optional vision SSL embeddings.
7. **Feature saving**: save sequence features with `features`, `timestamps_ms`, and `valid_mask`; save pooled summaries as JSON or parquet tables.
8. **Optional clip aggregation**: aggregate clip-level outputs only after the low-level preprocessing and feature extraction steps are complete.

## Folder Contents

```text
public_pipeline/
├── README.md                 # Pipeline overview and function summary
├── __init__.py               # Package-level exports for common APIs
├── configs.py                # Public default parameters and model lists
├── media_preprocessing.py    # Media stats, video standardization, audio extraction
├── audio_enhancement.py      # ClearerVoice audio denoising/enhancement wrapper
├── audio_features.py         # Log-Mel, MFCC, VAD, eGeMAPSv02, sequence alignment helpers
├── video_features.py         # Face, head-pose, behavior, VAD-video, body-pose, motion features
├── ssl_features.py           # Speech and vision SSL embedding extractors
├── feature_io.py             # Sequence and pooled-feature save utilities
└── orchestrator.py           # Simple clip/participant processing helpers
```