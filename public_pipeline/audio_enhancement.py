from __future__ import annotations

from pathlib import Path
import sys

from .configs import AudioDenoiseConfig


class ClearerVoiceEnhancer:
    def __init__(self, config: AudioDenoiseConfig | None = None) -> None:
        self.config = config or AudioDenoiseConfig()
        self._prepare_import_path()
        if self.config.gpu is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.set_device(self.config.gpu)
            except Exception:
                pass

        try:
            from clearvoice import ClearVoice
        except ImportError as exc:
            raise ImportError(
                "ClearerVoice enhancement requires the `clearvoice` package. "
                "Install ClearerVoice-Studio/clearvoice before using this function."
            ) from exc

        self._model = ClearVoice(task=self.config.task, model_names=[self.config.model_name])

    def _prepare_import_path(self) -> None:
        if self.config.clearvoice_root is None:
            return
        root = Path(self.config.clearvoice_root).expanduser().resolve()
        clearvoice_pkg = root / "clearvoice"
        for candidate in (clearvoice_pkg, root):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

    def enhance_file(self, input_wav: Path, output_wav: Path) -> Path:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        enhanced = self._model(input_path=str(input_wav), online_write=False)
        self._model.write(enhanced, output_path=str(output_wav))
        return output_wav


def denoise_audio_file(
    input_wav: Path,
    output_wav: Path,
    config: AudioDenoiseConfig | None = None,
) -> Path:
    enhancer = ClearerVoiceEnhancer(config)
    return enhancer.enhance_file(input_wav, output_wav)
