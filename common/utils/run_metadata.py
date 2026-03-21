from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_command_line() -> str:
    return " ".join(sys.argv)


class RunMetadata:
    def __init__(
        self,
        run_dir: Path,
        cfg: dict[str, Any],
        task: str,
        run_name: str,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.meta_path = self.run_dir / "run_meta.json"
        self.config_path = self.run_dir / "config_used.yaml"
        self.meta: dict[str, Any] = {
            "run_name": run_name,
            "task": task,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "git_commit": _get_git_commit(),
            "command_line": _get_command_line(),
            "full_config": cfg,
            "audio_ssl_tag": cfg.get("audio_ssl_model_tag", ""),
            "video_ssl_tag": cfg.get("video_ssl_model_tag", ""),
            "feature_combination": {
                "audio_features": cfg.get("audio_features", []),
                "video_features": cfg.get("video_features", []),
                "audio_ssl_model_tag": cfg.get("audio_ssl_model_tag", ""),
                "video_ssl_model_tag": cfg.get("video_ssl_model_tag", ""),
            },
            "best_epoch": None,
            "best_metrics": {},
            "status": "running",
        }
        self._save()

        try:
            import yaml
            with open(self.config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            with open(self.config_path.with_suffix(".json"), "w") as f:
                json.dump(cfg, f, indent=2)

    def update_best(self, epoch: int, metrics: dict[str, float]) -> None:
        """Update best epoch and metrics."""
        self.meta["best_epoch"] = epoch
        self.meta["best_metrics"] = metrics
        self._save()

    def finish(self, status: str = "completed") -> None:
        """Mark run as finished."""
        self.meta["end_time"] = datetime.now().isoformat()
        self.meta["status"] = status
        self._save()

    def set_extra(self, key: str, value: Any) -> None:
        """Set an arbitrary extra field."""
        self.meta[key] = value
        self._save()

    def _save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
