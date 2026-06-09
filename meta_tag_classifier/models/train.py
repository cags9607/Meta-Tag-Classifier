from __future__ import annotations

from pathlib import Path


def train_from_config(config_path: str | Path):
    raise NotImplementedError(
        "Training is not implemented in this deployment repo. "
        "Fine-tune Qwen externally, then package the adapter with scripts/package_qwen_artifacts.py."
    )
