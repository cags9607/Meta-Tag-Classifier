from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    artifacts_dir: str
    text_column: str = "selected_text"
    label_column: Optional[str] = None
    model: Dict[str, Any] = field(default_factory = dict)
    preprocessing: Dict[str, Any] = field(default_factory = dict)

    @staticmethod
    def load(path):
        with open(path, "r", encoding = "utf-8") as f:
            d = yaml.safe_load(f)
        return Config(**d)
