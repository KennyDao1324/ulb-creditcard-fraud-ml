from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("train_path", "test_path", "dataset_path", "features_path", "classes_path"):
        if key in config:
            candidate = Path(config[key])
            config[key] = str(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)
    return config
