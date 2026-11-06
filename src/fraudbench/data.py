from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"
NUMERIC_COLUMNS = ["Time", "Amount"] + [f"V{index}" for index in range(1, 29)]
CATEGORICAL_COLUMNS: list[str] = []
MODEL_COLUMNS = NUMERIC_COLUMNS


def load_file(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ULB creditcard.csv not found: {dataset_path}")
    frame = pd.read_csv(dataset_path, usecols=MODEL_COLUMNS + [TARGET])
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    if set(frame[TARGET].unique()) - {0, 1}:
        raise ValueError("Class must be binary")
    return frame.sort_values("Time", kind="stable").reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, MODEL_COLUMNS].copy()


def limit_rows(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.copy()
    selected, _ = train_test_split(
        frame, train_size=maximum, random_state=seed, stratify=frame[TARGET]
    )
    return selected.sort_values("Time", kind="stable")


def chronological_split(frame: pd.DataFrame, train_fraction: float, validation_fraction: float):
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    return {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }


def load_splits(config: dict) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], pd.DataFrame]:
    frame = load_file(config["dataset_path"])
    raw_splits = chronological_split(
        frame, float(config["train_fraction"]), float(config["validation_fraction"])
    )
    limits = {
        "train": config.get("max_train_rows"),
        "validation": config.get("max_validation_rows"),
        "test": config.get("max_test_rows"),
    }
    prepared = {}
    for offset, (name, split) in enumerate(raw_splits.items()):
        limited = limit_rows(split, limits[name], int(config["random_seed"]) + offset)
        prepared[name] = (make_features(limited), limited[TARGET].astype("int8"))
    return prepared, frame


def profile(frame: pd.DataFrame, splits: dict) -> dict:
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "fraud_rows": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "time_min_seconds": float(frame["Time"].min()),
        "time_max_seconds": float(frame["Time"].max()),
        "split_rows": {name: len(values[0]) for name, values in splits.items()},
        "feature_note": "V1-V28 are anonymized PCA components; their business meaning is unavailable.",
    }
