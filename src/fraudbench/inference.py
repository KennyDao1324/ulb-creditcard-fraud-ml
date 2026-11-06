from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import PROJECT_ROOT


def predict_record(record: dict, model_path: str | Path | None = None) -> dict:
    path = Path(model_path or PROJECT_ROOT / "checkpoint" / "production_model.joblib")
    bundle = joblib.load(path)
    features = pd.DataFrame([record]).loc[:, bundle["feature_columns"]]
    probability = float(bundle["model"].predict_proba(features)[0, 1])
    return {
        "fraud_probability": probability,
        "prediction": int(probability >= bundle["threshold"]),
        "threshold": float(bundle["threshold"]),
        "model": bundle["selected_model"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one engineered transaction JSON")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(predict_record(json.loads(args.input.read_text()), args.model), indent=2))


if __name__ == "__main__":
    main()
