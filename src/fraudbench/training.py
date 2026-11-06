from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import PROJECT_ROOT, load_config
from .data import CATEGORICAL_COLUMNS, MODEL_COLUMNS, NUMERIC_COLUMNS, load_splits, profile
from .metrics import best_f1_threshold, binary_metrics


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    transformers = [("numeric", numeric, NUMERIC_COLUMNS)]
    if CATEGORICAL_COLUMNS:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical, CATEGORICAL_COLUMNS))
    return ColumnTransformer(transformers)


def candidate_models(config: dict, positive_weight: float = 1.0) -> dict[str, Pipeline]:
    seed = int(config["random_seed"])
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=int(config["logistic_max_iter"]),
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(config["random_forest_trees"]),
                        max_depth=int(config["random_forest_max_depth"]),
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=4,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=int(config["xgboost_estimators"]),
                        max_depth=int(config["xgboost_max_depth"]),
                        learning_rate=float(config["xgboost_learning_rate"]),
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        scale_pos_weight=positive_weight,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        tree_method="hist",
                        n_jobs=4,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_processed(splits: dict, destination: Path) -> None:
    tables = []
    for name, (features, target) in splits.items():
        table = features.copy()
        table["target"] = target.to_numpy()
        table["_split"] = name
        tables.append(table)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(destination, index=False, compression="gzip")


def save_plots(comparison, validation, validation_scores, test, test_score, threshold, title):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = PROJECT_ROOT / "outputs"
    output.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        comparison["model"], comparison["pr_auc"], color=["#2563eb", "#0f766e", "#b45309"]
    )
    axis.set_ylabel("Validation PR-AUC")
    axis.set_title(f"{title}: model comparison")
    figure.tight_layout()
    figure.savefig(output / "model_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, score in validation_scores.items():
        PrecisionRecallDisplay.from_predictions(validation, score, name=name, ax=axis)
    axis.set_title(f"{title}: validation precision-recall")
    figure.tight_layout()
    figure.savefig(output / "precision_recall_curves.png", dpi=160)
    plt.close(figure)

    prediction = (test_score >= threshold).astype(int)
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        test, prediction, ax=axis, colorbar=False, values_format="d"
    )
    axis.set_title(f"{title}: held-out confusion matrix")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    target = test.to_numpy()
    axis.hist(test_score[target == 0], bins=40, alpha=0.7, label="negative")
    axis.hist(test_score[target == 1], bins=40, alpha=0.7, label="positive")
    axis.axvline(threshold, color="black", linestyle="--", label="threshold")
    axis.set_xlabel("Fraud score")
    axis.set_title(f"{title}: held-out score distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "prediction_distribution.png", dpi=160)
    plt.close(figure)


def train(config_path: str | Path) -> dict:
    config = load_config(config_path)
    splits, full_frame = load_splits(config)
    save_processed(splits, PROJECT_ROOT / "preprocessed_df" / "model_data.csv.gz")
    train_x, train_y = splits["train"]
    validation_x, validation_y = splits["validation"]
    fitted, validation_scores, thresholds, rows = {}, {}, {}, []
    positive_weight = float((train_y == 0).sum() / max(1, (train_y == 1).sum()))
    for name, model in candidate_models(config, positive_weight).items():
        model.fit(train_x, train_y)
        score = model.predict_proba(validation_x)[:, 1]
        threshold = best_f1_threshold(validation_y.to_numpy(), score)
        metrics = binary_metrics(validation_y, score, threshold)
        fitted[name], validation_scores[name], thresholds[name] = model, score, threshold
        rows.append({"model": name, **metrics})
    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    winner = str(comparison.iloc[0]["model"])
    threshold = float(thresholds[winner])
    test_x, test_y = splits["test"]
    test_score = fitted[winner].predict_proba(test_x)[:, 1]
    test_metrics = binary_metrics(test_y, test_score, threshold)

    development_target = pd.concat([train_y, validation_y], ignore_index=True)
    development_weight = float(
        (development_target == 0).sum() / max(1, (development_target == 1).sum())
    )
    production = candidate_models(config, development_weight)[winner]
    production.fit(
        pd.concat([train_x, validation_x], ignore_index=True),
        development_target,
    )
    bundle = {
        "model": production,
        "threshold": threshold,
        "feature_columns": MODEL_COLUMNS,
        "selected_model": winner,
        "dataset": config["dataset_name"],
    }
    for directory in (PROJECT_ROOT / "checkpoint", PROJECT_ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        joblib.dump(bundle, directory / "production_model.joblib")

    output = PROJECT_ROOT / "outputs"
    output.mkdir(exist_ok=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame({"target": test_y.to_numpy(), "fraud_probability": test_score}).to_csv(
        output / "test_predictions.csv", index=False
    )
    dataset_profile = profile(full_frame, splits)
    dataset_profile["source_hashes"] = {
        key: _file_hash(value)
        for key, value in config.items()
        if key.endswith("_path") and Path(value).is_file()
    }
    (output / "dataset_profile.json").write_text(
        json.dumps(dataset_profile, indent=2), encoding="utf-8"
    )
    results = {
        "selected_model": winner,
        "selection_metric": "validation_pr_auc",
        "validation": json.loads(comparison.iloc[0].to_json()),
        "test": test_metrics,
        "split_rows": {name: len(values[0]) for name, values in splits.items()},
        "leakage_note": "The held-out test period was not used for selection or threshold tuning.",
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    recorded_config = dict(config)
    for key, value in recorded_config.items():
        if key.endswith("_path"):
            recorded_config[key] = str(Path(value).relative_to(PROJECT_ROOT))
    model_card = {"config": recorded_config, "profile": dataset_profile, **results}
    for directory in (PROJECT_ROOT / "checkpoint", PROJECT_ROOT / "artifacts"):
        (directory / "model_card.json").write_text(
            json.dumps(model_card, indent=2), encoding="utf-8"
        )
    save_plots(
        comparison,
        validation_y,
        validation_scores,
        test_y,
        test_score,
        threshold,
        config["dataset_name"],
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train dataset-specific fraud benchmarks")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "training.json"))
    args = parser.parse_args(argv)
    print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__":
    main()
