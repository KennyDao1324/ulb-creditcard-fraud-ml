import pandas as pd

from fraudbench.data import MODEL_COLUMNS, chronological_split, make_features


def test_chronological_split_has_no_overlap() -> None:
    frame = pd.DataFrame({column: range(100) for column in MODEL_COLUMNS})
    frame["Class"] = [0] * 95 + [1] * 5
    splits = chronological_split(frame, 0.6, 0.2)
    assert splits["train"]["Time"].max() < splits["validation"]["Time"].min()
    assert splits["validation"]["Time"].max() < splits["test"]["Time"].min()
    assert list(make_features(frame).columns) == MODEL_COLUMNS
