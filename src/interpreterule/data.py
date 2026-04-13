"""Data loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    """Container for labeled training data and unlabeled test data."""

    train: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the coursework CSV and drop any exported index column."""
    df = pd.read_csv(path).copy()
    unnamed = [col for col in df.columns if col.lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def split_labeled_unlabeled(df: pd.DataFrame, label_column: str = "clan") -> DatasetSplit:
    """Split rows by whether the label column is observed."""
    train = df.loc[df[label_column].notna()].reset_index(drop=True)
    test = df.loc[df[label_column].isna()].reset_index(drop=True)
    return DatasetSplit(train=train, test=test)
