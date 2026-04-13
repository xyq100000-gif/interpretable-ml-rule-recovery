"""Simple filter-style feature ranking tools."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def fisher_score_1d(x: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> float:
    """One-dimensional Fisher score for class separation."""
    mu = float(x.mean())
    between = 0.0
    within = 0.0
    for label in classes:
        group = x[y == label]
        if len(group) < 2:
            continue
        between += len(group) * float((group.mean() - mu) ** 2)
        within += (len(group) - 1) * float(group.var(ddof=1))
    return between / (within + 1e-12)


def rank_features_by_fisher(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    label_column: str,
    classes: Sequence[str],
) -> pd.DataFrame:
    scores = [
        {
            "feature": feature,
            "fisher_score": fisher_score_1d(
                df[feature].to_numpy(dtype=float),
                df[label_column].to_numpy(),
                classes,
            ),
        }
        for feature in feature_names
    ]
    out = pd.DataFrame(scores).sort_values("fisher_score", ascending=False).reset_index(drop=True)
    return out
