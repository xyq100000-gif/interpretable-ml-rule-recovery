"""Metric utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def confusion_matrix_df(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence,
) -> pd.DataFrame:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for yt, yp in zip(y_true, y_pred):
        matrix[label_to_idx[yt], label_to_idx[yp]] += 1
    return pd.DataFrame(matrix, index=labels, columns=labels)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom == 0:
        return 1.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)
