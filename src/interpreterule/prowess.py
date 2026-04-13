"""Sparse, interpretable linear formula search for prowess."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import DEFAULT_CANDIDATE_PROWESS_FEATURES
from .metrics import mean_absolute_error, r_squared, root_mean_squared_error


@dataclass
class FormulaSearchResult:
    features: list[str]
    float_weights: list[float]
    integer_weights: list[int]
    cv_mae: float
    train_rmse: float
    train_r2: float

    @property
    def formula_string(self) -> str:
        terms = [f"{self.integer_weights[i + 1]}*{feature}" for i, feature in enumerate(self.features)]
        return f"prowess = {self.integer_weights[0]} + " + " + ".join(terms)


@dataclass
class IntegerLinearFormulaModel:
    """Search small linear models and round coefficients to integers."""

    candidate_features: Sequence[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATE_PROWESS_FEATURES))
    subset_sizes: Sequence[int] = (2, 3, 4)

    result_: FormulaSearchResult | None = None

    def fit(self, df_train: pd.DataFrame, target_column: str = "prowess") -> FormulaSearchResult:
        y = df_train[target_column].to_numpy(dtype=float)
        search_rows = []

        for subset_size in self.subset_sizes:
            for feature_subset in combinations(self.candidate_features, subset_size):
                cv_mae = self._cv_mae_for_subset(df_train, list(feature_subset), y, n_splits=5, seed=42)
                search_rows.append((cv_mae, tuple(feature_subset)))

        search_rows.sort(key=lambda item: (item[0], len(item[1])))
        best_mae, best_features = search_rows[0]

        X_best = self._make_design_matrix(df_train, list(best_features))
        float_weights, integer_weights = self._fit_integer_ols(X_best, y)
        train_pred = X_best @ integer_weights

        self.result_ = FormulaSearchResult(
            features=list(best_features),
            float_weights=[float(weight) for weight in float_weights],
            integer_weights=[int(weight) for weight in integer_weights],
            cv_mae=float(best_mae),
            train_rmse=root_mean_squared_error(y, train_pred),
            train_r2=r_squared(y, train_pred),
        )
        return self.result_

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Formula model must be fitted before prediction.")
        X = self._make_design_matrix(df, self.result_.features)
        return X @ np.array(self.result_.integer_weights, dtype=float)

    def coefficient_table(self) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("Formula model must be fitted before exporting coefficients.")
        return pd.DataFrame(
            {
                "term": ["intercept"] + self.result_.features,
                "float_weight": self.result_.float_weights,
                "integer_weight": self.result_.integer_weights,
            }
        )

    @staticmethod
    def _make_design_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
        X = df[features].to_numpy(dtype=float)
        bias = np.ones((len(df), 1), dtype=float)
        return np.hstack([bias, X])

    @staticmethod
    def _fit_integer_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        float_weights, *_ = np.linalg.lstsq(X, y, rcond=None)
        integer_weights = np.rint(float_weights).astype(int)
        return float_weights, integer_weights

    def _cv_mae_for_subset(
        self,
        df_train: pd.DataFrame,
        features: list[str],
        y: np.ndarray,
        n_splits: int,
        seed: int,
    ) -> float:
        X = self._make_design_matrix(df_train, features)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_mae = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            _, integer_weights = self._fit_integer_ols(X_train, y_train)
            fold_pred = X_val @ integer_weights
            fold_mae.append(mean_absolute_error(y_val, fold_pred))
        return float(np.mean(fold_mae))
