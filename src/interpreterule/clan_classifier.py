"""Interpretable 2D KNN classifier for clan prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_CLAN_ORDER, DEFAULT_CV_SEEDS, DEFAULT_FEATURES, DEFAULT_K_CANDIDATES
from .feature_selection import rank_features_by_fisher
from .metrics import confusion_matrix_df


@dataclass
class ClanClassificationResult:
    selected_features: list[str]
    best_k: int
    cv_summary: pd.DataFrame
    train_accuracy: float
    train_confusion: pd.DataFrame


@dataclass
class KNNClanClassifier:
    """Two-feature KNN classifier chosen by Fisher-score screening."""

    candidate_features: Sequence[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    classes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_CLAN_ORDER))
    k_candidates: Sequence[int] = field(default_factory=lambda: list(DEFAULT_K_CANDIDATES))
    cv_seeds: Sequence[int] = field(default_factory=lambda: list(DEFAULT_CV_SEEDS))

    selected_features_: list[str] | None = None
    best_k_: int | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    X_train_: np.ndarray | None = None
    y_train_: np.ndarray | None = None
    cv_summary_: pd.DataFrame | None = None

    def fit(self, df_train: pd.DataFrame) -> ClanClassificationResult:
        ranking = rank_features_by_fisher(
            df=df_train,
            feature_names=self.candidate_features,
            label_column="clan",
            classes=self.classes,
        )
        self.selected_features_ = ranking["feature"].head(2).tolist()

        X_raw = df_train[self.selected_features_].to_numpy(dtype=float)
        self.mean_ = X_raw.mean(axis=0)
        self.std_ = X_raw.std(axis=0) + 1e-12
        self.X_train_ = (X_raw - self.mean_) / self.std_
        self.y_train_ = df_train["clan"].to_numpy()

        cv_rows = []
        for k in self.k_candidates:
            mean_acc, std_acc = self._cv_accuracy_for_k(k)
            cv_rows.append({"k": int(k), "mean_accuracy": mean_acc, "std_accuracy": std_acc})
        self.cv_summary_ = (
            pd.DataFrame(cv_rows)
            .sort_values(["mean_accuracy", "k"], ascending=[False, True])
            .reset_index(drop=True)
        )
        self.best_k_ = int(self.cv_summary_.iloc[0]["k"])

        train_pred, _ = self.predict_with_confidence(df_train)
        train_accuracy = float(np.mean(train_pred == self.y_train_))
        train_confusion = confusion_matrix_df(self.y_train_, train_pred, self.classes)

        return ClanClassificationResult(
            selected_features=list(self.selected_features_),
            best_k=self.best_k_,
            cv_summary=self.cv_summary_.copy(),
            train_accuracy=train_accuracy,
            train_confusion=train_confusion,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.selected_features_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Classifier must be fitted before calling transform.")
        X = df[self.selected_features_].to_numpy(dtype=float)
        return (X - self.mean_) / self.std_

    def predict_with_confidence(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.best_k_ is None or self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Classifier must be fitted before prediction.")
        X_query = self.transform(df)
        return self._knn_predict(self.X_train_, self.y_train_, X_query, self.best_k_)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds, _ = self.predict_with_confidence(df)
        return preds

    def _cv_accuracy_for_k(self, k: int) -> tuple[float, float]:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Classifier must be fitted before cross-validation.")
        accuracies: list[float] = []
        for seed in self.cv_seeds:
            folds = self._build_stratified_folds(self.y_train_, n_splits=5, seed=seed)
            right = 0
            total = 0
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                preds, _ = self._knn_predict(
                    self.X_train_[train_idx],
                    self.y_train_[train_idx],
                    self.X_train_[val_idx],
                    k,
                )
                right += int(np.sum(preds == self.y_train_[val_idx]))
                total += len(val_idx)
            accuracies.append(right / max(total, 1))
        return float(np.mean(accuracies)), float(np.std(accuracies))

    def _knn_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions: list[str] = []
        confidence: list[float] = []
        class_index = {label: idx for idx, label in enumerate(self.classes)}

        for row in X_query:
            distances = np.sum((X_train - row) ** 2, axis=1)
            neighbour_idx = np.argpartition(distances, k)[:k]
            counts = {label: 0 for label in self.classes}
            distance_sums = {label: 0.0 for label in self.classes}

            for idx in neighbour_idx:
                label = y_train[idx]
                counts[label] += 1
                distance_sums[label] += float(distances[idx])

            best_key = None
            best_label = None
            for label in self.classes:
                key = (counts[label], -distance_sums[label], -class_index[label])
                if best_key is None or key > best_key:
                    best_key = key
                    best_label = label

            predictions.append(best_label)
            confidence.append(counts[best_label] / k)

        return np.array(predictions, dtype=object), np.array(confidence, dtype=float)

    def _build_stratified_folds(
        self,
        y: np.ndarray,
        n_splits: int,
        seed: int,
    ) -> list[np.ndarray]:
        rng = np.random.RandomState(seed)
        folds: list[list[int]] = [[] for _ in range(n_splits)]
        for label in self.classes:
            idx = np.where(y == label)[0]
            idx = rng.permutation(idx)
            parts = np.array_split(idx, n_splits)
            for fold_idx in range(n_splits):
                folds[fold_idx].extend(parts[fold_idx].tolist())
        return [np.array(fold, dtype=int) for fold in folds]
