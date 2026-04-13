"""Threshold-rule learning for within-clan rank prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_CLAN_ORDER, DEFAULT_FEATURES
from .metrics import confusion_matrix_df


@dataclass(frozen=True)
class ThresholdRule:
    feature: str
    t1: float
    t2: float
    direction: str
    train_accuracy: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        scores = (x >= self.t1).astype(int) + (x >= self.t2).astype(int)
        if self.direction == "desc":
            scores = 2 - scores
        return scores.astype(int)


@dataclass
class ClanRankRuleModel:
    """Learn one interpretable threshold rule per clan."""

    feature_names: Sequence[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    classes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_CLAN_ORDER))
    rules_: dict[str, ThresholdRule] | None = None

    def fit(self, df_train: pd.DataFrame) -> dict[str, ThresholdRule]:
        rules: dict[str, ThresholdRule] = {}
        for clan in self.classes:
            clan_df = df_train.loc[df_train["clan"] == clan].reset_index(drop=True)
            rules[clan] = self._best_rule_for_clan(clan_df)
        self.rules_ = rules
        return rules

    def predict(self, df: pd.DataFrame, clan_labels: Sequence[str]) -> np.ndarray:
        if self.rules_ is None:
            raise RuntimeError("Rule model must be fitted before prediction.")
        preds: list[int] = []
        for idx, clan in enumerate(clan_labels):
            rule = self.rules_.get(clan)
            if rule is None:
                preds.append(1)
                continue
            value = float(df.iloc[idx][rule.feature])
            preds.append(int(rule.apply(np.array([value]))[0]))
        return np.array(preds, dtype=int)

    def predict_train(self, df_train: pd.DataFrame) -> np.ndarray:
        if self.rules_ is None:
            raise RuntimeError("Rule model must be fitted before prediction.")
        preds: list[np.ndarray] = []
        for clan in self.classes:
            clan_df = df_train.loc[df_train["clan"] == clan]
            rule = self.rules_[clan]
            clan_values = clan_df[rule.feature].to_numpy(dtype=float)
            preds.append(rule.apply(clan_values))
        return np.concatenate(preds)

    def training_report(self, df_train: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        y_true = []
        y_pred = []
        if self.rules_ is None:
            raise RuntimeError("Rule model must be fitted before evaluation.")
        for clan in self.classes:
            clan_df = df_train.loc[df_train["clan"] == clan]
            rule = self.rules_[clan]
            truth = clan_df["rank"].to_numpy(dtype=int)
            preds = rule.apply(clan_df[rule.feature].to_numpy(dtype=float))
            y_true.append(truth)
            y_pred.append(preds)
        y_true_arr = np.concatenate(y_true)
        y_pred_arr = np.concatenate(y_pred)
        accuracy = float(np.mean(y_true_arr == y_pred_arr))
        confusion = confusion_matrix_df(y_true_arr, y_pred_arr, labels=[0, 1, 2])
        return accuracy, confusion

    def rule_table(self) -> pd.DataFrame:
        if self.rules_ is None:
            raise RuntimeError("Rule model must be fitted before exporting rules.")
        rows = []
        for clan, rule in self.rules_.items():
            rows.append(
                {
                    "clan": clan,
                    "feature": rule.feature,
                    "direction": rule.direction,
                    "t1": rule.t1,
                    "t2": rule.t2,
                    "train_accuracy": rule.train_accuracy,
                }
            )
        return pd.DataFrame(rows)

    def _best_rule_for_clan(self, df_clan: pd.DataFrame) -> ThresholdRule:
        y = df_clan["rank"].to_numpy(dtype=int)
        best: ThresholdRule | None = None

        for feature in self.feature_names:
            x = df_clan[feature].to_numpy(dtype=float)
            unique_values = np.unique(np.sort(x))
            if unique_values.size < 3:
                continue

            midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
            for i in range(len(midpoints) - 1):
                t1 = float(midpoints[i])
                t2_candidates = midpoints[i + 1 :]

                ascending_predictions = (x[:, None] >= t1).astype(int) + (x[:, None] >= t2_candidates).astype(int)
                ascending_accuracy = (ascending_predictions == y[:, None]).mean(axis=0)
                j = int(np.argmax(ascending_accuracy))
                asc_rule = ThresholdRule(
                    feature=feature,
                    t1=t1,
                    t2=float(t2_candidates[j]),
                    direction="asc",
                    train_accuracy=float(ascending_accuracy[j]),
                )
                if best is None or asc_rule.train_accuracy > best.train_accuracy:
                    best = asc_rule

                descending_accuracy = (2 - ascending_predictions == y[:, None]).mean(axis=0)
                j_desc = int(np.argmax(descending_accuracy))
                desc_rule = ThresholdRule(
                    feature=feature,
                    t1=t1,
                    t2=float(t2_candidates[j_desc]),
                    direction="desc",
                    train_accuracy=float(descending_accuracy[j_desc]),
                )
                if best is None or desc_rule.train_accuracy > best.train_accuracy:
                    best = desc_rule

        if best is None:
            raise ValueError("No valid rank rule could be learned for the provided clan.")
        return best
