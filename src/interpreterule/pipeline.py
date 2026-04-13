"""End-to-end orchestration for the refactored analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .clan_classifier import KNNClanClassifier
from .config import DEFAULT_CLAN_ORDER, DEFAULT_FEATURES
from .data import load_dataset, split_labeled_unlabeled
from .plots import (
    save_clan_separation_plot,
    save_correlation_heatmap,
    save_cv_curve,
    save_feature_vs_prowess_grid,
    save_prowess_diagnostics,
    save_rank_rule_boxplots,
    save_tournament_probabilities,
)
from .prowess import IntegerLinearFormulaModel
from .rank_rules import ClanRankRuleModel
from .tournament import estimate_peace_probability, estimate_tournament_probabilities


@dataclass
class PipelineArtifacts:
    predictions_path: Path
    metrics_path: Path
    rule_table_path: Path
    formula_table_path: Path


def run_full_pipeline(
    dataset_path: str | Path,
    output_dir: str | Path,
    include_extensions: bool = True,
) -> PipelineArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    data = load_dataset(dataset_path)
    split = split_labeled_unlabeled(data, label_column="clan")
    train_df = split.train.copy()
    test_df = split.test.copy()

    answers = data.copy()

    # Task 1 style diagnostics
    save_feature_vs_prowess_grid(train_df, DEFAULT_FEATURES, figures_dir / "feature_vs_prowess.png")
    save_correlation_heatmap(train_df, DEFAULT_FEATURES, figures_dir / "feature_correlation.png")

    # Task 2
    clan_model = KNNClanClassifier()
    clan_result = clan_model.fit(train_df)
    clan_test_pred, clan_test_conf = clan_model.predict_with_confidence(test_df)
    answers.loc[answers["clan"].isna(), "clan"] = clan_test_pred

    save_clan_separation_plot(
        df_train=train_df,
        feature_x=clan_result.selected_features[0],
        feature_y=clan_result.selected_features[1],
        clans=DEFAULT_CLAN_ORDER,
        output_path=figures_dir / "clan_separation.png",
    )
    save_cv_curve(clan_result.cv_summary, clan_result.best_k, figures_dir / "knn_cv_curve.png")

    # Task 3
    rank_model = ClanRankRuleModel()
    rank_model.fit(train_df)
    rank_accuracy, rank_confusion = rank_model.training_report(train_df)
    rule_table = rank_model.rule_table().sort_values("clan").reset_index(drop=True)
    predicted_ranks = rank_model.predict(test_df, answers.loc[answers["rank"].isna(), "clan"].tolist())
    answers.loc[answers["rank"].isna(), "rank"] = predicted_ranks
    save_rank_rule_boxplots(train_df, rule_table, figures_dir / "rank_rule_boxplots.png")

    # Task 4
    formula_model = IntegerLinearFormulaModel()
    formula_result = formula_model.fit(train_df)
    test_prowess_pred = formula_model.predict(test_df)
    answers.loc[answers["prowess"].isna(), "prowess"] = test_prowess_pred
    train_prowess_pred = formula_model.predict(train_df)
    formula_table = formula_model.coefficient_table()
    save_prowess_diagnostics(train_df, train_prowess_pred, DEFAULT_CLAN_ORDER, figures_dir / "prowess_diagnostics.png")

    extension_metrics: dict[str, object] = {}
    if include_extensions:
        predicted_test_df = answers.loc[answers["clan"].notna() & answers["rank"].notna() & answers["prowess"].notna()].iloc[len(train_df):].copy()
        tournament_probabilities = estimate_tournament_probabilities(
            train_df=train_df,
            predicted_df=predicted_test_df,
            clans=DEFAULT_CLAN_ORDER,
        )
        bloodbath_this_year, peace_over_38_years, peace_ci95 = estimate_peace_probability(
            tournament_probabilities,
            clans=DEFAULT_CLAN_ORDER,
        )
        save_tournament_probabilities(tournament_probabilities, DEFAULT_CLAN_ORDER, figures_dir / "tournament_probabilities.png")
        extension_metrics = {
            "tournament_win_probabilities": tournament_probabilities,
            "bloodbath_this_year": bloodbath_this_year,
            "peace_over_38_years": peace_over_38_years,
            "peace_over_38_years_ci95": peace_ci95,
        }

    predictions_path = output_dir / "answers_filled.csv"
    metrics_path = output_dir / "metrics.json"
    rule_table_path = output_dir / "rank_rules.csv"
    formula_table_path = output_dir / "prowess_formula.csv"

    answers.to_csv(predictions_path, index=False)
    rule_table.to_csv(rule_table_path, index=False)
    formula_table.to_csv(formula_table_path, index=False)

    metrics = {
        "dataset": {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_total": int(len(data)),
        },
        "clan_classification": {
            "selected_features": clan_result.selected_features,
            "best_k": clan_result.best_k,
            "cv_summary": clan_result.cv_summary.to_dict(orient="records"),
            "train_accuracy": clan_result.train_accuracy,
            "train_confusion": clan_result.train_confusion.to_dict(),
            "test_vote_share_mean": float(np.mean(clan_test_conf)),
            "test_vote_share_min": float(np.min(clan_test_conf)),
        },
        "rank_rule_learning": {
            "rules": rule_table.to_dict(orient="records"),
            "train_accuracy": rank_accuracy,
            "train_confusion": rank_confusion.to_dict(),
        },
        "prowess_formula": {
            "features": formula_result.features,
            "float_weights": formula_result.float_weights,
            "integer_weights": formula_result.integer_weights,
            "cv_mae": formula_result.cv_mae,
            "train_rmse": formula_result.train_rmse,
            "train_r2": formula_result.train_r2,
            "formula": formula_result.formula_string,
        },
        "extensions": extension_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return PipelineArtifacts(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        rule_table_path=rule_table_path,
        formula_table_path=formula_table_path,
    )
