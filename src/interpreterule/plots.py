"""Plotting utilities for the refactored coursework analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import r_squared, root_mean_squared_error


def save_feature_vs_prowess_grid(df_train: pd.DataFrame, feature_names: Sequence[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y = df_train["prowess"].to_numpy(dtype=float)

    def fit_line_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        X = np.column_stack([x, np.ones_like(x)])
        identity = np.eye(2)
        identity[1, 1] = 0.0
        weights = np.linalg.solve(X.T @ X + 0.0 * identity, X.T @ y)
        return float(weights[0]), float(weights[1])

    fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharey=True)
    axes = axes.ravel()

    for ax, feature in zip(axes, feature_names):
        x = df_train[feature].to_numpy(dtype=float)
        ax.scatter(x, y, s=10, alpha=0.6)
        slope, intercept = fit_line_ols(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept)
        ax.set_title(feature, fontsize=9)

    axes[0].set_ylabel("prowess")
    axes[5].set_ylabel("prowess")
    fig.suptitle("Feature-to-prowess relationships in the labeled training set", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df_train: pd.DataFrame, feature_names: Sequence[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corr = df_train[list(feature_names)].corr().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(corr, vmin=-1, vmax=1, origin="lower", aspect="equal")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("Feature correlation matrix")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_clan_separation_plot(
    df_train: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    clans: Sequence[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, clan in enumerate(clans):
        subset = df_train.loc[df_train["clan"] == clan]
        ax.scatter(subset[feature_x], subset[feature_y], s=20, alpha=0.7, label=clan, color=palette[idx % len(palette)])
        ax.plot(subset[feature_x].mean(), subset[feature_y].mean(), marker="+", ms=12, mew=2, color=palette[idx % len(palette)])

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title("Clan separation on the two selected features")
    ax.legend(title="clan", frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cv_curve(cv_summary: pd.DataFrame, best_k: int, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = cv_summary.sort_values("k")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.errorbar(ordered["k"], ordered["mean_accuracy"], yerr=ordered["std_accuracy"], fmt="-o", capsize=3)
    ax.axvline(best_k, linestyle="--", linewidth=1)
    ax.set_xlabel("k")
    ax.set_ylabel("CV accuracy")
    ax.set_title("Repeated stratified 5x5 CV for KNN")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rank_rule_boxplots(df_train: pd.DataFrame, rule_table: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, rule_table.iterrows()):
        clan = row["clan"]
        feature = row["feature"]
        subset = df_train.loc[df_train["clan"] == clan]
        groups = [subset.loc[subset["rank"] == rank, feature].to_numpy(dtype=float) for rank in [0, 1, 2]]
        ax.boxplot(groups, tick_labels=["0", "1", "2"], showfliers=False, widths=0.6)
        ax.hlines([row["t1"], row["t2"]], xmin=0.6, xmax=3.4, linestyles="dashed")
        ax.set_xlabel("rank")
        ax.set_ylabel(feature)
        ax.set_title(f"{clan} | {feature} | {row['direction']} | acc={row['train_accuracy']:.2f}")

    fig.suptitle("Per-clan threshold rules for rank", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prowess_diagnostics(
    df_train: pd.DataFrame,
    predictions: np.ndarray,
    clans: Sequence[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = df_train["prowess"].to_numpy(dtype=float)
    residuals = y_true - predictions

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(y_true, predictions, s=12, alpha=0.6)
    min_val = float(min(y_true.min(), predictions.min()))
    max_val = float(max(y_true.max(), predictions.max()))
    axes[0].plot([min_val, max_val], [min_val, max_val])
    axes[0].set_xlabel("true prowess")
    axes[0].set_ylabel("predicted prowess")
    axes[0].set_title(
        f"Parity plot | RMSE={root_mean_squared_error(y_true, predictions):.2f} | R²={r_squared(y_true, predictions):.4f}"
    )

    grouped_residuals = [residuals[df_train["clan"].to_numpy() == clan] for clan in clans]
    axes[1].boxplot(grouped_residuals, tick_labels=list(clans), showfliers=False)
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set_title("Residuals by clan")
    axes[1].set_ylabel("y - ŷ")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tournament_probabilities(probabilities: dict[str, float], clans: Sequence[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [clan.capitalize() for clan in clans] + ["Draw"]
    p = np.array([probabilities.get(clan, 0.0) for clan in clans] + [probabilities.get("Draw", 0.0)], dtype=float)
    n = 10_000
    se = np.sqrt(p * (1 - p) / n)

    order = list(np.argsort(-p[:-1])) + [len(p) - 1]
    labels = [labels[i] for i in order]
    p = p[order]
    se = se[order]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(labels, p)
    ax.errorbar(range(len(p)), p, yerr=1.96 * se, fmt="none", capsize=3)
    ax.set_ylim(0, max(0.05, p.max() * 1.15))
    ax.set_ylabel("Win probability")
    ax.set_title("Tournament win probability (Monte Carlo)")
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    for bar, value in zip(bars, p):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ax.get_ylim()[1] * 0.01, f"{value:.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
