"""Optional downstream simulation based on predicted clan, rank, and prowess."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class TournamentSimulationResult:
    win_probabilities: dict[str, float]
    draws_probability: float
    bloodbath_this_year: float
    peace_over_38_years: float
    peace_over_38_years_ci95: float


def estimate_tournament_probabilities(
    train_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    clans: Sequence[str],
    seed: int = 1,
    n_simulations: int = 10_000,
) -> dict[str, float]:
    """Simulate the synthetic tournament used in the coursework."""
    rng = np.random.RandomState(seed)

    player_pool: dict[tuple[str, int], np.ndarray] = {}
    for clan in clans:
        for rank in (0, 1, 2):
            pool = predicted_df.loc[
                (predicted_df["clan"] == clan) & (predicted_df["rank"] == rank),
                "prowess",
            ].to_numpy(dtype=float)
            if pool.size == 0:
                pool = train_df.loc[
                    (train_df["clan"] == clan) & (train_df["rank"] == rank),
                    "prowess",
                ].to_numpy(dtype=float)
            if pool.size == 0:
                pool = train_df.loc[train_df["rank"] == rank, "prowess"].to_numpy(dtype=float)
            if pool.size == 0:
                pool = np.array([float(train_df["prowess"].mean())], dtype=float)
            player_pool[(clan, rank)] = pool

    def match_winner(clan_a: str, clan_b: str, rank: int) -> str:
        prowess_a = player_pool[(clan_a, rank)]
        prowess_b = player_pool[(clan_b, rank)]
        a = prowess_a[rng.randint(0, len(prowess_a))]
        b = prowess_b[rng.randint(0, len(prowess_b))]
        if a == b:
            return clan_a if rng.rand() < 0.5 else clan_b
        return clan_a if a > b else clan_b

    def one_round(active_clans: Sequence[str]) -> dict[str, int]:
        wins = {clan: 0 for clan in active_clans}
        n = len(active_clans)
        for i in range(n):
            for j in range(i + 1, n):
                clan_a, clan_b = active_clans[i], active_clans[j]
                for rank in (0, 1, 2):
                    wins[match_winner(clan_a, clan_b, rank)] += 1
        return wins

    def tournament(active_clans: Sequence[str], rounds_left: int = 5) -> str:
        if rounds_left == 0:
            return "Draw"
        if len(active_clans) == 1:
            return active_clans[0]
        wins = one_round(active_clans)
        best = max(wins.values())
        leaders = [clan for clan, count in wins.items() if count == best]
        return leaders[0] if len(leaders) == 1 else tournament(leaders, rounds_left - 1)

    counts: dict[str, int] = {}
    for _ in range(n_simulations):
        winner = tournament(list(clans))
        counts[winner] = counts.get(winner, 0) + 1

    probabilities = {key: value / n_simulations for key, value in counts.items()}
    for clan in clans:
        probabilities.setdefault(clan, 0.0)
    probabilities.setdefault("Draw", 0.0)
    return probabilities


def estimate_peace_probability(
    win_probabilities: dict[str, float],
    clans: Sequence[str],
    seed: int = 7,
    years: int = 38,
    n_simulations: int = 100_000,
) -> tuple[float, float, float]:
    """Estimate bloodbath and peace probabilities under the assignment rules."""
    outcomes = list(clans) + ["Draw"]
    probabilities = np.array([win_probabilities.get(clan, 0.0) for clan in clans] + [win_probabilities.get("Draw", 0.0)], dtype=float)
    probabilities = probabilities / probabilities.sum()

    rng = np.random.RandomState(seed)
    p_draw = float(probabilities[-1])
    bloodbath_this_year = float(np.sum(probabilities[:-1] ** 5) + p_draw ** 2)

    def simulate_years() -> str:
        last_winner = None
        win_streak = 0
        draw_streak = 0
        for _ in range(years):
            outcome = rng.choice(outcomes, p=probabilities)
            if outcome == "Draw":
                draw_streak += 1
                last_winner = None
                win_streak = 0
                if draw_streak == 2:
                    return "bloodbath"
            else:
                draw_streak = 0
                if outcome == last_winner:
                    win_streak += 1
                else:
                    last_winner = outcome
                    win_streak = 1
                if win_streak == 5:
                    return "bloodbath"
        return "peace"

    peace_count = sum(simulate_years() == "peace" for _ in range(n_simulations))
    peace_probability = peace_count / n_simulations
    se = float(np.sqrt(peace_probability * (1 - peace_probability) / n_simulations))
    return bloodbath_this_year, float(peace_probability), float(1.96 * se)
