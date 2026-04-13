"""Project-wide constants."""

from __future__ import annotations

DEFAULT_FEATURES = [
    "angst",
    "asceticism",
    "avoidance",
    "charring",
    "forgiveness",
    "merriment",
    "mood",
    "peril",
    "sliminess",
    "spike_rate",
]

DEFAULT_TARGETS = ["clan", "rank", "prowess"]

DEFAULT_CLAN_ORDER = ["haldane", "irvine", "lewis", "wheatley"]

DEFAULT_CANDIDATE_PROWESS_FEATURES = [
    "charring",
    "peril",
    "sliminess",
    "spike_rate",
    "merriment",
]

DEFAULT_K_CANDIDATES = [1, 3, 5, 7, 9]
DEFAULT_CV_SEEDS = [0, 1, 2, 3, 4]
