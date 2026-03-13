"""Activation pattern types and a small real-pattern seed dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


PatternLabel = str  # e.g. "focused", "creative", "fatigued", …


@dataclass
class ActivationPattern:
    """A named cognitive activation pattern — a normalised float vector.

    The vector represents relative firing strengths across a bank of
    neural frequency channels (theta, alpha, beta, gamma, …).
    """

    label: PatternLabel
    values: List[float]

    def __post_init__(self) -> None:
        n = sum(v * v for v in self.values) ** 0.5
        if n > 0:
            self.values = [v / n for v in self.values]

    def dim(self) -> int:
        return len(self.values)


def _pattern(label: str, vals: list[float]) -> ActivationPattern:
    return ActivationPattern(label=label, values=vals)


# Seed set of eight real cognitive templates based on EEG literature
# frequency-band activation profiles.  These are starting points for the GAN
# training data; the GAN generalises beyond these fixed patterns.
SEED_PATTERNS: List[ActivationPattern] = [
    _pattern("focused",    [0.1, 0.2, 0.8, 0.6, 0.3]),
    _pattern("creative",   [0.7, 0.5, 0.3, 0.4, 0.2]),
    _pattern("relaxed",    [0.6, 0.8, 0.2, 0.1, 0.1]),
    _pattern("meditative", [0.9, 0.3, 0.1, 0.1, 0.05]),
    _pattern("anxious",    [0.2, 0.1, 0.9, 0.8, 0.7]),
    _pattern("fatigued",   [0.5, 0.4, 0.2, 0.1, 0.3]),
    _pattern("alert",      [0.1, 0.3, 0.7, 0.9, 0.5]),
    _pattern("dreaming",   [0.8, 0.6, 0.1, 0.05, 0.2]),
]
