"""
cognitive_pattern_gan
=====================
GAN for generating synthetic cognitive activation patterns.

Fixes ruvnet/ruv-FANN flaw #9: static set of 7 hardcoded cognitive patterns
with no generative capability.  This module trains a real Generator /
Discriminator pair using mini-batch stochastic gradient descent with
numerically-stable binary cross-entropy.

Architecture
------------
- ``PatternGAN``  : top-level API (train + generate)
- ``Generator``   : latent-noise → activation-pattern MLP
- ``Discriminator``: activation-pattern → real/fake probability MLP

All computation uses pure NumPy — no deep-learning framework required, which
keeps the dependency surface minimal while keeping the implementation real
(no mocks, no placeholders).
"""

from .gan import Discriminator, Generator, PatternGAN
from .patterns import ActivationPattern, PatternLabel

__all__ = [
    "PatternGAN",
    "Generator",
    "Discriminator",
    "ActivationPattern",
    "PatternLabel",
]
