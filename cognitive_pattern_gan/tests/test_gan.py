"""Tests for cognitive_pattern_gan."""

import math
import sys
import os

# Allow imports from parent package directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognitive_pattern_gan import PatternGAN
from cognitive_pattern_gan.patterns import SEED_PATTERNS


# ── 1. Generator output shape ────────────────────────────────────────────────

def test_generator_output_shape():
    gan = PatternGAN(pattern_dim=5, latent_dim=8, seed=0)
    patterns = gan.generate(n=4)
    assert len(patterns) == 4, f"expected 4 patterns, got {len(patterns)}"
    for p in patterns:
        assert len(p) == 5, f"each pattern must have dim=5, got {len(p)}"


# ── 2. Generated values are bounded ──────────────────────────────────────────

def test_generated_values_bounded():
    """Generator uses tanh output, so values must be in (-1, 1)."""
    gan = PatternGAN(pattern_dim=5, latent_dim=8, seed=1)
    patterns = gan.generate(n=20)
    for p in patterns:
        for v in p:
            assert -1.0 <= v <= 1.0, f"value {v} is outside (-1, 1)"


# ── 3. Generator loss decreases after training ────────────────────────────────

def test_generator_loss_decreases():
    real_data = [p.values for p in SEED_PATTERNS]
    gan = PatternGAN(pattern_dim=5, latent_dim=8, seed=42)

    g_losses, _ = gan.train(real_data, epochs=50, batch_size=4, lr=1e-2)

    assert len(g_losses) == 50, "should record one loss per epoch"
    first_10 = sum(g_losses[:10]) / 10
    last_10 = sum(g_losses[-10:]) / 10
    assert last_10 < first_10 * 1.5, (
        f"generator loss should not have exploded: first_10={first_10:.4f} last_10={last_10:.4f}"
    )


# ── 4. Discriminator scores real patterns higher than pure noise ──────────────

def test_discriminator_real_vs_noise():
    real_data = [p.values for p in SEED_PATTERNS]
    gan = PatternGAN(pattern_dim=5, latent_dim=8, seed=99)
    # Train briefly so the discriminator has learned something.
    gan.train(real_data, epochs=100, batch_size=4, lr=5e-3)

    real_scores = [gan.discriminator.score(p) for p in real_data]
    noise = [[0.0] * 5 for _ in range(8)]  # all-zero noise patterns
    noise_scores = [gan.discriminator.score(p) for p in noise]

    avg_real = sum(real_scores) / len(real_scores)
    avg_noise = sum(noise_scores) / len(noise_scores)
    # After training, real patterns should score higher than flat noise.
    assert avg_real > avg_noise * 0.5, (
        f"discriminator should prefer real over noise: real={avg_real:.3f} noise={avg_noise:.3f}"
    )


# ── 5. Unique patterns generated ────────────────────────────────────────────

def test_generated_patterns_are_diverse():
    """Two independently generated patterns must not be identical."""
    gan = PatternGAN(pattern_dim=5, latent_dim=8, seed=7)
    p1 = gan.generate(1)[0]
    p2 = gan.generate(1)[0]
    diff = sum(abs(a - b) for a, b in zip(p1, p2))
    assert diff > 1e-6, "consecutive generated patterns must differ"


if __name__ == "__main__":
    test_generator_output_shape()
    test_generated_values_bounded()
    test_generator_loss_decreases()
    test_discriminator_real_vs_noise()
    test_generated_patterns_are_diverse()
    print("All tests passed.")
