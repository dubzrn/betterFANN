"""
Performance benchmarks for cognitive_pattern_gan.

Measures wall-clock time for:
  1. Generator forward pass (single and batch)
  2. Discriminator forward pass
  3. GAN training loop throughput (epochs/s and patterns/s)
"""

import sys
import time

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])  # ensure package importable

from cognitive_pattern_gan.gan import Generator, Discriminator, PatternGAN
from cognitive_pattern_gan.patterns import SEED_PATTERNS


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(label: str, fn, n: int = 1000):
    """Run `fn` `n` times and report throughput."""
    # Warm-up
    for _ in range(min(n // 10, 100)):
        fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    ops_per_sec = n / elapsed
    us_per_op = elapsed / n * 1_000_000
    print(f"  {label:<45s}  {ops_per_sec:>10,.0f} ops/s  ({us_per_op:.2f} µs/op)")
    return ops_per_sec, us_per_op


# ── benchmark bodies ─────────────────────────────────────────────────────────

def bench_generator_forward():
    """Single-sample generator forward pass."""
    latent_dim, pattern_dim = 8, 5
    gen = Generator(latent_dim=latent_dim, pattern_dim=pattern_dim)
    noise = [0.1] * latent_dim
    _run("Generator.generate(single)", lambda: gen.generate(noise))


def bench_discriminator_forward():
    """Single-sample discriminator forward pass."""
    pattern_dim = 5
    disc = Discriminator(pattern_dim=pattern_dim)
    sample = [0.3] * pattern_dim
    _run("Discriminator.score(single)", lambda: disc.score(sample))


def bench_gan_generate_batch():
    """Generating a batch of 16 patterns."""
    gan = PatternGAN(pattern_dim=5, latent_dim=8)
    _run(
        "PatternGAN.generate(16)",
        lambda: gan.generate(16),
        n=500,
    )


def bench_gan_training():
    """GAN training loop: 10 epochs over the seed patterns."""
    real_patterns = [p.values for p in SEED_PATTERNS]
    pattern_dim = len(real_patterns[0])

    def _make_and_train():
        gan = PatternGAN(pattern_dim=pattern_dim, latent_dim=8)
        gan.train(real_patterns, epochs=10, batch_size=4, lr=0.001)

    ops_per_sec, us_per_op = _run(
        "PatternGAN.train(10 epochs, 4 batch)",
        _make_and_train,
        n=50,
    )
    print(f"    → {ops_per_sec * 10:,.0f} epochs/s")


# ── summary ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("cognitive_pattern_gan — performance benchmarks")
    print("=" * 70)
    print()

    bench_generator_forward()
    bench_discriminator_forward()
    bench_gan_generate_batch()
    bench_gan_training()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
