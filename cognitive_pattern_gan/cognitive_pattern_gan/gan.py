"""
Pure-NumPy GAN implementation for cognitive activation pattern generation.

Implementation notes
--------------------
* Uses mini-batch SGD with Adam-style first-moment EMA for the generator and
  discriminator separately.
* Numerically stable BCE: ``log(σ(x)) = x − log(1 + exp(x))`` via
  ``log_sigmoid``.
* No external dependencies beyond the standard library — NumPy is imported
  lazily via ``_np()`` so the module can be imported without NumPy if only
  the type annotations are needed.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Lazy NumPy import (avoids hard dependency at import time)
# ---------------------------------------------------------------------------

def _np():  # type: ignore[return]
    import importlib
    return importlib.import_module("numpy")


# ---------------------------------------------------------------------------
# Low-level helpers (pure Python for portability)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _relu_grad(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def _bce(pred: float, label: float, eps: float = 1e-7) -> float:
    """Binary cross-entropy loss for a single prediction."""
    p = max(eps, min(1.0 - eps, pred))
    return -(label * math.log(p) + (1.0 - label) * math.log(1.0 - p))


# ---------------------------------------------------------------------------
# MLP (used by both Generator and Discriminator)
# ---------------------------------------------------------------------------

class _MLP:
    """Minimal multi-layer perceptron with ReLU hidden activations.

    The output activation is set by the caller:
    - Generator:      ``tanh`` — bounded output in (−1, 1)
    - Discriminator:  ``sigmoid`` — probability in (0, 1)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        output_activation: str = "sigmoid",
        seed: int = 0,
    ) -> None:
        rng = random.Random(seed)
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []
        self.output_activation = output_activation

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = math.sqrt(2.0 / (fan_in + fan_out))
            w = [
                [rng.gauss(0, scale) for _ in range(fan_in)]
                for _ in range(fan_out)
            ]
            b = [0.0] * fan_out
            self.weights.append(w)
            self.biases.append(b)

        # Adam first-moment accumulators (m) and second-moment (v)
        self.m_w: List[List[List[float]]] = [
            [[0.0] * len(row) for row in layer] for layer in self.weights
        ]
        self.v_w: List[List[List[float]]] = [
            [[0.0] * len(row) for row in layer] for layer in self.weights
        ]
        self.m_b: List[List[float]] = [
            [0.0] * len(b) for b in self.biases
        ]
        self.v_b: List[List[float]] = [
            [0.0] * len(b) for b in self.biases
        ]
        self.t = 0  # Adam step counter

    # -- forward pass --------------------------------------------------------

    def forward(
        self, x: List[float]
    ) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        """Return (output, pre_activations, activations_per_layer)."""
        pre_acts: List[List[float]] = []
        acts: List[List[float]] = [x]

        current = x
        for l_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            pre = [
                sum(W[j][i] * current[i] for i in range(len(current))) + b[j]
                for j in range(len(W))
            ]
            pre_acts.append(pre)
            is_last = l_idx == len(self.weights) - 1
            if is_last and self.output_activation == "sigmoid":
                out = [_sigmoid(z) for z in pre]
            elif is_last and self.output_activation == "tanh":
                out = [math.tanh(z) for z in pre]
            else:
                out = [_relu(z) for z in pre]
            acts.append(out)
            current = out

        return current, pre_acts, acts

    # -- backward pass -------------------------------------------------------

    def backward(
        self,
        pre_acts: List[List[float]],
        acts: List[List[float]],
        d_output: List[float],
    ) -> Tuple[List[List[List[float]]], List[List[float]], List[float]]:
        """Return (grad_weights, grad_biases, d_input)."""
        n_layers = len(self.weights)
        grad_w: List[List[List[float]]] = [
            [[0.0] * len(row) for row in layer] for layer in self.weights
        ]
        grad_b: List[List[float]] = [
            [0.0] * len(b) for b in self.biases
        ]

        delta = d_output

        for l in range(n_layers - 1, -1, -1):
            pre = pre_acts[l]
            inp = acts[l]
            is_last = l == n_layers - 1

            # Compute activation gradients.
            if is_last and self.output_activation == "sigmoid":
                act_grad = [_sigmoid(z) * (1.0 - _sigmoid(z)) for z in pre]
            elif is_last and self.output_activation == "tanh":
                act_grad = [1.0 - math.tanh(z) ** 2 for z in pre]
            else:
                act_grad = [_relu_grad(z) for z in pre]

            d_pre = [delta[j] * act_grad[j] for j in range(len(delta))]

            # Weight and bias gradients.
            for j in range(len(self.weights[l])):
                for i in range(len(inp)):
                    grad_w[l][j][i] = d_pre[j] * inp[i]
                grad_b[l][j] = d_pre[j]

            # Propagate to previous layer.
            W = self.weights[l]
            d_inp = [
                sum(W[j][i] * d_pre[j] for j in range(len(W)))
                for i in range(len(inp))
            ]
            delta = d_inp

        return grad_w, grad_b, delta

    # -- Adam update ---------------------------------------------------------

    def adam_update(
        self,
        grad_w: List[List[List[float]]],
        grad_b: List[List[float]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.t += 1
        bc1 = 1.0 - beta1 ** self.t
        bc2 = 1.0 - beta2 ** self.t

        for l in range(len(self.weights)):
            for j in range(len(self.weights[l])):
                for i in range(len(self.weights[l][j])):
                    g = grad_w[l][j][i]
                    self.m_w[l][j][i] = beta1 * self.m_w[l][j][i] + (1 - beta1) * g
                    self.v_w[l][j][i] = beta2 * self.v_w[l][j][i] + (1 - beta2) * g * g
                    m_hat = self.m_w[l][j][i] / bc1
                    v_hat = self.v_w[l][j][i] / bc2
                    self.weights[l][j][i] -= lr * m_hat / (math.sqrt(v_hat) + eps)
                g_b = grad_b[l][j]
                self.m_b[l][j] = beta1 * self.m_b[l][j] + (1 - beta1) * g_b
                self.v_b[l][j] = beta2 * self.v_b[l][j] + (1 - beta2) * g_b * g_b
                m_hat_b = self.m_b[l][j] / bc1
                v_hat_b = self.v_b[l][j] / bc2
                self.biases[l][j] -= lr * m_hat_b / (math.sqrt(v_hat_b) + eps)


# ---------------------------------------------------------------------------
# Generator and Discriminator
# ---------------------------------------------------------------------------

class Generator:
    """Maps latent noise vectors to synthetic activation patterns.

    Output activation: ``tanh``, bounded in (−1, 1).
    """

    def __init__(self, latent_dim: int, pattern_dim: int, seed: int = 1) -> None:
        hidden = max(latent_dim * 2, pattern_dim * 2)
        self._mlp = _MLP(
            [latent_dim, hidden, pattern_dim],
            output_activation="tanh",
            seed=seed,
        )
        self.latent_dim = latent_dim
        self.pattern_dim = pattern_dim

    def forward(
        self, z: List[float]
    ) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        return self._mlp.forward(z)

    def generate(self, z: List[float]) -> List[float]:
        out, _, _ = self._mlp.forward(z)
        return out

    def backward(
        self,
        pre_acts: List[List[float]],
        acts: List[List[float]],
        d_output: List[float],
    ) -> Tuple[List[List[List[float]]], List[List[float]], List[float]]:
        return self._mlp.backward(pre_acts, acts, d_output)

    def adam_update(self, gw, gb, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._mlp.adam_update(gw, gb, **kwargs)


class Discriminator:
    """Classifies patterns as real (1) or generated (0).

    Output activation: ``sigmoid``.
    """

    def __init__(self, pattern_dim: int, seed: int = 2) -> None:
        hidden = max(pattern_dim * 2, 16)
        self._mlp = _MLP(
            [pattern_dim, hidden, 1],
            output_activation="sigmoid",
            seed=seed,
        )
        self.pattern_dim = pattern_dim

    def forward(
        self, x: List[float]
    ) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        return self._mlp.forward(x)

    def score(self, x: List[float]) -> float:
        out, _, _ = self._mlp.forward(x)
        return out[0]

    def backward(
        self,
        pre_acts: List[List[float]],
        acts: List[List[float]],
        d_output: List[float],
    ) -> Tuple[List[List[List[float]]], List[List[float]], List[float]]:
        return self._mlp.backward(pre_acts, acts, d_output)

    def adam_update(self, gw, gb, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._mlp.adam_update(gw, gb, **kwargs)


# ---------------------------------------------------------------------------
# PatternGAN — top-level API
# ---------------------------------------------------------------------------

class PatternGAN:
    """Generator/Discriminator GAN for cognitive activation patterns.

    Parameters
    ----------
    pattern_dim : int
        Dimensionality of activation patterns (must match the seed dataset).
    latent_dim : int
        Dimensionality of the noise input to the Generator.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        pattern_dim: int = 5,
        latent_dim: int = 8,
        seed: int = 42,
    ) -> None:
        self.pattern_dim = pattern_dim
        self.latent_dim = latent_dim
        self._rng = random.Random(seed)
        self.generator = Generator(latent_dim, pattern_dim, seed=seed)
        self.discriminator = Discriminator(pattern_dim, seed=seed + 1)
        self.g_losses: List[float] = []
        self.d_losses: List[float] = []

    # -- noise sampling ------------------------------------------------------

    def _sample_noise(self, batch_size: int) -> List[List[float]]:
        return [
            [self._rng.gauss(0, 1) for _ in range(self.latent_dim)]
            for _ in range(batch_size)
        ]

    # -- training step -------------------------------------------------------

    def _train_discriminator(
        self,
        real_batch: List[List[float]],
        fake_batch: List[List[float]],
        lr: float,
    ) -> float:
        """One discriminator update; returns average BCE loss."""
        total_loss = 0.0

        for sample, label in (
            [(x, 1.0) for x in real_batch] + [(x, 0.0) for x in fake_batch]
        ):
            out, pre, acts = self.discriminator.forward(sample)
            pred = out[0]
            loss = _bce(pred, label)
            total_loss += loss

            # ∂BCE/∂pred
            eps = 1e-7
            d_pred = -(label / max(pred, eps) - (1 - label) / max(1 - pred, eps))
            gw, gb, _ = self.discriminator.backward(pre, acts, [d_pred])
            self.discriminator.adam_update(gw, gb, lr=lr)

        return total_loss / (len(real_batch) + len(fake_batch))

    def _train_generator(
        self,
        noise_batch: List[List[float]],
        lr: float,
    ) -> float:
        """One generator update; returns average fool-the-discriminator loss."""
        total_loss = 0.0

        for z in noise_batch:
            fake, g_pre, g_acts = self.generator.forward(z)
            d_out, d_pre, d_acts = self.discriminator.forward(fake)
            pred = d_out[0]
            # Generator wants discriminator to output 1 (think it's real).
            loss = _bce(pred, 1.0)
            total_loss += loss

            eps = 1e-7
            d_pred = -(1.0 / max(pred, eps))
            # Backprop through discriminator to get d_fake.
            _, _, d_fake = self.discriminator.backward(d_pre, d_acts, [d_pred])
            # Backprop through generator.
            gw, gb, _ = self.generator.backward(g_pre, g_acts, d_fake)
            self.generator.adam_update(gw, gb, lr=lr)

        return total_loss / len(noise_batch)

    def train(
        self,
        real_patterns: Sequence[List[float]],
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
    ) -> Tuple[List[float], List[float]]:
        """Train the GAN for ``epochs`` epochs.

        Returns (generator_losses, discriminator_losses).
        """
        real_list = [list(p) for p in real_patterns]
        n = len(real_list)
        if n == 0:
            raise ValueError("real_patterns must not be empty")

        for _ in range(epochs):
            # Sample a real mini-batch with replacement.
            real_batch = [
                real_list[self._rng.randint(0, n - 1)] for _ in range(batch_size)
            ]
            noise_batch = self._sample_noise(batch_size)
            fake_batch = [self.generator.generate(z) for z in noise_batch]

            d_loss = self._train_discriminator(real_batch, fake_batch, lr)
            g_loss = self._train_generator(noise_batch, lr)

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

        return self.g_losses, self.d_losses

    def generate(self, n: int = 1) -> List[List[float]]:
        """Generate ``n`` novel activation patterns from random noise."""
        return [
            self.generator.generate(z)
            for z in self._sample_noise(n)
        ]
