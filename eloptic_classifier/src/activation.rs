//! Open activation-function trait and built-in implementations.
//!
//! Unlike `ruvnet/ruv-FANN`'s closed `ActivationFunction` enum
//! (`src/activation.rs` lines 12-96), the [`Activation`] trait is a plain
//! Rust trait.  Any downstream crate can `impl Activation for MyType` and
//! pass it directly to [`DenseLayer`](crate::DenseLayer).

/// Open activation-function contract.
///
/// Implement this trait to create a custom activation that integrates
/// seamlessly with [`ElopticClassifier`](crate::ElopticClassifier).
pub trait Activation: Send + Sync + 'static {
    /// Apply the activation element-wise to a single scalar pre-activation.
    fn apply(&self, x: f64) -> f64;
    /// First derivative at `x` — used during backpropagation.
    fn derivative(&self, x: f64) -> f64;
    /// Human-readable name for diagnostics.
    fn name(&self) -> &'static str;
}

// ── Built-in implementations ──────────────────────────────────────────────────

/// Hyperbolic tangent: output ∈ (−1, 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

impl Activation for Tanh {
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        x.tanh()
    }
    #[inline(always)]
    fn derivative(&self, x: f64) -> f64 {
        let t = x.tanh();
        1.0 - t * t
    }
    fn name(&self) -> &'static str {
        "tanh"
    }
}

/// Rectified linear unit: max(0, x).
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;

impl Activation for ReLU {
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        x.max(0.0)
    }
    #[inline(always)]
    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
    fn name(&self) -> &'static str {
        "relu"
    }
}

/// Logistic sigmoid: output ∈ (0, 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline(always)]
    fn derivative(&self, x: f64) -> f64 {
        let s = self.apply(x);
        s * (1.0 - s)
    }
    fn name(&self) -> &'static str {
        "sigmoid"
    }
}

/// Leaky ReLU: max(alpha * x, x).
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    pub alpha: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

impl Activation for LeakyReLU {
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        if x >= 0.0 { x } else { self.alpha * x }
    }
    #[inline(always)]
    fn derivative(&self, x: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { self.alpha }
    }
    fn name(&self) -> &'static str {
        "leaky_relu"
    }
}

/// Softmax applied over a vector — stored as a unit struct; `apply` is
/// identity (softmax is computed over the full output vector in the
/// classifier).
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax;

impl Activation for Softmax {
    /// For scalar use during backprop the softmax derivative is approximated
    /// as `s * (1 − s)` (cross-entropy combined gradient).
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        // Scalar branch — full softmax is applied vector-wise in the
        // classifier.
        x.exp()
    }
    #[inline(always)]
    fn derivative(&self, x: f64) -> f64 {
        let s = x.exp();
        s * (1.0 - s)
    }
    fn name(&self) -> &'static str {
        "softmax"
    }
}

/// Apply vector-wise softmax to a mutable slice in place.
///
/// Numerically stable: subtracts the maximum value before exponentiation.
pub fn softmax_inplace(v: &mut [f64]) {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_positive() {
        assert_eq!(ReLU.apply(3.0), 3.0);
        assert_eq!(ReLU.apply(-2.0), 0.0);
        assert_eq!(ReLU.derivative(1.0), 1.0);
        assert_eq!(ReLU.derivative(-1.0), 0.0);
    }

    #[test]
    fn sigmoid_bounds() {
        assert!((Sigmoid.apply(100.0) - 1.0).abs() < 1e-6, "sigmoid(100) must be ≈ 1");
        assert!(Sigmoid.apply(-100.0) > 0.0);
        let mid = Sigmoid.apply(0.0);
        assert!((mid - 0.5).abs() < 1e-10);
    }

    #[test]
    fn tanh_antisymmetric() {
        assert!((Tanh.apply(1.0) + Tanh.apply(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn leaky_relu_negative_leak() {
        let act = LeakyReLU { alpha: 0.1 };
        assert!((act.apply(-5.0) - (-0.5)).abs() < 1e-12);
        assert_eq!(act.derivative(-1.0), 0.1);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }
}
