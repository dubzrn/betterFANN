//! Integration tests for `eloptic_classifier`.
//!
//! 1. Custom activation — a user-defined struct implementing `Activation` works
//!    end-to-end in a classifier.
//! 2. SIMD-optimised dot product — verifies numerical correctness.
//! 3. Memory safety — `SecureWeights::zeroize()` zeros all elements.
//! 4. XOR convergence — SGD reduces MSE on the XOR problem.

use std::sync::Arc;
use eloptic_classifier::{Activation, ElopticClassifier, SecureWeights, Sigmoid, ReLU};

// ── 1. Custom activation ──────────────────────────────────────────────────────

/// A user-defined SWISH activation: x * sigmoid(x).
struct Swish;

impl Activation for Swish {
    fn apply(&self, x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }
    fn derivative(&self, x: f64) -> f64 {
        let s = 1.0 / (1.0 + (-x).exp());
        s + x * s * (1.0 - s)
    }
    fn name(&self) -> &'static str {
        "swish"
    }
}

#[test]
fn custom_activation_compiles_and_runs() {
    let clf = ElopticClassifier::new(vec![
        (4, 8, Arc::new(Swish) as Arc<dyn Activation>),
        (8, 2, Arc::new(Swish)),
    ]);
    let out = clf.forward(&[0.5, -0.5, 1.0, -1.0]);
    assert_eq!(out.len(), 2);
    for &v in &out {
        assert!(v.is_finite(), "Swish output must be finite");
    }
}

// ── 2. Dot product correctness ────────────────────────────────────────────────

#[test]
fn optimised_dot_product_is_correct() {
    // 17 elements: tests both the unrolled 4-wide chunks (4*4=16) and the 1-element tail.
    let n = 17usize;
    let weights: Vec<f64> = (1..=n as i64).map(|x| x as f64).collect();
    let inputs: Vec<f64> = vec![1.0; n];

    let sw = SecureWeights::from_vec(weights.clone());
    let expected: f64 = weights.iter().sum();
    let got = sw.dot(&inputs);
    assert!((got - expected).abs() < 1e-10, "dot product mismatch: {got} vs {expected}");
}

// ── 3. Memory safety ──────────────────────────────────────────────────────────

#[test]
fn secure_weights_zeroed_after_zeroize() {
    use zeroize::Zeroize;
    let mut sw = SecureWeights::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    sw.zeroize();
    assert!(sw.as_slice().iter().all(|&v| v == 0.0), "all weights must be zero after zeroize()");
}

// ── 4. XOR convergence ────────────────────────────────────────────────────────

#[test]
fn xor_loss_decreases_over_training() {
    let data: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let mut clf = ElopticClassifier::new(vec![
        (2, 8, Arc::new(ReLU) as Arc<dyn Activation>),
        (8, 1, Arc::new(Sigmoid)),
    ]);

    // Measure initial loss before any training.
    let initial_loss: f64 = {
        let total: f64 = data.iter().map(|(x, y)| {
            let out = clf.forward(x);
            (out[0] - y[0]).powi(2)
        }).sum();
        total / data.len() as f64
    };

    let final_loss = clf.train(&data, 300, 0.1);

    assert!(
        final_loss < initial_loss,
        "training must reduce loss: initial={initial_loss:.4} final={final_loss:.4}"
    );
}
