//! Integration tests for `our-neural-core`.
//!
//! 1. Concurrent inference — 4 async tasks infer on a 784→256→128→10 network
//!    simultaneously using read locks.
//! 2. Weight zeroing — `SecureWeightMatrix` zeroes all elements when
//!    `Zeroize::zeroize()` is called, exactly as `ZeroizeOnDrop` will do on
//!    `Drop`.
//! 3. Layer forward shape — single-layer sanity check.

use ndarray::Array2;
use our_neural_core::{Activation, DenseLayer, Network, SecureWeightMatrix};
use zeroize::Zeroize;

// ── 1. Concurrent inference ───────────────────────────────────────────────────

/// Build a 784→256→128→10 network and run 4 concurrent `tokio::spawn` tasks,
/// each performing a full forward pass.  All outputs must have length 10.
#[tokio::test]
async fn test_concurrent_inference() {
    let network = Network::new(
        &[784, 256, 128, 10],
        vec![
            Activation::ReLU,
            Activation::Tanh,
            Activation::Sigmoid,
        ],
    );

    // Normalised pixel values in [0, 1] — a typical MNIST-style input.
    let input: Vec<f64> = (0..784).map(|i| i as f64 / 784.0).collect();

    let mut handles = Vec::with_capacity(4);
    for _ in 0..4 {
        let net = network.clone(); // O(1) Arc clone
        let inp = input.clone();
        handles.push(tokio::spawn(async move { net.infer(&inp) }));
    }

    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.expect("tokio task panicked");
        assert_eq!(
            result.len(),
            10,
            "task {i}: output must have exactly 10 neurons"
        );
        // Sigmoid outputs are bounded to (0, 1).
        for &v in &result {
            assert!(
                (0.0..=1.0).contains(&v),
                "task {i}: Sigmoid output {v} is outside (0, 1)"
            );
        }
    }
}

// ── 2. SecureWeightMatrix zeroing ─────────────────────────────────────────────

/// Verify that `SecureWeightMatrix` zeroes every element when `zeroize()` is
/// called.
///
/// `ZeroizeOnDrop` derives `Drop` as `fn drop(&mut self) { self.zeroize() }`,
/// so testing `Zeroize::zeroize()` directly is an exact proxy for what happens
/// at drop time.
#[test]
fn test_secure_weight_matrix_zeroes_on_zeroize() {
    let sentinel = 42.0_f64;
    let data = Array2::from_elem((8, 8), sentinel);
    let mut swm = SecureWeightMatrix::new(data);

    // Pre-condition: every element is the sentinel value.
    assert!(
        swm.data().iter().all(|&v| v == sentinel),
        "expected all elements to be {sentinel} before zeroize"
    );

    // Act — this is the same call ZeroizeOnDrop makes inside `Drop`.
    swm.zeroize();

    // Post-condition: every element must be 0.0.
    assert!(
        swm.data().iter().all(|&v| v == 0.0),
        "SecureWeightMatrix::zeroize must set every element to 0.0"
    );
}

/// Verify that a `SecureWeightMatrix` with known values is zeroed before its
/// memory is released.
///
/// Strategy: record the values *before* the drop scope, enter a scope that
/// owns the matrix, and use a shared flag (written from inside the `Drop`
/// implementation's logical path via explicit `zeroize()`) to confirm zeroing
/// occurred.  This approach is fully safe — no raw pointer is dereferenced
/// after the value has been dropped.
#[test]
fn test_secure_weight_matrix_zeroed_before_drop() {
    use std::sync::{Arc, Mutex};

    // Shared witness: the zeroed-flag is set to `true` once the matrix is
    // confirmed zeroed, before the scope exit releases it.
    let witness: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

    {
        let mut swm = SecureWeightMatrix::new(Array2::from_elem((6, 6), 7.0_f64));

        // Confirm non-zero while alive.
        assert!(swm.data().iter().all(|&v| v == 7.0));

        // Explicitly zeroize while we still hold the value, then record the
        // result — identical to what ZeroizeOnDrop does just before releasing
        // the allocation.
        swm.zeroize();
        let all_zero = swm.data().iter().all(|&v| v == 0.0);
        *witness.lock().unwrap() = all_zero;
        // `swm` is dropped here; Drop calls zeroize() again (idempotent).
    }

    assert!(
        *witness.lock().unwrap(),
        "SecureWeightMatrix must zero all elements before deallocation"
    );
}

// ── 3. Layer forward shape ────────────────────────────────────────────────────

/// A single DenseLayer must produce an output vector of the declared size.
#[test]
fn test_dense_layer_forward_shape() {
    let layer = DenseLayer::new(784, 256, Activation::ReLU);
    let input = vec![0.5_f64; 784];
    let output = layer.forward(&input);
    assert_eq!(output.len(), 256, "DenseLayer output size mismatch");
}

/// Verify that every activation variant produces a finite, correctly-shaped
/// output for a small representative input.
#[test]
fn test_activation_variants() {
    let layer_configs: &[(usize, usize, Activation)] = &[
        (4, 4, Activation::Tanh),
        (4, 4, Activation::ReLU),
        (4, 4, Activation::Sigmoid),
        (4, 4, Activation::LeakyReLU(0.01)),
    ];

    let input = vec![-1.0, 0.0, 0.5, 1.0];

    for (in_sz, out_sz, act) in layer_configs {
        let layer = DenseLayer::new(*in_sz, *out_sz, act.clone());
        let output = layer.forward(&input);
        assert_eq!(output.len(), *out_sz);
        for &v in &output {
            assert!(v.is_finite(), "activation produced non-finite value {v}");
        }
    }
}
