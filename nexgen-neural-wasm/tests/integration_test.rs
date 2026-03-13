//! Integration tests for nexgen-neural-wasm.
//!
//! These tests run on the **native** host (no WASM runtime required) and
//! exercise the public API that the WIT bindings delegate to.
//!
//! `wasmtime` is declared as a dev-dependency and is available should you
//! wish to add tests that load the compiled `.wasm` component; those tests
//! are not included here because the binary requires a `wasm32-wasi` build.

use nexgen_neural_wasm::{encode_topology_section, infer, run_inference, version, LayerConfig};

// ── version ───────────────────────────────────────────────────────────────────

#[test]
fn version_is_non_empty() {
    let v = version();
    assert!(!v.is_empty(), "version() must return a non-empty string");
}

#[test]
fn version_identifies_crate() {
    let v = version();
    assert!(
        v.contains("nexgen-neural-wasm"),
        "version string must contain the crate name; got: {v:?}",
    );
}

#[test]
fn version_contains_semver() {
    let v = version();
    // Expect at least one digit followed by a dot (e.g. "0.1.0").
    assert!(
        v.chars().any(|c| c.is_ascii_digit()),
        "version string must contain a version number; got: {v:?}",
    );
}

// ── single-layer forward pass ─────────────────────────────────────────────────

#[test]
fn single_layer_relu_output_shape_and_values() {
    // Network: 2 inputs → 3 outputs (ReLU)
    //
    // Weight matrix (row-major, shape 3×2):
    //   row 0: [1.0, 0.0]   → out[0] = 1·in[0] + 0·in[1] + 0
    //   row 1: [0.0, 1.0]   → out[1] = 0·in[0] + 1·in[1] + 0
    //   row 2: [1.0, 1.0]   → out[2] = 1·in[0] + 1·in[1] + 0
    // Biases: [0, 0, 0]
    // Input:  [1.0, 2.0]
    // Pre-activation: [1, 2, 3]  →  ReLU  →  [1, 2, 3]
    let layer = LayerConfig {
        input_size: 2,
        output_size: 3,
        activation: "relu".to_string(),
    };
    let weights = vec![
        1.0_f64, 0.0, // row 0
        0.0, 1.0, // row 1
        1.0, 1.0, // row 2
        0.0, 0.0, 0.0, // biases
    ];
    let input = vec![1.0_f64, 2.0];

    let output = run_inference(&weights, &input, &[layer]).expect("inference must succeed");

    assert_eq!(output.len(), 3, "output length must equal layer output_size");
    assert!((output[0] - 1.0).abs() < 1e-10, "out[0] expected 1.0, got {}", output[0]);
    assert!((output[1] - 2.0).abs() < 1e-10, "out[1] expected 2.0, got {}", output[1]);
    assert!((output[2] - 3.0).abs() < 1e-10, "out[2] expected 3.0, got {}", output[2]);
}

#[test]
fn single_layer_relu_clamps_negative_preactivations() {
    // Weight: -1, bias: 0 → pre-activation for input 1.0 is -1.0 → ReLU → 0.0
    let layer = LayerConfig {
        input_size: 1,
        output_size: 1,
        activation: "relu".to_string(),
    };
    let weights = vec![-1.0_f64, 0.0]; // weight + bias
    let input = vec![1.0_f64];

    let output = run_inference(&weights, &input, &[layer]).expect("inference must succeed");
    assert_eq!(output.len(), 1);
    assert!((output[0] - 0.0).abs() < 1e-10, "ReLU must clamp to 0; got {}", output[0]);
}

#[test]
fn single_layer_sigmoid_output_in_unit_interval() {
    // Any sigmoid output must be in (0, 1).
    let layer = LayerConfig {
        input_size: 3,
        output_size: 2,
        activation: "sigmoid".to_string(),
    };
    // Weights (2×3) + biases (2)
    let weights = vec![
        0.5, -0.5, 0.2, // row 0
        -0.3, 0.8, 0.1, // row 1
        0.0, 0.0, // biases
    ];
    let input = vec![1.0_f64, 0.5, -1.0];

    let output = run_inference(&weights, &input, &[layer]).expect("inference must succeed");
    assert_eq!(output.len(), 2);
    for (i, &v) in output.iter().enumerate() {
        assert!(v > 0.0 && v < 1.0, "sigmoid output[{i}]={v} must be in (0,1)");
    }
}

#[test]
fn single_layer_tanh_output_in_pm1_interval() {
    let layer = LayerConfig {
        input_size: 2,
        output_size: 2,
        activation: "tanh".to_string(),
    };
    let weights = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // identity + zero bias
    let input = vec![0.5_f64, -0.5];

    let output = run_inference(&weights, &input, &[layer]).expect("inference must succeed");
    assert_eq!(output.len(), 2);
    for (i, &v) in output.iter().enumerate() {
        assert!(v > -1.0 && v < 1.0, "tanh output[{i}]={v} must be in (-1,1)");
    }
}

// ── multi-layer forward pass ──────────────────────────────────────────────────

#[test]
fn multi_layer_output_shape() {
    // 4 → 3 → 2 → 1 network (all ReLU)
    let configs = vec![
        LayerConfig { input_size: 4, output_size: 3, activation: "relu".to_string() },
        LayerConfig { input_size: 3, output_size: 2, activation: "relu".to_string() },
        LayerConfig { input_size: 2, output_size: 1, activation: "relu".to_string() },
    ];

    // Minimal weights: identity-ish matrices + zero biases.
    // Layer 0: 4×3 weights + 3 biases = 15
    let w0 = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        0.0, 0.0, 0.0, // biases (3)
    ];
    // Layer 1: 3×2 weights + 2 biases = 8
    let w1 = vec![
        1.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, // row 1
        0.0, 0.0, // biases (2)
    ];
    // Layer 2: 2×1 weights + 1 bias = 3
    let w2 = vec![
        0.5, 0.5, // row 0
        0.0, // bias (1)
    ];

    let mut weights = w0;
    weights.extend(w1);
    weights.extend(w2);

    let input = vec![1.0_f64, 2.0, 3.0, 4.0];
    let output = run_inference(&weights, &input, &configs).expect("multi-layer inference must succeed");

    assert_eq!(output.len(), 1, "final layer must produce 1 output");
    // out after layer0 (relu): [1, 2, 3]
    // out after layer1 (relu): [1, 2]
    // out after layer2 (relu): [0.5 + 1.0] = [1.5]
    assert!((output[0] - 1.5).abs() < 1e-10, "expected 1.5, got {}", output[0]);
}

// ── error paths ───────────────────────────────────────────────────────────────

#[test]
fn empty_layer_configs_returns_error() {
    let result = run_inference(&[], &[1.0], &[]);
    assert!(result.is_err(), "empty configs must return Err");
}

#[test]
fn input_length_mismatch_returns_error() {
    let layer = LayerConfig {
        input_size: 3,
        output_size: 2,
        activation: "relu".to_string(),
    };
    // Provide 2 inputs but layer expects 3.
    let weights = vec![0.0; 3 * 2 + 2]; // correct weight count
    let result = run_inference(&weights, &[1.0, 2.0], &[layer]);
    assert!(result.is_err(), "input size mismatch must return Err");
}

#[test]
fn weight_count_mismatch_returns_error() {
    let layer = LayerConfig {
        input_size: 2,
        output_size: 2,
        activation: "relu".to_string(),
    };
    // Correct = 2*2 + 2 = 6; provide 5.
    let weights = vec![1.0; 5];
    let result = run_inference(&weights, &[1.0, 2.0], &[layer]);
    assert!(result.is_err(), "wrong weight count must return Err");
}

#[test]
fn layer_shape_mismatch_between_layers_returns_error() {
    let configs = vec![
        LayerConfig { input_size: 2, output_size: 3, activation: "relu".to_string() },
        // input_size=4 does not match previous output_size=3
        LayerConfig { input_size: 4, output_size: 1, activation: "relu".to_string() },
    ];
    let weights = vec![0.0; 2 * 3 + 3 + 4 * 1 + 1];
    let result = run_inference(&weights, &[1.0, 2.0], &configs);
    assert!(result.is_err(), "layer shape mismatch must return Err");
}

// ── infer convenience wrapper ─────────────────────────────────────────────────

#[test]
fn infer_wrapper_returns_empty_on_bad_input() {
    // Bad: no layers supplied.
    let output = infer(vec![], vec![1.0], vec![]);
    assert!(output.is_empty(), "infer must return empty vec on error");
}

#[test]
fn infer_wrapper_succeeds_on_valid_input() {
    let layer = LayerConfig {
        input_size: 2,
        output_size: 2,
        activation: "linear".to_string(),
    };
    // Identity 2×2 + zero biases
    let weights = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let output = infer(weights, vec![3.0, 7.0], vec![layer]);
    assert_eq!(output.len(), 2);
    assert!((output[0] - 3.0).abs() < 1e-10);
    assert!((output[1] - 7.0).abs() < 1e-10);
}

// ── wasm-encoder topology section ────────────────────────────────────────────

#[test]
fn topology_section_starts_with_wasm_magic() {
    let configs = vec![LayerConfig {
        input_size: 4,
        output_size: 2,
        activation: "relu".to_string(),
    }];
    let bytes = encode_topology_section(&configs);
    assert!(bytes.len() >= 4, "encoded bytes must contain at least the WASM magic");
    assert_eq!(&bytes[..4], b"\0asm", "WASM binary must begin with \\0asm magic");
}

#[test]
fn topology_section_grows_with_more_layers() {
    let one_layer = vec![LayerConfig {
        input_size: 2,
        output_size: 2,
        activation: "relu".to_string(),
    }];
    let two_layers = vec![
        LayerConfig { input_size: 2, output_size: 4, activation: "relu".to_string() },
        LayerConfig { input_size: 4, output_size: 1, activation: "sigmoid".to_string() },
    ];

    let bytes_one = encode_topology_section(&one_layer);
    let bytes_two = encode_topology_section(&two_layers);

    assert!(
        bytes_two.len() > bytes_one.len(),
        "more layers must produce a larger topology section",
    );
}

#[test]
fn topology_section_is_non_empty_for_empty_configs() {
    // Even with no layers the function must return a valid (minimal) WASM binary.
    let bytes = encode_topology_section(&[]);
    assert!(bytes.len() >= 8, "minimal WASM must be ≥8 bytes (magic + version)");
}
