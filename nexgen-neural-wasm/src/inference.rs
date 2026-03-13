//! Neural inference engine — layer-based forward pass.
//!
//! This module is the computational core of the component.  It is pure Rust
//! with no WASM-specific dependencies and compiles identically on every host
//! target, making it directly testable without a WASM runtime.
//!
//! # Weight layout
//!
//! All weights for a network are passed as a single flat `&[f64]`.  For each
//! layer `l`, weights are laid out in **row-major order**:
//!
//! ```text
//! [ w[0,0]  w[0,1] … w[0,input_size-1]   ← row 0 (output neuron 0)
//!   w[1,0]  w[1,1] …                      ← row 1
//!   …
//!   w[out-1, 0] …                         ← last row
//!   b[0]  b[1] … b[out-1] ]              ← bias vector
//! ```
//!
//! Layers are consumed left-to-right matching the `layer_configs` slice.

use std::borrow::Cow;

use wasm_encoder::{CustomSection, Module};

// ── LayerConfig ───────────────────────────────────────────────────────────────

/// Runtime description of a single fully-connected layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Number of input features this layer consumes.
    pub input_size: usize,
    /// Number of output neurons this layer produces.
    pub output_size: usize,
    /// Name of the element-wise activation applied after the affine transform.
    /// Recognised values: `"relu"`, `"sigmoid"`, `"tanh"`, `"leaky-relu"`,
    /// `"linear"` (and any unrecognised string → identity).
    pub activation: String,
}

// ── Activation ────────────────────────────────────────────────────────────────

fn apply_activation(value: f64, activation: &str) -> f64 {
    match activation {
        "relu" => value.max(0.0),
        "sigmoid" => 1.0 / (1.0 + (-value).exp()),
        "tanh" => value.tanh(),
        // Small negative slope (0.01) prevents dead neurons in cold layers.
        "leaky-relu" => {
            if value >= 0.0 {
                value
            } else {
                0.01 * value
            }
        }
        // "linear" or any unknown name → identity pass-through.
        _ => value,
    }
}

// ── run_inference ─────────────────────────────────────────────────────────────

/// Execute a full forward pass through a multi-layer network.
///
/// # Errors
///
/// Returns an error string when:
/// * `layer_configs` is empty.
/// * `input.len()` does not match the first layer's `input_size`.
/// * `weights.len()` does not equal the total number of weights and biases
///   implied by `layer_configs`.
pub fn run_inference(
    weights: &[f64],
    input: &[f64],
    layer_configs: &[LayerConfig],
) -> Result<Vec<f64>, String> {
    if layer_configs.is_empty() {
        return Err("at least one layer configuration is required".to_string());
    }

    let first_input_size = layer_configs[0].input_size;
    if input.len() != first_input_size {
        return Err(format!(
            "input length {} does not match first-layer input_size {}",
            input.len(),
            first_input_size,
        ));
    }

    // Verify connectivity: each layer's input_size must equal the previous
    // layer's output_size.
    for pair in layer_configs.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        if prev.output_size != next.input_size {
            return Err(format!(
                "layer shape mismatch: previous output_size {} ≠ next input_size {}",
                prev.output_size, next.input_size,
            ));
        }
    }

    let required_weights: usize = layer_configs
        .iter()
        .map(|l| l.input_size * l.output_size + l.output_size)
        .sum();

    if weights.len() != required_weights {
        return Err(format!(
            "expected {} weights+biases for the given layer configs, got {}",
            required_weights,
            weights.len(),
        ));
    }

    let mut current: Vec<f64> = input.to_vec();
    let mut offset = 0usize;

    for layer in layer_configs {
        let w_len = layer.input_size * layer.output_size;
        let b_len = layer.output_size;

        let w = &weights[offset..offset + w_len];
        let b = &weights[offset + w_len..offset + w_len + b_len];
        offset += w_len + b_len;

        let mut next = vec![0.0_f64; layer.output_size];
        for j in 0..layer.output_size {
            let mut acc = b[j];
            for i in 0..layer.input_size {
                acc += w[j * layer.input_size + i] * current[i];
            }
            next[j] = apply_activation(acc, &layer.activation);
        }
        current = next;
    }

    Ok(current)
}

// ── encode_topology_section ───────────────────────────────────────────────────

/// Encode the layer topology as a WASM custom section and return the full
/// WASM binary (magic + version + one custom section).
///
/// The custom section name is `"betterFANN.neural.topology"` and its data
/// payload is:
///
/// ```text
/// [layer_count: u32 LE]
/// for each layer:
///   [input_size: u32 LE] [output_size: u32 LE]
///   [activation_len: u32 LE] [activation bytes: UTF-8]
/// ```
///
/// This binary blob can be embedded in a WASM component for tooling and
/// introspection without affecting execution semantics.
pub fn encode_topology_section(layer_configs: &[LayerConfig]) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&(layer_configs.len() as u32).to_le_bytes());

    for layer in layer_configs {
        payload.extend_from_slice(&(layer.input_size as u32).to_le_bytes());
        payload.extend_from_slice(&(layer.output_size as u32).to_le_bytes());
        let act = layer.activation.as_bytes();
        payload.extend_from_slice(&(act.len() as u32).to_le_bytes());
        payload.extend_from_slice(act);
    }

    let mut module = Module::new();
    module.section(&CustomSection {
        name: Cow::Borrowed("betterFANN.neural.topology"),
        data: Cow::Borrowed(&payload),
    });
    module.finish()
}
