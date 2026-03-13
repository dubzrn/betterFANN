//! # nexgen-neural-wasm — MODULE 3
//!
//! WASM Component Model neural inference crate targeting `wasm32-wasi`.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  WIT interface  (wit/neural.wit)        │
//! │  world neural-world { export neural; }  │
//! └──────────────┬──────────────────────────┘
//!                │ wit-bindgen::generate!  (wasm32 only)
//! ┌──────────────▼──────────────────────────┐
//! │  src/lib.rs   — public API + WIT glue   │
//! └──────────────┬──────────────────────────┘
//!                │ delegates to
//! ┌──────────────▼──────────────────────────┐
//! │  src/inference.rs — pure forward pass   │
//! │  + wasm-encoder topology section        │
//! └─────────────────────────────────────────┘
//! ```
//!
//! The `#[cfg(target_arch = "wasm32")]` block wires the WIT-generated bindings
//! to the native inference engine.  Everything outside that block compiles on
//! any host target and is exercised by the integration tests without a WASM
//! runtime.
//!
//! ## Quick start (native)
//!
//! ```rust
//! use nexgen_neural_wasm::{infer, version, LayerConfig};
//!
//! let layer = LayerConfig { input_size: 2, output_size: 1, activation: "sigmoid".to_string() };
//! let weights = vec![0.5, 0.5, 0.0]; // 2 weights + 1 bias
//! let output  = infer(weights, vec![1.0, 1.0], vec![layer]);
//! assert_eq!(output.len(), 1);
//! ```
//!
//! ## Build for WASM
//!
//! ```sh
//! cargo build --target wasm32-wasi --release
//! # then adapt with wasm-tools:
//! # wasm-tools component new target/wasm32-wasi/release/nexgen_neural_wasm.wasm \
//! #   --adapt wasi_snapshot_preview1.wasm -o neural-component.wasm
//! ```

pub mod inference;

pub use inference::{encode_topology_section, run_inference, LayerConfig};

/// Human-readable component version string.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the human-readable component version string.
///
/// The format is `"nexgen-neural-wasm v<semver>"` and is guaranteed non-empty.
pub fn version() -> String {
    format!("nexgen-neural-wasm v{CRATE_VERSION}")
}

/// Convenience wrapper: run inference and return the result, or an empty
/// vector when the inputs are inconsistent (error details are discarded).
///
/// Pass-through to [`run_inference`] with an empty-on-error contract so that
/// the WIT export and native callers share the same failure semantics.
pub fn infer(weights: Vec<f64>, input: Vec<f64>, layer_configs: Vec<LayerConfig>) -> Vec<f64> {
    run_inference(&weights, &input, &layer_configs).unwrap_or_default()
}

// ── WASM Component Model export glue ─────────────────────────────────────────
//
// This entire block is compiled **only** when targeting wasm32.  It generates
// the canonical Component Model ABI glue via wit-bindgen and delegates every
// exported function to the native implementations above.
//
// The block is excluded from native builds so that integration tests can
// exercise the inference engine directly without a WASM runtime.

#[cfg(target_arch = "wasm32")]
mod wasm_component {
    // Generate Rust bindings from the WIT world.
    // On wasm32-wasi this expands to:
    //   • type aliases / structs for WIT record types  (LayerConfig)
    //   • a `Guest` trait with `infer` and `version` methods
    //   • the `export!` macro that emits the component ABI entry points
    wit_bindgen::generate!({
        world: "neural-world",
        path: "wit/neural.wit",
    });

    struct NeuralComponent;

    // The generated `Guest` trait is in `exports::local::neural::neural`
    // (derived from package `local:neural@0.1.0`, interface `neural`).
    impl exports::local::neural::neural::Guest for NeuralComponent {
        fn infer(
            weights: Vec<f64>,
            input: Vec<f64>,
            layer_configs: Vec<exports::local::neural::neural::LayerConfig>,
        ) -> Vec<f64> {
            // Map the WIT-generated LayerConfig (u32 sizes) to the native one.
            let native_configs: Vec<crate::inference::LayerConfig> = layer_configs
                .into_iter()
                .map(|c| crate::inference::LayerConfig {
                    input_size: c.input_size as usize,
                    output_size: c.output_size as usize,
                    activation: c.activation,
                })
                .collect();

            crate::infer(weights, input, native_configs)
        }

        fn version() -> String {
            crate::version()
        }
    }

    // Emit the `#[export_name = "…"]` entry points required by the
    // Component Model ABI.
    export!(NeuralComponent);
}
