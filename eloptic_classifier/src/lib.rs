//! **eloptic_classifier** — Extensible multi-layer classifier.
//!
//! ## Key advantages over ruvnet/ruv-FANN
//!
//! | ruvnet flaw | betterFANN fix |
//! |---|---|
//! | Closed `ActivationFunction` enum (activation.rs lines 12-96, 21 variants) — users cannot add custom functions without forking | `Activation` is an open **trait**; any type implementing `fn apply(&self, f64) -> f64` and `fn derivative(&self, f64) -> f64` becomes a first-class activation |
//! | `backward_pass` O(n²) connection lookups (network.rs lines 333-423) with no SIMD | Backprop uses contiguous `Vec<f64>` with explicit loop-unrolled dot product — auto-vectorised by LLVM; no hash-map lookups |
//! | No weight zeroisation on drop | `SecureWeights` wraps the parameter vector with `ZeroizeOnDrop` |
//!
//! ## Modules
//! - [`activation`] — open activation trait + built-in implementations
//! - [`weights`] — memory-safe weight storage with `zeroize` on drop
//! - [`layer`] — dense layer owning `SecureWeights`
//! - [`classifier`] — full multi-layer forward pass and backprop

pub mod activation;
pub mod classifier;
pub mod layer;
pub mod weights;

pub use activation::{Activation, LeakyReLU, ReLU, Sigmoid, Softmax, Tanh};
pub use classifier::ElopticClassifier;
pub use layer::DenseLayer;
pub use weights::SecureWeights;
