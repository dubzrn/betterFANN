//! # NeuralCore — MODULE 1
//!
//! Centripetal inference engine modelling the Schauberger temperature gradient:
//! high-entropy input concentrates through successive layers toward a cold,
//! low-entropy output core.
//!
//! ## Architecture
//! - [`Activation`]  — element-wise non-linearities (Tanh / ReLU / Sigmoid / LeakyReLU)
//! - [`DenseLayer`]  — fully-connected layer backed by [`SecureWeightMatrix`]
//! - [`Network`]     — `Arc<RwLock<…>>` protected multi-layer graph for concurrent inference
//!
//! Zero `unsafe` blocks are present anywhere in this crate.

pub mod activation;
pub mod layer;
pub mod network;

pub use activation::Activation;
pub use layer::{DenseLayer, SecureWeightMatrix};
pub use network::Network;
