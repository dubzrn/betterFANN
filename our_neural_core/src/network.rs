//! Concurrent multi-layer network with `RwLock`-protected inference.
//!
//! [`Network`] owns its layers behind an `Arc<RwLock<…>>`. Any number of
//! threads may call [`Network::infer`] concurrently; each acquires a *read*
//! lock, leaving the weight graph immutable and contention-free during normal
//! operation.

use std::sync::{Arc, RwLock};

use crate::activation::Activation;
use crate::layer::DenseLayer;

/// A feed-forward neural network with centripetal data flow.
///
/// Layers are stored in an `Arc<RwLock<Vec<DenseLayer>>>`, making the network
/// cheaply clonable (shallow `Arc` copy) and safe for concurrent inference
/// across many threads or async tasks.
///
/// # Example
/// ```
/// use our_neural_core::{Activation, Network};
///
/// let net = Network::new(&[4, 8, 2], vec![Activation::ReLU, Activation::Sigmoid]);
/// let output = net.infer(&[0.1, 0.2, 0.3, 0.4]);
/// assert_eq!(output.len(), 2);
/// ```
pub struct Network {
    layers: Arc<RwLock<Vec<DenseLayer>>>,
}

impl Network {
    /// Build a fully-connected network from a list of layer sizes and matching
    /// activation functions.
    ///
    /// # Arguments
    /// * `layer_sizes` — e.g. `&[784, 256, 128, 10]` creates three layers.
    ///   Must have at least two elements.
    /// * `activations` — one activation per layer transition
    ///   (`layer_sizes.len() - 1` entries).
    ///
    /// # Panics
    /// Panics if `layer_sizes.len() < 2` or if the length of `activations`
    /// does not equal `layer_sizes.len() - 1`.
    pub fn new(layer_sizes: &[usize], activations: Vec<Activation>) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Network requires at least an input and an output size"
        );
        assert_eq!(
            layer_sizes.len() - 1,
            activations.len(),
            "activations.len() must equal layer_sizes.len() - 1"
        );

        let layers: Vec<DenseLayer> = layer_sizes
            .windows(2)
            .zip(activations)
            .map(|(w, act)| DenseLayer::new(w[0], w[1], act))
            .collect();

        Self {
            layers: Arc::new(RwLock::new(layers)),
        }
    }

    /// Run a forward pass (inference) through all layers.
    ///
    /// Acquires a *read* lock so multiple concurrent callers are never blocked
    /// by one another. Returns a `Vec<f64>` whose length equals the output
    /// size of the last layer.
    ///
    /// # Panics
    /// Panics if the `RwLock` has been poisoned by a previous panic.
    pub fn infer(&self, input: &[f64]) -> Vec<f64> {
        let layers = self
            .layers
            .read()
            .expect("RwLock poisoned during inference");

        layers
            .iter()
            .fold(input.to_vec(), |current, layer| layer.forward(&current))
    }
}

impl Clone for Network {
    /// Produce a shallow clone that shares the same underlying layer graph.
    ///
    /// This is an `O(1)` operation — only the `Arc` reference count is
    /// incremented. Both the original and the clone observe the same weights.
    fn clone(&self) -> Self {
        Self {
            layers: Arc::clone(&self.layers),
        }
    }
}
