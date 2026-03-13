//! Dense (fully-connected) layer with trait-based activation.

use std::sync::Arc;

use rand::Rng;

use crate::activation::Activation;
use crate::weights::SecureWeights;

/// A single fully-connected layer.
///
/// # Storage layout
///
/// Weights are stored in row-major order: `weights[r]` is the weight row for
/// output neuron `r`, with `input_size` columns.  Biases are stored in a
/// separate `SecureWeights` of length `output_size`.
pub struct DenseLayer {
    /// One `SecureWeights` row per output neuron.
    weights: Vec<SecureWeights>,
    biases: SecureWeights,
    activation: Arc<dyn Activation>,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    /// Construct a layer with Xavier-initialised weights.
    ///
    /// Xavier initialisation scales the initial weights by
    /// `sqrt(2 / (fan_in + fan_out))`, giving stable gradient magnitudes
    /// across the first forward pass.
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Arc<dyn Activation>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0_f64 / (input_size + output_size) as f64).sqrt();

        let weights: Vec<SecureWeights> = (0..output_size)
            .map(|_| {
                SecureWeights::from_fn(input_size, |_| rng.gen::<f64>() * 2.0 * scale - scale)
            })
            .collect();

        let biases = SecureWeights::from_fn(output_size, |_| 0.0);

        Self {
            weights,
            biases,
            activation,
            input_size,
            output_size,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Forward pass: returns (pre-activations, activations).
    ///
    /// Pre-activations are kept so the caller can run backprop without
    /// recomputing them.
    pub fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        debug_assert_eq!(input.len(), self.input_size);
        let biases = self.biases.as_slice();

        let pre: Vec<f64> = self
            .weights
            .iter()
            .enumerate()
            .map(|(r, row)| row.dot(input) + biases[r])
            .collect();

        let post: Vec<f64> = pre.iter().map(|&z| self.activation.apply(z)).collect();

        (pre, post)
    }

    /// Compute the output-layer delta from the loss gradient `d_output`
    /// (∂L/∂a * ∂a/∂z for each neuron).
    pub fn output_delta(&self, pre: &[f64], d_output: &[f64]) -> Vec<f64> {
        pre.iter()
            .zip(d_output)
            .map(|(&z, &d_l)| d_l * self.activation.derivative(z))
            .collect()
    }

    /// Backprop: given the delta at this layer's output, compute:
    /// - `grad_w[r][c]` = delta[r] * input[c]  (weight gradient)
    /// - `d_input[c]`   = Σ_r delta[r] * w[r][c]  (input gradient for prev layer)
    ///
    /// Returns `(grad_w, d_input)`.
    pub fn backward(
        &self,
        delta: &[f64],
        input: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        // Weight gradients — one row per output neuron.
        let grad_w: Vec<Vec<f64>> = delta
            .iter()
            .map(|&d| input.iter().map(|&x| d * x).collect())
            .collect();

        // Input gradient — contiguous loop over input columns; LLVM
        // auto-vectorises this because `self.weights[r].as_slice()` is a
        // plain contiguous slice and there are no aliasing constraints.
        let mut d_input = vec![0.0_f64; self.input_size];
        for (r, d) in delta.iter().enumerate() {
            let row = self.weights[r].as_slice();
            for (c, di) in d_input.iter_mut().enumerate() {
                *di += d * row[c];
            }
        }

        (grad_w, d_input)
    }

    /// Apply a gradient update in-place: `w -= lr * grad_w`.
    pub fn apply_gradients(&mut self, grad_w: &[Vec<f64>], delta: &[f64], lr: f64) {
        for (r, (row, gw_row)) in self.weights.iter_mut().zip(grad_w).enumerate() {
            let row_slice = row.as_mut_slice();
            for (c, gw) in gw_row.iter().enumerate() {
                row_slice[c] -= lr * gw;
            }
            self.biases.as_mut_slice()[r] -= lr * delta[r];
        }
    }
}

impl std::fmt::Debug for DenseLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DenseLayer({} → {}, activation={})",
            self.input_size,
            self.output_size,
            self.activation.name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::ReLU;

    #[test]
    fn forward_shape() {
        let layer = DenseLayer::new(8, 4, Arc::new(ReLU));
        let input = vec![1.0; 8];
        let (_pre, post) = layer.forward(&input);
        assert_eq!(post.len(), 4);
        for &v in &post {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn backward_gradient_shape() {
        let layer = DenseLayer::new(4, 3, Arc::new(ReLU));
        let input = vec![0.5; 4];
        let delta = vec![0.1; 3];
        let (grad_w, d_input) = layer.backward(&delta, &input);
        assert_eq!(grad_w.len(), 3);
        assert_eq!(grad_w[0].len(), 4);
        assert_eq!(d_input.len(), 4);
    }
}
