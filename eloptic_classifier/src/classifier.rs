//! Multi-layer classifier with SGD training and mean-squared-error loss.
//!
//! [`ElopticClassifier`] chains any number of [`DenseLayer`]s and exposes:
//! - `forward` — full inference pass  
//! - `train_step` — single SGD update (forward + backward)
//! - `train` — convenience loop over multiple epochs

use std::sync::Arc;

use crate::activation::Activation;
use crate::layer::DenseLayer;

/// A fully-connected, multi-layer classifier with backprop.
pub struct ElopticClassifier {
    layers: Vec<DenseLayer>,
}

impl ElopticClassifier {
    /// Build a classifier from a list of `(input_size, output_size, activation)`
    /// triples.
    ///
    /// # Panics
    /// Panics if `specs` is empty.
    pub fn new(specs: Vec<(usize, usize, Arc<dyn Activation>)>) -> Self {
        assert!(!specs.is_empty(), "ElopticClassifier requires at least one layer");
        let layers = specs
            .into_iter()
            .map(|(i, o, act)| DenseLayer::new(i, o, act))
            .collect();
        Self { layers }
    }

    /// Run a full forward pass and return the output activations.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            let (_pre, post) = layer.forward(&current);
            current = post;
        }
        current
    }

    /// Run one SGD training step:
    /// 1. Forward pass — store all pre-activations and activations.
    /// 2. Compute MSE loss gradient at the output.
    /// 3. Backpropagate deltas through all layers.
    /// 4. Apply weight updates with learning rate `lr`.
    ///
    /// Returns the MSE loss for this sample.
    pub fn train_step(&mut self, input: &[f64], target: &[f64], lr: f64) -> f64 {
        // ── Forward pass ─────────────────────────────────────────────────────
        let mut all_inputs: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        all_inputs.push(input.to_vec());

        let mut all_pre: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (pre, post) = layer.forward(all_inputs.last().unwrap());
            all_pre.push(pre);
            all_inputs.push(post);
        }

        let output = all_inputs.last().unwrap();

        // ── MSE loss ─────────────────────────────────────────────────────────
        let n = output.len() as f64;
        let mse: f64 = output
            .iter()
            .zip(target)
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>()
            / n;

        // ── Output gradient: ∂MSE/∂a = 2(a − t)/n ───────────────────────────
        let d_output: Vec<f64> = output
            .iter()
            .zip(target)
            .map(|(o, t)| 2.0 * (o - t) / n)
            .collect();

        // ── Backward pass ─────────────────────────────────────────────────────
        let num_layers = self.layers.len();
        let mut delta = self.layers[num_layers - 1]
            .output_delta(&all_pre[num_layers - 1], &d_output);

        // Reverse-iterate and accumulate (grad_w, delta) pairs.
        let mut updates: Vec<(Vec<Vec<f64>>, Vec<f64>)> = Vec::with_capacity(num_layers);

        let layer_input = &all_inputs[num_layers - 1];
        let (gw, d_in) = self.layers[num_layers - 1].backward(&delta, layer_input);
        updates.push((gw, delta.clone()));
        let mut d_prev = d_in;

        for l in (0..num_layers - 1).rev() {
            delta = self.layers[l].output_delta(&all_pre[l], &d_prev);
            let (gw, d_in) = self.layers[l].backward(&delta, &all_inputs[l]);
            updates.push((gw, delta.clone()));
            d_prev = d_in;
        }

        // Apply updates in forward order (updates were pushed in reverse).
        updates.reverse();
        for (l, (gw, dlt)) in updates.into_iter().enumerate() {
            self.layers[l].apply_gradients(&gw, &dlt, lr);
        }

        mse
    }

    /// Train for `epochs` epochs over a dataset of `(input, target)` pairs.
    ///
    /// Returns the final-epoch average MSE.
    pub fn train(
        &mut self,
        dataset: &[(Vec<f64>, Vec<f64>)],
        epochs: usize,
        lr: f64,
    ) -> f64 {
        let mut last_loss = f64::MAX;
        for _ in 0..epochs {
            let total: f64 = dataset
                .iter()
                .map(|(x, y)| self.train_step(x, y, lr))
                .sum();
            last_loss = total / dataset.len() as f64;
        }
        last_loss
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl std::fmt::Debug for ElopticClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ElopticClassifier({} layers)", self.layers.len())?;
        for (i, l) in self.layers.iter().enumerate() {
            write!(f, "\n  [{i}] {l:?}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{ReLU, Sigmoid};

    fn xor_dataset() -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ]
    }

    #[test]
    fn forward_shape_is_correct() {
        let clf = ElopticClassifier::new(vec![
            (4, 8, Arc::new(ReLU) as Arc<dyn Activation>),
            (8, 2, Arc::new(Sigmoid)),
        ]);
        let out = clf.forward(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(out.len(), 2);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn loss_decreases_after_training() {
        let mut clf = ElopticClassifier::new(vec![
            (2, 4, Arc::new(ReLU) as Arc<dyn Activation>),
            (4, 1, Arc::new(Sigmoid)),
        ]);
        let data = xor_dataset();
        let initial_loss = {
            let total: f64 = data.iter().map(|(x, y)| {
                let out = clf.forward(x);
                let n = out.len() as f64;
                out.iter().zip(y).map(|(o, t)| (o - t).powi(2)).sum::<f64>() / n
            }).sum();
            total / data.len() as f64
        };

        // Train for 200 epochs.
        let final_loss = clf.train(&data, 200, 0.1);

        assert!(
            final_loss < initial_loss,
            "loss should decrease: initial={initial_loss:.4} final={final_loss:.4}"
        );
    }
}
