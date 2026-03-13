use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use secrecy::{ExposeSecret, Secret};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

use crate::secure_matrix::SecureWeightMatrix;

pub struct CryptoEphemeralNet {
    layers: Vec<SecureWeightMatrix>,
    /// Bias vector stored as a secret; zeroed explicitly in Drop.
    bias: Secret<Vec<f64>>,
    zeroed_flag: Option<Arc<AtomicBool>>,
}

impl CryptoEphemeralNet {
    pub fn new(layers: Vec<SecureWeightMatrix>, bias: Vec<f64>) -> Self {
        Self {
            layers,
            bias: Secret::new(bias),
            zeroed_flag: None,
        }
    }

    /// Attach a drop probe; the flag is set to `true` after zeroing completes.
    pub fn with_drop_probe(mut self, flag: Arc<AtomicBool>) -> Self {
        self.zeroed_flag = Some(flag);
        self
    }

    pub fn layers(&self) -> &[SecureWeightMatrix] {
        &self.layers
    }

    /// Returns `true` (as a subtle `Choice`) when `self` and `other` have the same layer count.
    pub fn same_depth(&self, other: &CryptoEphemeralNet) -> subtle::Choice {
        let a = (self.layers.len() as u64).to_le_bytes();
        let b = (other.layers.len() as u64).to_le_bytes();
        a.ct_eq(&b)
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let bias = self.bias.expose_secret();
        let mut output: Vec<f64> = input
            .iter()
            .enumerate()
            .map(|(i, &v)| v + bias.get(i).copied().unwrap_or(0.0))
            .collect();

        for layer in &self.layers {
            let rows = layer.rows();
            let cols = layer.cols();
            let mut next = vec![0.0_f64; rows];
            for r in 0..rows {
                let mut acc = 0.0_f64;
                for c in 0..cols.min(output.len()) {
                    acc += layer.get(r, c) * output[c];
                }
                next[r] = acc.tanh();
            }
            output = next;
        }
        output
    }
}

impl Drop for CryptoEphemeralNet {
    fn drop(&mut self) {
        // Explicitly zero every layer before the fields are dropped.
        for layer in self.layers.iter_mut() {
            layer.zeroize();
        }
        // Signal the drop probe after zeroing is confirmed.
        if let Some(flag) = &self.zeroed_flag {
            flag.store(true, Ordering::SeqCst);
        }
    }
}
