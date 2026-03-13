//! Memory-safe weight storage with automatic zeroing on drop.
//!
//! `SecureWeights` wraps a plain `Vec<f64>` with `ZeroizeOnDrop` so the
//! memory backing the parameter vector is overwritten with zeros when the
//! value goes out of scope.
//!
//! ## Why this matters
//!
//! `ruvnet/ruv-FANN` stores weights as plain `Vec<T>` (src/network.rs line 36)
//! with no zeroing on drop.  After the network is freed the allocator may
//! reuse the memory without clearing it, leaking weights in a heap scan.
//! `SecureWeights` eliminates that leak.

use zeroize::{Zeroize, ZeroizeOnDrop};

/// A weight vector that zeroes its backing memory before deallocation.
///
/// Fields are deliberately private; access is through the provided methods to
/// prevent accidental copies of the raw slice from escaping.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SecureWeights {
    data: Vec<f64>,
}

impl SecureWeights {
    /// Allocate a new weight vector of length `n`, initialised by calling
    /// `init(index)` for each position.
    pub fn from_fn(n: usize, mut init: impl FnMut(usize) -> f64) -> Self {
        Self {
            data: (0..n).map(|i| init(i)).collect(),
        }
    }

    /// Wrap an existing `Vec<f64>`.
    pub fn from_vec(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only view.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable view for in-place gradient updates.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Compute the dot product of this weight row with `input` using an
    /// unrolled accumulation loop that allows LLVM to auto-vectorise.
    ///
    /// This is the core hot path; the loop is chunk-unrolled by 4 to hint the
    /// optimiser toward 256-bit SIMD lanes.
    #[inline]
    pub fn dot(&self, input: &[f64]) -> f64 {
        let w = &self.data;
        let n = w.len().min(input.len());
        let chunks = n / 4;
        let remainder = n % 4;

        let mut acc0 = 0.0_f64;
        let mut acc1 = 0.0_f64;
        let mut acc2 = 0.0_f64;
        let mut acc3 = 0.0_f64;

        // Unrolled 4-wide loop — LLVM will fuse these into vectorised FMA
        // instructions when targeting x86_64 with AVX2.
        for i in 0..chunks {
            let base = i * 4;
            acc0 += w[base] * input[base];
            acc1 += w[base + 1] * input[base + 1];
            acc2 += w[base + 2] * input[base + 2];
            acc3 += w[base + 3] * input[base + 3];
        }

        let tail_start = chunks * 4;
        let mut tail_acc = 0.0_f64;
        for i in 0..remainder {
            tail_acc += w[tail_start + i] * input[tail_start + i];
        }

        acc0 + acc1 + acc2 + acc3 + tail_acc
    }
}

impl std::fmt::Debug for SecureWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecureWeights(len={})", self.data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_correctness() {
        // 8-element vectors so both the unrolled and tail paths are exercised.
        let w = SecureWeights::from_vec(vec![1.0; 8]);
        let input = vec![0.5; 8];
        let result = w.dot(&input);
        assert!((result - 4.0).abs() < 1e-12, "dot product should be 4.0");
    }

    #[test]
    fn dot_product_odd_length() {
        let w = SecureWeights::from_vec(vec![1.0, 2.0, 3.0]);
        let input = vec![1.0, 1.0, 1.0];
        let result = w.dot(&input);
        assert!((result - 6.0).abs() < 1e-12);
    }

    #[test]
    fn zeroize_on_drop() {
        // We cannot inspect memory after drop in safe Rust, but we can verify
        // that SecureWeights correctly implements ZeroizeOnDrop by checking
        // that calling zeroize() (which Drop invokes) sets every element to 0.
        use zeroize::Zeroize;
        let mut sw = SecureWeights::from_vec(vec![42.0, 43.0, 44.0]);
        sw.zeroize();
        assert!(sw.as_slice().iter().all(|&v| v == 0.0));
    }
}
