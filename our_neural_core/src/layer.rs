//! Dense layer and secure weight storage.
//!
//! [`SecureWeightMatrix`] wraps an [`ndarray::Array2<f64>`] and guarantees that
//! all weight memory is overwritten with zeros at drop time via [`ZeroizeOnDrop`].
//!
//! [`DenseLayer`] performs a single affine transformation followed by an
//! [`Activation`], implementing one stage of the centripetal data flow:
//!
//! ```text
//! output = activation( weights · input + bias )
//! ```

use ndarray::{Array1, Array2};
use rand::Rng;
use secrecy::{ExposeSecret, Secret};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::activation::Activation;

// ── SecureWeightMatrix ────────────────────────────────────────────────────────

/// An [`Array2<f64>`] weight matrix whose backing memory is cryptographically
/// zeroed before deallocation.
///
/// Implements [`Zeroize`] by iterating over every element and setting it to
/// `0.0_f64`. The [`ZeroizeOnDrop`] derive generates a `Drop` impl that calls
/// `self.zeroize()` automatically, providing defence-in-depth against cold-boot
/// or heap-inspection attacks on model parameters.
///
/// `Array2<f64>` does not itself implement [`Zeroize`], so the field is
/// annotated `#[zeroize(skip)]` to suppress the derive's field-level assertion;
/// the manual [`Zeroize`] implementation on the struct handles erasure.
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use our_neural_core::SecureWeightMatrix;
///
/// let w = SecureWeightMatrix::new(Array2::ones((4, 4)));
/// assert_eq!(w.shape(), (4, 4));
/// ```
#[derive(ZeroizeOnDrop)]
pub struct SecureWeightMatrix {
    #[zeroize(skip)]
    data: Array2<f64>,
}

impl SecureWeightMatrix {
    /// Wrap an existing weight matrix.
    pub fn new(data: Array2<f64>) -> Self {
        Self { data }
    }

    /// Read-only view of the underlying weight data.
    pub fn data(&self) -> &Array2<f64> {
        &self.data
    }

    /// Return `(rows, cols)` — i.e. `(output_size, input_size)`.
    pub fn shape(&self) -> (usize, usize) {
        let s = self.data.shape();
        (s[0], s[1])
    }
}

impl Zeroize for SecureWeightMatrix {
    /// Overwrite every weight element with `0.0`.
    ///
    /// Called automatically by the [`ZeroizeOnDrop`] derive on `Drop`.
    fn zeroize(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0_f64);
    }
}

// ── DenseLayer ────────────────────────────────────────────────────────────────

/// A fully-connected layer: one stage in the centripetal concentration
/// of information from a wide input toward the cold output core.
///
/// Weights are stored in a [`SecureWeightMatrix`] (shape `output × input`).
/// Biases are stored in a [`Secret<Vec<f64>>`], which zeroes its memory on
/// drop via the `secrecy` crate.
///
/// # Construction
/// Weights are initialised with **Glorot / Xavier uniform** sampling:
/// `U(−limit, +limit)` where `limit = √(6 / (fan_in + fan_out))`.
/// Biases are initialised to zero.
pub struct DenseLayer {
    /// Weight matrix, shape `(output_size, input_size)`.
    weights: SecureWeightMatrix,
    /// Bias vector, length `output_size`, protected by `secrecy::Secret`.
    bias: Secret<Vec<f64>>,
    /// Element-wise non-linearity applied after the affine step.
    activation: Activation,
    /// Cached output dimension for shape-assertion-free external queries.
    output_size: usize,
}

impl DenseLayer {
    /// Create a new dense layer with Glorot-uniform weight initialisation.
    ///
    /// # Arguments
    /// * `input_size`  — fan-in (number of incoming connections)
    /// * `output_size` — fan-out (number of neurons in this layer)
    /// * `activation`  — non-linearity applied after the affine transform
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (6.0_f64 / (input_size + output_size) as f64).sqrt();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.gen_range(-limit..limit)
        });

        Self {
            weights: SecureWeightMatrix::new(weights),
            bias: Secret::new(vec![0.0_f64; output_size]),
            activation,
            output_size,
        }
    }

    /// Compute the forward pass for a single sample.
    ///
    /// Implements the centripetal step:
    /// `activation( W · input + b )`
    ///
    /// # Arguments
    /// * `input` — slice of length `input_size`
    ///
    /// # Returns
    /// A `Vec<f64>` of length `output_size`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let input_arr = Array1::from_vec(input.to_vec());
        let bias_arr = Array1::from_vec(self.bias.expose_secret().clone());
        let pre_activation = self.weights.data().dot(&input_arr) + bias_arr;
        self.activation.apply(&pre_activation).to_vec()
    }

    /// Number of neurons (outputs) in this layer.
    pub fn output_size(&self) -> usize {
        self.output_size
    }
}
