//! Activation functions modelling the Schauberger temperature gradient.
//!
//! In the centripetal flow model, each activation acts as a "cooling stage":
//! it concentrates the signal energy toward the cold output core while
//! preserving directional information.

use ndarray::Array1;

/// Non-linear activation applied element-wise after each dense transformation.
///
/// Variants are ordered from highest to lowest energy dissipation, mirroring
/// the Schauberger vortex cooling sequence:
/// `Tanh` (widest gradient suppression) → `LeakyReLU` (lowest suppression).
#[derive(Debug, Clone)]
pub enum Activation {
    /// Hyperbolic tangent — smooth, bounded to (−1, 1); strongest centripetal
    /// compression.
    Tanh,

    /// Rectified linear unit — passes positive signal, zeros negative; hard
    /// energy gate.
    ReLU,

    /// Logistic sigmoid — smooth bounded to (0, 1); probabilistic
    /// concentration.
    Sigmoid,

    /// Leaky ReLU — passes positive signal at unit slope; scales negative
    /// values by `alpha` preventing dead neurons in cold layers.
    LeakyReLU(f64),
}

impl Activation {
    /// Apply this activation function element-wise to `x`.
    ///
    /// Returns a new array of the same shape.
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::LeakyReLU(alpha) => {
                let a = *alpha;
                x.mapv(|v| if v >= 0.0 { v } else { a * v })
            }
        }
    }
}
