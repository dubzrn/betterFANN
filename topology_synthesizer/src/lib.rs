//! MODULE 2 — TopologySynthesizer
//!
//! Synthesises directed-graph topologies that are optimal across three
//! network-engineering objectives: latency, throughput, and fault-tolerance.
//! The Pareto frontier is computed over a population of at least 50 candidate
//! graphs; the best non-dominated solution is returned.
//!
//! The Schappeller Ur-Maschine default produces a 7-node sphere topology with
//! an orbital quorum ring — a configuration that naturally dominates the Pareto
//! frontier for small, resilient networks.

pub mod objectives;
pub mod pareto;
pub mod synthesizer;

pub use objectives::ObjectiveScores;
pub use pareto::{compute_pareto_frontier, is_dominated};
pub use synthesizer::{synthesize, SynthesisResult, TopologyObjective, TopologySpec, TopologyType};
