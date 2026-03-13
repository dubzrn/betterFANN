//! **sphere_node** — Distributed neural-network node with real CRDT-based
//! weight synchronisation and gossip-protocol peer exchange.
//!
//! ## Architecture
//!
//! Each `SphereNode` owns a weight vector stored as a last-write-wins (LWW)
//! CRDT: every weight cell carries a logical clock so merging two weight sets
//! always produces the causally-correct result without coordinator involvement.
//!
//! Gossip is carried over an in-process `tokio::sync::mpsc` channel in this
//! library; a real deployment swaps the channel ends for TCP/QUIC sockets
//! without changing any CRDT logic.
//!
//! ## Why this fixes ruvnet/ruv-FANN flaw \#6
//!
//! `ruv-swarm/crates/ruv-swarm-wasm/src/neural_swarm_coordinator.rs` (lines
//! 606-661) defines `DistributedTrainingMode` and `InferenceMode` enums but
//! executes coordination logic with mock/simulated data.  `sphere_node` uses
//! a real LWW-CRDT merge (`WeightCell::merge`) and a genuine gossip
//! fanout-2 loop that converges the full cluster without a central
//! coordinator.

pub mod crdt;
pub mod gossip;
pub mod node;

pub use crdt::{WeightCell, WeightSet};
pub use gossip::GossipMsg;
pub use node::{NodeId, NodeLifecycle, SphereNode};
