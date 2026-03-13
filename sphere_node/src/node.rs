//! Node lifecycle management for `sphere_node`.
//!
//! A [`SphereNode`] goes through a well-defined lifecycle:
//!
//! ```text
//! Bootstrapping ──► Active ──► Draining ──► Stopped
//!                       ↑         │
//!                       └─────────┘  (re-join after drain if not stopped)
//! ```
//!
//! While `Active` the node merges incoming gossip messages and periodically
//! broadcasts its own weights to peers via the gossip fanout.

use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::crdt::WeightSet;
use crate::gossip::{gossip_round, GossipMsg};

/// Opaque node identifier — a 64-bit integer assigned at construction.
pub type NodeId = u64;

/// Lifecycle state machine for a distributed sphere node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLifecycle {
    Bootstrapping,
    Active,
    Draining,
    Stopped,
}

/// A distributed neural-network node that synchronises weights via CRDT gossip.
///
/// # Example
/// ```rust
/// use sphere_node::{SphereNode, NodeLifecycle};
///
/// let node = SphereNode::new(1, 16);
/// assert_eq!(node.lifecycle(), NodeLifecycle::Bootstrapping);
/// let node = node.activate();
/// assert_eq!(node.lifecycle(), NodeLifecycle::Active);
/// ```
pub struct SphereNode {
    id: NodeId,
    /// Local weight CRDT — protected by a `Mutex` for interior mutability.
    weights: Arc<Mutex<WeightSet>>,
    lifecycle: NodeLifecycle,
    /// Broadcast channel on which this node publishes gossip messages.
    gossip_tx: mpsc::Sender<GossipMsg>,
    /// Receive-half — kept alive so the channel is not closed prematurely.
    gossip_rx: Arc<Mutex<mpsc::Receiver<GossipMsg>>>,
}

impl SphereNode {
    /// Create a new node with `weight_dim`-dimensional parameters initialised
    /// to zero.
    pub fn new(id: NodeId, weight_dim: usize) -> Self {
        let (tx, rx) = mpsc::channel::<GossipMsg>(64);
        Self {
            id,
            weights: Arc::new(Mutex::new(WeightSet::new(weight_dim, id))),
            lifecycle: NodeLifecycle::Bootstrapping,
            gossip_tx: tx,
            gossip_rx: Arc::new(Mutex::new(rx)),
        }
    }

    /// Transition from `Bootstrapping` to `Active`.
    ///
    /// # Panics
    /// Panics if the node is not currently in `Bootstrapping` state.
    pub fn activate(mut self) -> Self {
        assert_eq!(
            self.lifecycle,
            NodeLifecycle::Bootstrapping,
            "activate() called on a node that is not Bootstrapping"
        );
        self.lifecycle = NodeLifecycle::Active;
        self
    }

    /// Transition from `Active` to `Draining`.
    pub fn drain(mut self) -> Self {
        self.lifecycle = NodeLifecycle::Draining;
        self
    }

    /// Transition from `Draining` to `Stopped`.
    pub fn stop(mut self) -> Self {
        assert_eq!(
            self.lifecycle,
            NodeLifecycle::Draining,
            "stop() called on a node that is not Draining"
        );
        self.lifecycle = NodeLifecycle::Stopped;
        self
    }

    pub fn lifecycle(&self) -> NodeLifecycle {
        self.lifecycle
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Return a clone of the current weight values.
    pub fn weight_snapshot(&self) -> Vec<f64> {
        self.weights.lock().unwrap().values()
    }

    /// Update local weights with a freshly computed gradient step.
    ///
    /// Each cell's clock is incremented so remote peers will accept the update
    /// during the next gossip round.
    pub fn write_weights(&self, new_values: &[f64]) {
        self.weights.lock().unwrap().write_all(new_values);
    }

    /// Merge an incoming `GossipMsg` into the local CRDT.
    ///
    /// Returns the number of cells that were updated.
    pub fn apply_gossip(&self, msg: &GossipMsg) -> usize {
        self.weights.lock().unwrap().merge(&msg.weights)
    }

    /// Receive all pending gossip messages from the internal channel and merge
    /// them into the local CRDT.
    ///
    /// Returns the total number of weight cells updated across all messages.
    pub fn drain_inbox(&self) -> usize {
        let mut total = 0usize;
        let mut rx = self.gossip_rx.lock().unwrap();
        while let Ok(msg) = rx.try_recv() {
            total += self.weights.lock().unwrap().merge(&msg.weights);
        }
        total
    }

    /// Build a gossip message from the current local state.
    pub fn build_gossip_msg(&self) -> GossipMsg {
        let ws = self.weights.lock().unwrap();
        GossipMsg {
            from: self.id,
            weights: ws.clone(),
            sender_clock: ws.max_clock(),
        }
    }

    /// Broadcast the local weights to up to two peers (fanout-2 gossip).
    ///
    /// Returns the number of peers reached.
    pub async fn broadcast(&self, peers: &[mpsc::Sender<GossipMsg>]) -> usize {
        let msg = self.build_gossip_msg();
        gossip_round(msg, peers, self.id ^ (self.weights.lock().unwrap().max_clock())).await
    }

    /// Return a `Sender` handle so other nodes can gossip to this node.
    pub fn inbox(&self) -> mpsc::Sender<GossipMsg> {
        self.gossip_tx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_transitions_are_correct() {
        let node = SphereNode::new(42, 8);
        assert_eq!(node.lifecycle(), NodeLifecycle::Bootstrapping);

        let node = node.activate();
        assert_eq!(node.lifecycle(), NodeLifecycle::Active);

        let node = node.drain();
        assert_eq!(node.lifecycle(), NodeLifecycle::Draining);

        let node = node.stop();
        assert_eq!(node.lifecycle(), NodeLifecycle::Stopped);
    }

    #[tokio::test]
    async fn two_nodes_converge_via_gossip() {
        // Node A trains first, node B starts with zeros.
        let node_a = SphereNode::new(1, 4).activate();
        let node_b = SphereNode::new(2, 4).activate();

        // A writes learned weights.
        node_a.write_weights(&[0.1, 0.2, 0.3, 0.4]);

        // A broadcasts directly to B's inbox.
        let b_inbox = node_b.inbox();
        let sent = node_a.broadcast(&[b_inbox]).await;
        assert_eq!(sent, 1);

        // B drains its inbox.
        let updates = node_b.drain_inbox();
        assert_eq!(updates, 4, "all four weight cells should have updated");

        let b_weights = node_b.weight_snapshot();
        assert!(
            (b_weights[0] - 0.1).abs() < 1e-10,
            "weight[0] should have converged to 0.1"
        );
        assert!(
            (b_weights[3] - 0.4).abs() < 1e-10,
            "weight[3] should have converged to 0.4"
        );
    }
}
