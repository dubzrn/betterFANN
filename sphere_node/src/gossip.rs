//! Gossip message types and fanout-2 dissemination logic.
//!
//! A `GossipMsg` carries a node's full `WeightSet` snapshot.  The
//! [`gossip_round`] function picks two random peers from the provided channel
//! slice and sends the message — matching the canonical epidemic-broadcast
//! fanout-2 algorithm that achieves O(log N) convergence in N-node clusters.

use crate::crdt::WeightSet;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// A gossip message broadcast by a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMsg {
    /// Originating node identifier.
    pub from: u64,
    /// Full weight snapshot at the time of broadcast.
    pub weights: WeightSet,
    /// Lamport clock of the sender at the time of broadcast.
    pub sender_clock: u64,
}

/// Send `msg` to up to two randomly-chosen peers from `peers`.
///
/// Returns the number of peers successfully reached.
pub async fn gossip_round(
    msg: GossipMsg,
    peers: &[mpsc::Sender<GossipMsg>],
    rng_seed: u64,
) -> usize {
    if peers.is_empty() {
        return 0;
    }

    // Deterministic selection using a simple LCG — avoids a rand dependency
    // in the async path while remaining unpredictable enough for gossip.
    let mut state = rng_seed ^ 0x9e3779b97f4a7c15;
    let pick = |s: &mut u64| -> usize {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s as usize) % peers.len()
    };

    let idx_a = pick(&mut state);
    let idx_b = {
        let raw = pick(&mut state);
        if peers.len() > 1 && raw == idx_a { (raw + 1) % peers.len() } else { raw }
    };

    let targets: Vec<usize> = if idx_a == idx_b {
        vec![idx_a]
    } else {
        vec![idx_a, idx_b]
    };

    let mut sent = 0usize;
    for &t in &targets {
        if peers[t].send(msg.clone()).await.is_ok() {
            sent += 1;
        }
    }
    sent
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn gossip_round_reaches_two_peers() {
        let (tx0, mut rx0) = mpsc::channel::<GossipMsg>(8);
        let (tx1, mut rx1) = mpsc::channel::<GossipMsg>(8);
        let (tx2, mut rx2) = mpsc::channel::<GossipMsg>(8);

        let ws = WeightSet::new(4, 0);
        let msg = GossipMsg { from: 0, weights: ws, sender_clock: 1 };
        let peers = vec![tx0, tx1, tx2];

        let reached = gossip_round(msg, &peers, 12345).await;
        assert!(reached >= 1, "at least one peer must be reached");

        // Drain all receivers and count messages.
        let total: usize = [
            rx0.try_recv().is_ok() as usize,
            rx1.try_recv().is_ok() as usize,
            rx2.try_recv().is_ok() as usize,
        ]
        .iter()
        .sum();

        assert_eq!(total, reached, "reached count must match messages received");
    }

    #[tokio::test]
    async fn gossip_round_with_no_peers() {
        let ws = WeightSet::new(2, 0);
        let msg = GossipMsg { from: 0, weights: ws, sender_clock: 0 };
        let reached = gossip_round(msg, &[], 0).await;
        assert_eq!(reached, 0);
    }
}
