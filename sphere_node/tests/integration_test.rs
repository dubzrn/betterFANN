//! Integration tests for `sphere_node`.
//!
//! 1. Three-node cluster convergence via gossip.
//! 2. LWW CRDT causality: stale update rejected.
//! 3. Lifecycle state machine: Bootstrapping → Active → Draining → Stopped.

use sphere_node::{NodeLifecycle, SphereNode, WeightCell, WeightSet};

// ── 1. Cluster convergence ────────────────────────────────────────────────────

/// Build a three-node cluster where node A holds the freshest weights and
/// verify that after two gossip rounds all nodes converge to A's values.
#[tokio::test]
async fn three_node_cluster_convergence() {
    let node_a = SphereNode::new(1, 8).activate();
    let node_b = SphereNode::new(2, 8).activate();
    let node_c = SphereNode::new(3, 8).activate();

    let target: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    node_a.write_weights(&target);

    // Round 1: A → B and A → C.
    let inboxes = vec![node_b.inbox(), node_c.inbox()];
    let sent = node_a.broadcast(&inboxes).await;
    assert!(sent >= 1, "A must reach at least one peer");

    node_b.drain_inbox();
    node_c.drain_inbox();

    // Round 2: B → C (so C catches up even if A missed it).
    let c_inbox = vec![node_c.inbox()];
    node_b.broadcast(&c_inbox).await;
    node_c.drain_inbox();

    // All three should converge (at least the cells that received updates).
    let b_vals = node_b.weight_snapshot();
    let c_vals = node_c.weight_snapshot();

    for i in 0..8 {
        assert!(
            (b_vals[i] - target[i]).abs() < 1e-10,
            "node_b weight[{i}] = {} expected {}",
            b_vals[i],
            target[i]
        );
        assert!(
            (c_vals[i] - target[i]).abs() < 1e-10,
            "node_c weight[{i}] = {} expected {}",
            c_vals[i],
            target[i]
        );
    }
}

// ── 2. LWW causality ─────────────────────────────────────────────────────────

/// A stale write (lower clock) must never overwrite a fresher one.
#[test]
fn stale_update_is_rejected() {
    let fresh = WeightCell { value: 99.0, clock: 10, origin: 1 };
    let stale = WeightCell { value: 0.0, clock: 2, origin: 2 };

    let mut ws = WeightSet { cells: vec![fresh] };
    let remote = WeightSet { cells: vec![stale] };

    let updates = ws.merge(&remote);
    assert_eq!(updates, 0, "stale write must not update any cell");
    assert_eq!(ws.cells[0].value, 99.0);
}

// ── 3. Lifecycle state machine ────────────────────────────────────────────────

#[test]
fn full_lifecycle_transition() {
    let node = SphereNode::new(7, 4);
    assert_eq!(node.lifecycle(), NodeLifecycle::Bootstrapping);

    let node = node.activate();
    assert_eq!(node.lifecycle(), NodeLifecycle::Active);

    let node = node.drain();
    assert_eq!(node.lifecycle(), NodeLifecycle::Draining);

    let node = node.stop();
    assert_eq!(node.lifecycle(), NodeLifecycle::Stopped);
}
