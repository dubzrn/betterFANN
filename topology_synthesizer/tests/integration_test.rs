//! Integration tests for the topology-synthesizer crate.

use petgraph::graph::DiGraph;
use topology_synthesizer::{
    compute_pareto_frontier, is_dominated, synthesize, ObjectiveScores, TopologyObjective,
    TopologySpec, TopologyType,
};

// ─── Helper ──────────────────────────────────────────────────────────────────

fn scores(lat: f64, thr: f64, ft: f64) -> ObjectiveScores {
    ObjectiveScores {
        latency: lat,
        throughput: thr,
        fault_tolerance: ft,
    }
}

// ─── 1. ParetoOptimal synthesis produces a valid graph ───────────────────────

#[test]
fn pareto_optimal_synthesis_produces_valid_graph() {
    let spec = TopologySpec::default();
    let result = synthesize(TopologyObjective::ParetoOptimal, spec.clone());

    // Graph has the expected number of nodes.
    assert_eq!(
        result.graph.node_count(),
        spec.node_count,
        "graph node count should match spec"
    );

    // Graph has at least one edge.
    assert!(
        result.graph.edge_count() > 0,
        "synthesised graph must have at least one edge"
    );

    // The Pareto frontier is non-empty.
    assert!(
        !result.pareto_frontier.is_empty(),
        "Pareto frontier must contain at least one solution"
    );
}

// ─── 2. Pareto frontier contains only non-dominated solutions ─────────────────

#[test]
fn pareto_frontier_contains_only_non_dominated_solutions() {
    let result = synthesize(TopologyObjective::ParetoOptimal, TopologySpec::default());
    let frontier = &result.pareto_frontier;

    assert!(
        !frontier.is_empty(),
        "frontier must be non-empty after synthesis"
    );

    // No member of the frontier must be dominated by another member.
    for i in 0..frontier.len() {
        for j in 0..frontier.len() {
            if i != j {
                assert!(
                    !is_dominated(&frontier[i], &frontier[j]),
                    "frontier solution {} is dominated by {} — frontier is invalid",
                    i,
                    j
                );
            }
        }
    }
}

// ─── 3. Ur-Maschine 7-node configuration ─────────────────────────────────────

#[test]
fn ur_maschine_default_spec_is_7_node_sphere() {
    let spec = TopologySpec::default();

    // Default spec must encode the Ur-Maschine parameters.
    assert_eq!(spec.node_count, 7, "Ur-Maschine default must have 7 nodes");
    assert_eq!(
        spec.topology_type,
        TopologyType::Sphere,
        "Ur-Maschine default must use Sphere topology"
    );
    assert!(
        (spec.edge_density - 0.5).abs() < f64::EPSILON,
        "Ur-Maschine default edge density must be 0.5"
    );

    let result = synthesize(TopologyObjective::ParetoOptimal, spec);

    // Synthesised graph must have exactly 7 nodes.
    assert_eq!(result.graph.node_count(), 7);

    // Graph must have at least as many edges as the sphere hub-ring baseline
    // (12 hub + 12 ring = 24 directed edges minimum).
    assert!(
        result.graph.edge_count() >= 24,
        "7-node sphere must have ≥ 24 directed edges, got {}",
        result.graph.edge_count()
    );
}

// ─── 4. All objective scores are within [0.0, 1.0] ───────────────────────────

#[test]
fn all_objective_scores_are_within_unit_bounds() {
    let objectives = [
        TopologyObjective::ParetoOptimal,
        TopologyObjective::MinLatency,
        TopologyObjective::MaxThroughput,
        TopologyObjective::MaxFaultTolerance,
    ];

    for objective in objectives {
        let result = synthesize(objective, TopologySpec::default());

        assert!(
            (0.0..=1.0).contains(&result.scores.latency),
            "latency score out of bounds: {}",
            result.scores.latency
        );
        assert!(
            (0.0..=1.0).contains(&result.scores.throughput),
            "throughput score out of bounds: {}",
            result.scores.throughput
        );
        assert!(
            (0.0..=1.0).contains(&result.scores.fault_tolerance),
            "fault_tolerance score out of bounds: {}",
            result.scores.fault_tolerance
        );

        // Frontier scores must also be within bounds.
        for s in &result.pareto_frontier {
            assert!((0.0..=1.0).contains(&s.latency));
            assert!((0.0..=1.0).contains(&s.throughput));
            assert!((0.0..=1.0).contains(&s.fault_tolerance));
        }
    }
}

// ─── 5. compute_pareto_frontier eliminates dominated candidates ───────────────

#[test]
fn compute_pareto_frontier_eliminates_dominated_candidate() {
    // Candidate 0: (0.8, 0.8, 0.8) — dominates candidate 1.
    // Candidate 1: (0.5, 0.5, 0.5) — dominated.
    // Candidate 2: (0.9, 0.2, 0.5) — not dominated (leads on latency).
    let candidates: Vec<(DiGraph<u32, f64>, ObjectiveScores)> = vec![
        (DiGraph::new(), scores(0.8, 0.8, 0.8)),
        (DiGraph::new(), scores(0.5, 0.5, 0.5)),
        (DiGraph::new(), scores(0.9, 0.2, 0.5)),
    ];

    let frontier = compute_pareto_frontier(candidates);

    assert_eq!(
        frontier.len(),
        2,
        "dominated candidate must be excluded; expected 2 frontier members, got {}",
        frontier.len()
    );

    // Verify mutual non-dominance within the frontier.
    for i in 0..frontier.len() {
        for j in 0..frontier.len() {
            if i != j {
                assert!(
                    !is_dominated(&frontier[i].1, &frontier[j].1),
                    "frontier member {} is dominated by {} — invalid frontier",
                    i,
                    j
                );
            }
        }
    }
}

// ─── 6. Synthesis works for non-default specs ─────────────────────────────────

#[test]
fn synthesis_works_for_various_node_counts() {
    for n in [3usize, 5, 10, 15] {
        let spec = TopologySpec {
            node_count: n,
            edge_density: 0.4,
            topology_type: TopologyType::Random,
        };
        let result = synthesize(TopologyObjective::MaxFaultTolerance, spec);

        assert_eq!(result.graph.node_count(), n);
        assert!((0.0..=1.0).contains(&result.scores.fault_tolerance));
    }
}
