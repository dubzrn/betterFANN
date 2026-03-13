//! Objective scoring functions for synthesised topologies.
//!
//! Every score is normalised to **[0.0, 1.0]** where higher is better.
//! Latency score  — lower average shortest-path → higher score.
//! Throughput score — higher edge density → higher score.
//! Fault-tolerance score — higher minimum degree + strong connectivity bonus.

use petgraph::algo::{dijkstra, tarjan_scc};
use petgraph::graph::DiGraph;
use petgraph::Direction;

/// Scores for the three network-engineering objectives.
#[derive(Debug, Clone)]
pub struct ObjectiveScores {
    /// Inverse average-shortest-path score, ∈ [0, 1].
    pub latency: f64,
    /// Edge-density score, ∈ [0, 1].
    pub throughput: f64,
    /// Vertex-connectivity approximation score, ∈ [0, 1].
    pub fault_tolerance: f64,
}

/// Evaluate all three objectives for a single candidate graph.
pub fn evaluate_objectives(graph: &DiGraph<u32, f64>) -> ObjectiveScores {
    ObjectiveScores {
        latency: score_latency(graph),
        throughput: score_throughput(graph),
        fault_tolerance: score_fault_tolerance(graph),
    }
}

/// Compute a latency score from average shortest-path length.
///
/// Unreachable ordered pairs are penalised with a path-length equal to `n`
/// (an upper bound that exceeds any real shortest path in a connected graph).
fn score_latency(graph: &DiGraph<u32, f64>) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return 1.0;
    }

    let n_pairs = n * (n - 1);
    let mut sum_dist = 0.0_f64;
    let mut reachable: usize = 0;

    for source in graph.node_indices() {
        let dists = dijkstra(graph, source, None, |e| *e.weight());
        for (&target, &d) in &dists {
            if target != source {
                sum_dist += d;
                reachable += 1;
            }
        }
    }

    if reachable == 0 {
        return 0.0;
    }

    // Penalise each unreachable pair as if it had distance `n`.
    let unreachable = n_pairs - reachable;
    let penalised_sum = sum_dist + unreachable as f64 * n as f64;
    // Worst-case for a fully-connected graph: every pair takes (n-1) hops.
    let max_possible = n_pairs as f64 * (n - 1) as f64;

    (1.0 - penalised_sum / max_possible).clamp(0.0, 1.0)
}

/// Edge-density score: actual edges divided by maximum possible directed edges.
fn score_throughput(graph: &DiGraph<u32, f64>) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return 1.0;
    }
    let e = graph.edge_count() as f64;
    let max_edges = (n * (n - 1)) as f64;
    (e / max_edges).clamp(0.0, 1.0)
}

/// Fault-tolerance score based on minimum degree and strong connectivity.
///
/// Uses `min(in_degree, out_degree)` across all nodes as a lower bound on
/// vertex connectivity, normalised to (0, 0.8]. A strongly-connected graph
/// gains an additional 0.2 bonus, capping the score at 1.0.
fn score_fault_tolerance(graph: &DiGraph<u32, f64>) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return 1.0;
    }

    let min_deg = graph
        .node_indices()
        .map(|v| {
            let in_d = graph.edges_directed(v, Direction::Incoming).count();
            let out_d = graph.edges_directed(v, Direction::Outgoing).count();
            in_d.min(out_d)
        })
        .min()
        .unwrap_or(0);

    let sccs = tarjan_scc(graph);
    let connectivity_bonus = if sccs.len() == 1 { 0.2_f64 } else { 0.0 };

    let deg_score = (min_deg as f64 / (n - 1) as f64).clamp(0.0, 0.8);
    (deg_score + connectivity_bonus).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::DiGraph;

    fn triangle() -> DiGraph<u32, f64> {
        let mut g = DiGraph::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(c, a, 1.0);
        g
    }

    #[test]
    fn scores_are_in_unit_interval() {
        let g = triangle();
        let s = evaluate_objectives(&g);
        assert!((0.0..=1.0).contains(&s.latency));
        assert!((0.0..=1.0).contains(&s.throughput));
        assert!((0.0..=1.0).contains(&s.fault_tolerance));
    }

    #[test]
    fn isolated_node_returns_perfect_scores() {
        let mut g: DiGraph<u32, f64> = DiGraph::new();
        g.add_node(0);
        let s = evaluate_objectives(&g);
        assert_eq!(s.latency, 1.0);
        assert_eq!(s.throughput, 1.0);
        assert_eq!(s.fault_tolerance, 1.0);
    }
}
