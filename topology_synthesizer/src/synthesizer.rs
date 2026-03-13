//! Core synthesiser — topology generation, objective scoring, and Pareto
//! selection.
//!
//! # Quick-start
//! ```rust
//! use topology_synthesizer::{synthesize, TopologyObjective, TopologySpec};
//!
//! let result = synthesize(TopologyObjective::ParetoOptimal, TopologySpec::default());
//! assert_eq!(result.graph.node_count(), 7);
//! assert!(result.scores.latency > 0.0);
//! ```

use petgraph::graph::DiGraph;
use rand::Rng;

use crate::objectives::{evaluate_objectives, ObjectiveScores};
use crate::pareto::frontier_indices;

// ─── Public types ────────────────────────────────────────────────────────────

/// Optimisation objective passed to [`synthesize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyObjective {
    /// Select the graph that maximises the geometric mean of all three
    /// objectives (full Pareto-optimal trade-off).
    ParetoOptimal,
    /// Minimise average shortest-path latency.
    MinLatency,
    /// Maximise edge-density throughput.
    MaxThroughput,
    /// Maximise fault-tolerance (vertex connectivity proxy).
    MaxFaultTolerance,
}

/// Shape of the topology to synthesise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyType {
    /// Unconstrained random graph biased toward the given edge density.
    Random,
    /// Schappeller Ur-Maschine sphere: central hub + orbital quorum ring.
    /// This is the canonical 7-node default.
    Sphere,
    /// Bidirectional ring lattice.
    Ring,
    /// All-pairs directed mesh.
    FullMesh,
    /// Single-hub star with bidirectional spokes.
    Star,
}

impl Default for TopologyType {
    fn default() -> Self {
        TopologyType::Sphere
    }
}

/// Parameters that describe the desired topology.
#[derive(Debug, Clone)]
pub struct TopologySpec {
    /// Number of nodes in the synthesised graph.
    pub node_count: usize,
    /// Target edge density in (0.0, 1.0]; used to seed random candidates.
    pub edge_density: f64,
    /// Preferred structural pattern; guides the candidate pool.
    pub topology_type: TopologyType,
}

impl Default for TopologySpec {
    /// Ur-Maschine default: 7-node sphere with 0.5 edge-density target.
    fn default() -> Self {
        TopologySpec {
            node_count: 7,
            edge_density: 0.5,
            topology_type: TopologyType::Sphere,
        }
    }
}

/// The output of a [`synthesize`] call.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Best candidate graph selected from the Pareto frontier.
    pub graph: DiGraph<u32, f64>,
    /// Objective scores for the returned graph.
    pub scores: ObjectiveScores,
    /// Spec used for synthesis (echoed back for convenience).
    pub spec: TopologySpec,
    /// Scores of every non-dominated solution found during synthesis.
    /// At least one entry is always present.
    pub pareto_frontier: Vec<ObjectiveScores>,
}

// ─── Public entry-point ──────────────────────────────────────────────────────

/// Synthesise a topology graph for the given objective and spec.
///
/// # Algorithm
/// 1. Generate a population of ≥ 50 candidate graphs (structured + random).
/// 2. Evaluate each candidate on all three objectives.
/// 3. Compute the Pareto frontier (non-dominated subset).
/// 4. Return the frontier member that best satisfies `objective`.
pub fn synthesize(objective: TopologyObjective, spec: TopologySpec) -> SynthesisResult {
    let mut rng = rand::thread_rng();
    // Guarantee at least 50 candidates regardless of node count.
    let population_size = 50.max(spec.node_count * 8);

    let graphs = generate_population(&spec, population_size, &mut rng);
    let scores: Vec<ObjectiveScores> = graphs.iter().map(evaluate_objectives).collect();

    let f_indices = frontier_indices(&scores);
    let pareto_frontier: Vec<ObjectiveScores> = f_indices.iter().map(|&i| scores[i].clone()).collect();

    let best_idx = if f_indices.is_empty() {
        // Pathological edge-case: pick the globally best by objective.
        best_index_by_objective(&scores, &objective)
    } else {
        best_index_from_subset(&scores, &f_indices, &objective)
    };

    SynthesisResult {
        graph: graphs[best_idx].clone(),
        scores: scores[best_idx].clone(),
        spec,
        pareto_frontier,
    }
}

// ─── Population generation ───────────────────────────────────────────────────

fn generate_population(
    spec: &TopologySpec,
    size: usize,
    rng: &mut impl Rng,
) -> Vec<DiGraph<u32, f64>> {
    let n = spec.node_count;
    let mut pop: Vec<DiGraph<u32, f64>> = Vec::with_capacity(size);

    // Always include all deterministic structured patterns.
    pop.push(build_sphere(n));
    pop.push(build_ring(n));
    pop.push(build_star(n));

    // Full mesh has O(n²) directed edges; cap at n≤14 (≤182 edges) to avoid
    // excessive memory and scoring overhead for larger populations.
    if n <= 14 {
        pop.push(build_full_mesh(n));
    }

    // Sweep a grid of densities to ensure broad coverage of the objective space.
    let density_steps: Vec<f64> = (1..=9).map(|i| i as f64 * 0.1).collect();
    for &d in &density_steps {
        pop.push(build_random_graph(n, d, rng));
    }

    // Fill remainder with random graphs centred on the spec's edge density,
    // using Gaussian-ish jitter via multiple uniform samples.
    while pop.len() < size {
        let jitter: f64 = (rng.gen::<f64>() + rng.gen::<f64>() + rng.gen::<f64>()) / 3.0;
        let density = (spec.edge_density + jitter - 0.5).clamp(0.05, 0.95);
        pop.push(build_random_graph(n, density, rng));
    }

    pop
}

// ─── Structured topology builders ────────────────────────────────────────────

/// Schappeller Ur-Maschine sphere topology.
///
/// For the canonical `n = 7` case:
/// * Node 0 is the central hub, bidirectionally connected to all six orbitals.
/// * Nodes 1–6 form a bidirectional orbital ring (1↔2↔3↔4↔5↔6↔1).
/// * Quorum cross-links connect diametrically opposite pairs:
///   (1↔4), (2↔5), (3↔6).
///
/// For general `n`, the hub-and-ring structure is maintained without the
/// diametric cross-links (which are only well-defined for even-orbital counts).
pub(crate) fn build_sphere(n: usize) -> DiGraph<u32, f64> {
    let mut g: DiGraph<u32, f64> = DiGraph::new();
    let nodes: Vec<_> = (0..n as u32).map(|i| g.add_node(i)).collect();

    if n < 2 {
        return g;
    }

    let center = nodes[0];

    // Hub ↔ every orbital.
    for node in nodes.iter().skip(1) {
        g.add_edge(center, *node, 1.0);
        g.add_edge(*node, center, 1.0);
    }

    // Bidirectional orbital ring: nodes[1] ↔ nodes[2] ↔ … ↔ nodes[n-1] ↔ nodes[1].
    for i in 1..n {
        let next = if i == n - 1 { 1 } else { i + 1 };
        g.add_edge(nodes[i], nodes[next], 1.0);
        g.add_edge(nodes[next], nodes[i], 1.0);
    }

    // Quorum cross-links for the 7-node Ur-Maschine (3 diametric pairs).
    if n == 7 {
        let pairs = [(1usize, 4usize), (2, 5), (3, 6)];
        for (a, b) in pairs {
            g.add_edge(nodes[a], nodes[b], 1.0);
            g.add_edge(nodes[b], nodes[a], 1.0);
        }
    }

    g
}

fn build_ring(n: usize) -> DiGraph<u32, f64> {
    let mut g: DiGraph<u32, f64> = DiGraph::new();
    let nodes: Vec<_> = (0..n as u32).map(|i| g.add_node(i)).collect();
    for i in 0..n {
        let next = (i + 1) % n;
        g.add_edge(nodes[i], nodes[next], 1.0);
        g.add_edge(nodes[next], nodes[i], 1.0);
    }
    g
}

fn build_star(n: usize) -> DiGraph<u32, f64> {
    let mut g: DiGraph<u32, f64> = DiGraph::new();
    let nodes: Vec<_> = (0..n as u32).map(|i| g.add_node(i)).collect();
    if n < 2 {
        return g;
    }
    let center = nodes[0];
    for node in nodes.iter().skip(1) {
        g.add_edge(center, *node, 1.0);
        g.add_edge(*node, center, 1.0);
    }
    g
}

fn build_full_mesh(n: usize) -> DiGraph<u32, f64> {
    let mut g: DiGraph<u32, f64> = DiGraph::new();
    let nodes: Vec<_> = (0..n as u32).map(|i| g.add_node(i)).collect();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                g.add_edge(nodes[i], nodes[j], 1.0);
            }
        }
    }
    g
}

fn build_random_graph(n: usize, density: f64, rng: &mut impl Rng) -> DiGraph<u32, f64> {
    let mut g: DiGraph<u32, f64> = DiGraph::new();
    let nodes: Vec<_> = (0..n as u32).map(|i| g.add_node(i)).collect();

    if n < 2 {
        return g;
    }

    // Guarantee strong connectivity via a random directed spanning tree.
    for i in 1..n {
        let parent = rng.gen_range(0..i);
        g.add_edge(nodes[parent], nodes[i], 1.0);
        g.add_edge(nodes[i], nodes[parent], 1.0);
    }

    // Probabilistically add remaining edges according to `density`.
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.gen_bool(density) && !g.contains_edge(nodes[i], nodes[j]) {
                g.add_edge(nodes[i], nodes[j], 1.0);
            }
        }
    }

    g
}

// ─── Selection helpers ────────────────────────────────────────────────────────

fn best_index_by_objective(scores: &[ObjectiveScores], objective: &TopologyObjective) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            scalar_objective(a, objective)
                .partial_cmp(&scalar_objective(b, objective))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn best_index_from_subset(
    scores: &[ObjectiveScores],
    indices: &[usize],
    objective: &TopologyObjective,
) -> usize {
    indices
        .iter()
        .copied()
        .max_by(|&a, &b| {
            scalar_objective(&scores[a], objective)
                .partial_cmp(&scalar_objective(&scores[b], objective))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

/// Reduce multi-objective scores to a single scalar for final selection.
fn scalar_objective(s: &ObjectiveScores, objective: &TopologyObjective) -> f64 {
    match objective {
        TopologyObjective::ParetoOptimal => {
            // Geometric mean ensures balanced trade-offs.
            (s.latency * s.throughput * s.fault_tolerance).powf(1.0 / 3.0)
        }
        TopologyObjective::MinLatency => s.latency,
        TopologyObjective::MaxThroughput => s.throughput,
        TopologyObjective::MaxFaultTolerance => s.fault_tolerance,
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_7_has_correct_structure() {
        let g = build_sphere(7);
        assert_eq!(g.node_count(), 7);
        // Hub (node 0): 6 outgoing + 6 incoming = 12 hub edges
        // Ring: 6 bidirectional = 12 ring edges
        // Quorum cross-links: 3 bidirectional = 6 cross edges
        assert_eq!(g.edge_count(), 30);
    }

    #[test]
    fn ring_is_strongly_connected() {
        let g = build_ring(5);
        assert_eq!(g.node_count(), 5);
        let sccs = petgraph::algo::tarjan_scc(&g);
        assert_eq!(sccs.len(), 1);
    }

    #[test]
    fn generate_population_meets_minimum_size() {
        let spec = TopologySpec::default();
        let mut rng = rand::thread_rng();
        let pop = generate_population(&spec, 50, &mut rng);
        assert!(pop.len() >= 50);
        for g in &pop {
            assert_eq!(g.node_count(), spec.node_count);
        }
    }

    #[test]
    fn synthesize_returns_correct_node_count() {
        let result = synthesize(TopologyObjective::ParetoOptimal, TopologySpec::default());
        assert_eq!(result.graph.node_count(), 7);
    }

    #[test]
    fn all_single_objectives_return_valid_scores() {
        for obj in [
            TopologyObjective::MinLatency,
            TopologyObjective::MaxThroughput,
            TopologyObjective::MaxFaultTolerance,
        ] {
            let r = synthesize(obj, TopologySpec::default());
            assert!((0.0..=1.0).contains(&r.scores.latency));
            assert!((0.0..=1.0).contains(&r.scores.throughput));
            assert!((0.0..=1.0).contains(&r.scores.fault_tolerance));
        }
    }
}
