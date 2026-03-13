//! Pareto-frontier computation for multi-objective topology optimisation.
//!
//! Uses an ndarray matrix to represent the objective space; Pareto dominance
//! is checked pairwise over the (n_candidates × 3) objective matrix.
//!
//! A solution **b** *dominates* solution **a** when b is at least as good as a
//! on every objective and strictly better on at least one.

use ndarray::Array2;
use petgraph::graph::DiGraph;

use crate::objectives::ObjectiveScores;

/// Returns `true` when solution `b` Pareto-dominates solution `a`.
///
/// Domination requires b ≥ a on all three objectives *and* b > a on at least
/// one.  Two solutions with identical scores do **not** dominate each other.
pub fn is_dominated(a: &ObjectiveScores, b: &ObjectiveScores) -> bool {
    let b_geq_a = b.latency >= a.latency
        && b.throughput >= a.throughput
        && b.fault_tolerance >= a.fault_tolerance;

    let b_strictly_better = b.latency > a.latency
        || b.throughput > a.throughput
        || b.fault_tolerance > a.fault_tolerance;

    b_geq_a && b_strictly_better
}

/// Extract the non-dominated (Pareto-optimal) subset from a scored candidate
/// list.
///
/// The function builds an (n × 3) ndarray objective matrix and performs an
/// O(n²) dominance sweep.  All surviving non-dominated candidates are
/// returned in their original order.
pub fn compute_pareto_frontier(
    candidates: Vec<(DiGraph<u32, f64>, ObjectiveScores)>,
) -> Vec<(DiGraph<u32, f64>, ObjectiveScores)> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    // Build objective matrix: rows = candidates, cols = [latency, throughput, ft]
    let mut obj = Array2::<f64>::zeros((n, 3));
    for (i, (_, s)) in candidates.iter().enumerate() {
        obj[[i, 0]] = s.latency;
        obj[[i, 1]] = s.throughput;
        obj[[i, 2]] = s.fault_tolerance;
    }

    let dominated = dominance_mask(&obj);

    candidates
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !dominated[*i])
        .map(|(_, item)| item)
        .collect()
}

/// Return a bool vector where `dominated[i] == true` means candidate `i` is
/// Pareto-dominated by at least one other candidate.
pub(crate) fn frontier_indices(scores: &[ObjectiveScores]) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }

    let mut obj = Array2::<f64>::zeros((n, 3));
    for (i, s) in scores.iter().enumerate() {
        obj[[i, 0]] = s.latency;
        obj[[i, 1]] = s.throughput;
        obj[[i, 2]] = s.fault_tolerance;
    }

    let dominated = dominance_mask(&obj);
    (0..n).filter(|&i| !dominated[i]).collect()
}

/// Core O(n²) dominance sweep over an (n × 3) objective matrix.
fn dominance_mask(obj: &Array2<f64>) -> Vec<bool> {
    let n = obj.nrows();
    let mut dominated = vec![false; n];

    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dominated[j] {
                continue;
            }
            // Check whether j dominates i.
            let j_geq_i = (0..3).all(|k| obj[[j, k]] >= obj[[i, k]]);
            let j_better = (0..3).any(|k| obj[[j, k]] > obj[[i, k]]);
            if j_geq_i && j_better {
                dominated[i] = true;
                break;
            }
        }
    }

    dominated
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::DiGraph;

    fn s(lat: f64, thr: f64, ft: f64) -> ObjectiveScores {
        ObjectiveScores {
            latency: lat,
            throughput: thr,
            fault_tolerance: ft,
        }
    }

    #[test]
    fn dominance_check_basic() {
        let good = s(0.9, 0.9, 0.9);
        let bad = s(0.5, 0.5, 0.5);
        assert!(is_dominated(&bad, &good));
        assert!(!is_dominated(&good, &bad));
    }

    #[test]
    fn equal_scores_do_not_dominate() {
        let a = s(0.5, 0.5, 0.5);
        let b = s(0.5, 0.5, 0.5);
        assert!(!is_dominated(&a, &b));
        assert!(!is_dominated(&b, &a));
    }

    #[test]
    fn incomparable_solutions_both_on_frontier() {
        let a = s(0.9, 0.2, 0.5);
        let b = s(0.2, 0.9, 0.5);
        assert!(!is_dominated(&a, &b));
        assert!(!is_dominated(&b, &a));
    }

    #[test]
    fn frontier_removes_dominated_candidate() {
        let candidates: Vec<(DiGraph<u32, f64>, ObjectiveScores)> = vec![
            (DiGraph::new(), s(0.8, 0.8, 0.8)), // dominates all below
            (DiGraph::new(), s(0.5, 0.5, 0.5)), // dominated
            (DiGraph::new(), s(0.9, 0.2, 0.5)), // not dominated (better latency)
        ];
        let frontier = compute_pareto_frontier(candidates);
        assert_eq!(frontier.len(), 2);
        // Ensure no member of the frontier is dominated by another.
        for i in 0..frontier.len() {
            for j in 0..frontier.len() {
                if i != j {
                    assert!(!is_dominated(&frontier[i].1, &frontier[j].1));
                }
            }
        }
    }
}
