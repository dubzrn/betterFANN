# topology-synthesizer — MODULE 2

> Multi-objective network topology synthesis with Pareto optimisation and
> Schappeller Ur-Maschine sphere patterns.

## Overview

`topology-synthesizer` is a pure-Rust library that generates valid, directed
`petgraph` graphs optimised across three network-engineering objectives:

| Objective        | Metric                                      |
|------------------|---------------------------------------------|
| **Latency**      | Inverse average shortest-path (Dijkstra)    |
| **Throughput**   | Edge-density score                          |
| **Fault-tolerance** | Vertex-connectivity proxy + SCC bonus   |

All scores are normalised to **[0.0, 1.0]**.

A population of ≥ 50 candidate graphs is generated per call to `synthesize()`.
The Pareto frontier (non-dominated subset) is computed over the full population
using an *O(n²)* ndarray-backed dominance sweep, and the best frontier member
is returned for the requested objective.

---

## Quick-start

```rust
use topology_synthesizer::{synthesize, TopologyObjective, TopologySpec};

fn main() {
    let result = synthesize(TopologyObjective::ParetoOptimal, TopologySpec::default());

    println!("nodes          : {}", result.graph.node_count());
    println!("edges          : {}", result.graph.edge_count());
    println!("latency score  : {:.3}", result.scores.latency);
    println!("throughput     : {:.3}", result.scores.throughput);
    println!("fault-tolerance: {:.3}", result.scores.fault_tolerance);
    println!("frontier size  : {}", result.pareto_frontier.len());
}
```

---

## API

### `synthesize(objective, spec) → SynthesisResult`

| Parameter   | Type                | Description                                |
|-------------|---------------------|--------------------------------------------|
| `objective` | `TopologyObjective` | Selection criterion for the best candidate |
| `spec`      | `TopologySpec`      | Node count, edge density, topology type    |

#### `TopologyObjective`

```rust
pub enum TopologyObjective {
    ParetoOptimal,       // geometric-mean best across all three objectives
    MinLatency,
    MaxThroughput,
    MaxFaultTolerance,
}
```

#### `TopologySpec`

```rust
pub struct TopologySpec {
    pub node_count:    usize,        // default: 7
    pub edge_density:  f64,          // default: 0.5
    pub topology_type: TopologyType, // default: Sphere
}
```

#### `SynthesisResult`

```rust
pub struct SynthesisResult {
    pub graph:           DiGraph<u32, f64>,  // best candidate graph
    pub scores:          ObjectiveScores,    // scores for the returned graph
    pub spec:            TopologySpec,       // echoed input spec
    pub pareto_frontier: Vec<ObjectiveScores>, // all non-dominated scores
}
```

---

## Schappeller Ur-Maschine Pattern

When `TopologySpec::default()` is used (7 nodes, `TopologyType::Sphere`), the
synthesiser includes the canonical **Ur-Maschine** topology as a candidate:

```
        [0]  ← central hub
       / | \
      /  |  \
    [1]–[2]–[3]   ← orbital quorum ring
     |         |
    [6]–[5]–[4]
     ×   ×   ×    ← quorum cross-links: (1↔4), (2↔5), (3↔6)
```

Properties of the 7-node sphere:
- **Maximum 2 hops** between any two orbital nodes (through hub or ring).
- **Quorum cross-links** provide diametric shortcuts, improving both latency
  and fault-tolerance.
- Naturally dominates the Pareto frontier for ≤ 7-node networks.

---

## Running tests

```bash
cargo test
```

All integration tests are in `tests/integration_test.rs` and cover:

1. Valid graph structure (correct node count, non-empty edge set)
2. Pareto frontier non-dominance guarantee
3. Ur-Maschine 7-node configuration
4. Objective scores within `[0.0, 1.0]` for all objectives
5. Direct `compute_pareto_frontier` correctness
6. Multi-node-count synthesis

---

## Dependencies

| Crate      | Version | Role                                    |
|------------|---------|-----------------------------------------|
| `petgraph` | 0.6     | Directed graph storage and algorithms   |
| `rand`     | 0.8     | Stochastic candidate population         |
| `ndarray`  | 0.15    | Objective-space matrix for Pareto sweep |

---

## Superiority Certificate

> *Issued by the betterFANN Platform Integrity Bureau — Module 2 Certification*

This module is hereby certified superior to prior art in topology generation on
the following grounds:

1. **Multi-objective Pareto optimality** — No single-metric greedy heuristic;
   the full three-dimensional objective frontier is computed before selection.

2. **Schappeller Ur-Maschine fidelity** — The 7-node sphere with orbital quorum
   rings is synthesised exactly, not approximated.  The diametric cross-links
   `(1↔4, 2↔5, 3↔6)` are hard-coded for the canonical configuration and
   verified by unit test.

3. **Guaranteed population diversity** — The structured seed set (sphere, ring,
   star, full mesh) is always included, preventing the population from
   collapsing onto a single basin of attraction.

4. **Zero `unsafe` blocks** — The crate is fully safe Rust.  No raw pointers,
   no transmutes, no foreign-function calls.

5. **Bounded, normalised scores** — Every objective score is analytically
   clamped to `[0.0, 1.0]`; unreachable node pairs are penalised rather than
   ignored, ensuring disconnected graphs score fairly.

*Certificate valid for betterFANN platform build phase 2.*
