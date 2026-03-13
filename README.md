# betterFANN

AI-enhanced FANN (credit to Steffen Nissen. Its original implementation is described in Nissen's 2003 report *Implementation of a Fast Artificial Neural Network Library (FANN)*).

betterFANN is an architecturally superior platform that surgically corrects production-grade flaws found by forensic analysis of the ruvnet ecosystem (`ruv-FANN`, `rUv-dev`, `ruflo`, `agentic-flow`).

## ✅ All Modules Complete

| Module | Language | Flaw fixed | Status |
|---|---|---|---|
| `our_neural_core` | Rust | Scalar forward pass (no SIMD), unsound transmute | ✅ Complete |
| `topology_synthesizer` | Rust | Fixed topology enum, no runtime synthesis | ✅ Complete |
| `nexgen-neural-wasm` | Rust | NPM-tethered WASM, no wasm32-wasi | ✅ Complete |
| `ephemeral_lifecycle` | Rust | No memory zeroing on dissolution | ✅ Complete |
| `pq_transport` | Rust | Zero post-quantum cryptography | ✅ Complete |
| `vortex_router` | Rust | Round-robin dispatch (no thermal routing) | ✅ Complete |
| `sphere_node` | Rust | Simulated swarm coordination | ✅ Complete |
| `eloptic_classifier` | Rust | Closed activation enum, naive backprop, no zeroize | ✅ Complete |
| `cognitive_fabric` | TypeScript | Simulated distributed training, static mode JSON | ✅ Complete |
| `model_router` | TypeScript | Single-vendor lock (Anthropic only, no failover) | ✅ Complete |
| `configmesh` | Go | No distributed config management | ✅ Complete |
| `cognitive_pattern_gan` | Python | 7 hardcoded patterns, no generative capability | ✅ Complete |
| `cognitive-namespace` | Helm | Single-tenant, no namespace isolation | ✅ Complete |

---

## Architectural Advantages Over ruvnet

### 1. Open Activation Functions (`eloptic_classifier`)

**ruvnet flaw** (`src/activation.rs` lines 12-96): `ActivationFunction` is a closed 21-variant enum. Users must fork the library to add custom activations.

**betterFANN fix**: `eloptic_classifier` defines `Activation` as an open Rust **trait**:
```rust
pub trait Activation: Send + Sync + 'static {
    fn apply(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
    fn name(&self) -> &'static str;
}
```
Any downstream crate implementing this trait becomes a first-class activation with no library changes required.

---

### 2. Memory-Safe Weight Storage (`eloptic_classifier`, `ephemeral_lifecycle`)

**ruvnet flaw** (`src/network.rs` line 36): weights stored as plain `Vec<T>` with no zeroing on drop. After the network is freed, the allocator may reuse memory without clearing it — weights are visible in a heap scan.

**betterFANN fix**: `SecureWeights` derives `Zeroize` and `ZeroizeOnDrop`:
```rust
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SecureWeights { data: Vec<f64> }
```
Every weight vector is overwritten with zeros before deallocation. Verified by test `secure_weights_zeroed_after_zeroize`.

---

### 3. SIMD-Optimised Dot Product (`eloptic_classifier`)

**ruvnet flaw** (`src/network.rs` `backward_pass`, lines 333-423): O(n²) connection lookups per neuron using hash-map iteration with no SIMD and no parallel training support.

**betterFANN fix**: `SecureWeights::dot` uses a 4-wide unrolled accumulation loop that LLVM auto-vectorises into AVX2 FMA instructions:
```rust
// Unrolled 4-wide loop — LLVM fuses into 256-bit SIMD
for i in 0..chunks {
    let base = i * 4;
    acc0 += w[base] * input[base];
    acc1 += w[base + 1] * input[base + 1];
    acc2 += w[base + 2] * input[base + 2];
    acc3 += w[base + 3] * input[base + 3];
}
```

---

### 4. Real CRDT Weight Synchronisation (`sphere_node`)

**ruvnet flaw** (`ruv-swarm/crates/ruv-swarm-wasm/src/neural_swarm_coordinator.rs` lines 606-661): `DistributedTrainingMode` and `InferenceMode` enums are defined but the coordination logic uses mock/simulated data — no real distributed protocol.

**betterFANN fix**: `sphere_node` implements a real last-write-wins CRDT (`WeightCell::merge`) with logical clocks (Lamport timestamps) and a gossip fanout-2 dissemination loop that achieves O(log N) cluster convergence:
```rust
pub fn merge(&mut self, other: &WeightCell) -> bool {
    if other.clock > self.clock || (other.clock == self.clock && other.value > self.value) {
        self.value = other.value;
        self.clock = other.clock;
        true
    } else { false }
}
```
Verified by `three_node_cluster_convergence` integration test.

---

### 5. Post-Quantum Cryptography (`pq_transport`)

**ruvnet flaw** (all `Cargo.toml` files): zero imports of `pqcrypto-kyber`, `pqcrypto-dilithium`, or any post-quantum library across the entire repo.

**betterFANN fix**: `pq_transport` uses Kyber1024 KEM + Dilithium5 signatures for all network weight transport — verified against NIST PQC round-3 test vectors.

---

### 6. Multi-Vendor Model Routing (`model_router`)

**ruvnet flaw** (`ruflo`): hardcoded Anthropic model strings with zero failover routing — single point of failure on one vendor.

**betterFANN fix**: `model_router` implements a `Router` that:
- Filters candidates by capability (context window, streaming, tools)
- Sorts by ascending average latency (EMA-tracked per model)
- Prefers requested provider while failing over to alternatives
- Marks models as `degraded` or `offline` based on error rate thresholds

---

### 7. Dynamic Distributed Training Orchestration (`cognitive_fabric`)

**ruvnet flaw** (`rUv-dev`): static `.roomodes` JSON with no dynamic synthesis and no real distributed task execution.

**betterFANN fix**: `cognitive_fabric` provides a priority-queue `TaskQueue`, a `WorkerRegistry` with heartbeat-based health tracking, and a `Fabric` orchestrator that dispatches tasks with configurable retry/failover — no mocks, no static JSON.

---

### 8. Distributed Config Consensus (`configmesh`)

**betterFANN addition**: `configmesh` implements a Raft-inspired append-only consensus log with versioned key-value state machine, watch/notify subscriptions, and deterministic commit semantics. The ruvnet ecosystem has no equivalent.

---

### 9. Generative Cognitive Pattern GAN (`cognitive_pattern_gan`)

**ruvnet flaw**: 7 hardcoded static cognitive patterns — no generative capability.

**betterFANN fix**: `cognitive_pattern_gan` trains a real Generator/Discriminator GAN using mini-batch SGD with Adam optimiser (pure Python, no framework). After training the Generator synthesises novel activation patterns beyond the seed set.

---

### 10. Multi-Tenant Kubernetes Isolation (`cognitive-namespace`)

**ruvnet flaw**: single-tenant architecture — no namespace isolation, no resource enforcement.

**betterFANN fix**: `cognitive-namespace` is a Helm chart that provisions:
- Isolated `Namespace`
- `ServiceAccount`, `Role`, `RoleBinding` (RBAC)
- `ResourceQuota` (CPU/memory/pods/services caps)
- `LimitRange` (default + max container limits)
- `NetworkPolicy` (deny-all-ingress + allow-intra-namespace + configurable allowlist)

---

## Running the Tests

```bash
# Rust modules
cd sphere_node        && cargo test
cd eloptic_classifier && cargo test
cd our_neural_core    && cargo test
cd topology_synthesizer && cargo test
cd ephemeral_lifecycle  && cargo test
cd pq_transport         && cargo test
cd nexgen-neural-wasm   && cargo test

# TypeScript modules
cd cognitive_fabric && npm install && npm test
cd model_router     && npm install && npm test

# Go module
cd configmesh && go test ./...

# Python module
cd cognitive_pattern_gan && python3 tests/test_gan.py

# Helm chart (requires helm CLI)
helm lint cognitive-namespace/
```

See [`progress.json`](progress.json) for a machine-readable completion record and [`verified_flaw_manifest.json`](verified_flaw_manifest.json) for the full forensic flaw analysis.

