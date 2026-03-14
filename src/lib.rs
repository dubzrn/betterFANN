//! betterFANN — vortex physics-inspired redesign of FANN.
//!
//! A ground-up reimplementation of the Fast Artificial Neural Network
//! library (FANN) by Steffen Nissen, extended with CRDT-based weight
//! synchronisation, post-quantum secure channels, and WASM support.
//!
//! Individual modules are published as separate crates:
//! - `our-neural-core` — centripetal inference engine
//! - `topology-synthesizer` — Pareto-optimised topology synthesis
//! - `nexgen-neural-wasm` — WASM Component Model inference target
//! - `sphere-node` — distributed node with CRDT weight sync
//! - `eloptic_classifier` — SIMD-optimised trait-based classifier
//! - `ephemeral-lifecycle` — secure weight zeroing on drop
//! - `pq-transport` — post-quantum secure channel (Kyber1024 + Dilithium5)
//! - `vortex_router` — vortex-physics routing layer
