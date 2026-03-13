# nexgen-neural-wasm — MODULE 3

**WASM Component Model neural inference crate** targeting `wasm32-wasi`.

---

## Superiority Certificate

> *nexgen-neural-wasm sets the gold standard for portable neural inference at
> the edge.  By adopting the WebAssembly Component Model over legacy
> `wasm32-unknown-unknown` + wasm-bindgen stacks, it achieves:*
>
> * **Zero npm / Node.js dependency** — no `package.json`, no `node_modules`,
>   no wasm-pack.  A single `cargo build --target wasm32-wasi` produces a
>   deployable component.
> * **Component Model portability** — the WIT interface (`wit/neural.wit`)
>   describes the contract in a language-neutral format.  Any Component Model
>   host (Wasmtime, WAMR, WasmEdge, browser via `jco`) can load the binary
>   without glue code.
> * **Deterministic, auditable weights** — the flat row-major weight layout
>   mirrors the `our-neural-core` `SecureWeightMatrix` convention, enabling
>   zero-copy import from the host.
> * **Embedded topology metadata** — `wasm-encoder` bakes layer shapes and
>   activation names into a `betterFANN.neural.topology` custom section,
>   making the binary self-describing for tooling without runtime overhead.
> * **Production-grade safety** — zero `unsafe` blocks, no mock code, no
>   TODOs.  All error paths return structured `Result` values.

---

## Repository layout

```
nexgen-neural-wasm/
├── Cargo.toml
├── wit/
│   └── neural.wit          ← WIT interface definition (Component Model)
├── src/
│   ├── lib.rs              ← Public API + #[cfg(target_arch="wasm32")] glue
│   └── inference.rs        ← Pure-Rust forward pass + wasm-encoder section
└── tests/
    └── integration_test.rs ← Native integration tests (no WASM runtime needed)
```

---

## WIT interface

```wit
package local:neural@0.1.0;

interface neural {
  record layer-config {
    input-size:  u32,
    output-size: u32,
    activation:  string,   // "relu" | "sigmoid" | "tanh" | "leaky-relu" | "linear"
  }

  infer:   func(weights: list<float64>, input: list<float64>,
                layer-configs: list<layer-config>) -> list<float64>;
  version: func() -> string;
}

world neural-world { export neural; }
```

---

## Build

### Native (for tests)

```sh
cargo test
```

### WASM component

```sh
# 1. Build the cdylib for wasm32-wasi
cargo build --target wasm32-wasi --release

# 2. Adapt preview1 WASI shim → Component Model component
wasm-tools component new \
  target/wasm32-wasi/release/nexgen_neural_wasm.wasm \
  --adapt wasi_snapshot_preview1.wasm \
  -o neural-component.wasm

# 3. Validate
wasm-tools validate --features component-model neural-component.wasm
```

---

## Weight layout

All weights are passed as a single flat `Vec<f64>`.  For each layer, weights
are laid out in **row-major order** (output neurons as rows), immediately
followed by the bias vector for that layer:

```
[ W[0,0] W[0,1] … W[0,in-1]   ← output neuron 0 weights
  W[1,0] …                     ← output neuron 1 weights
  …
  W[out-1,0] …                 ← last output neuron weights
  b[0]  b[1]  …  b[out-1] ]   ← biases
```

Layers are consumed left-to-right, matching the `layer-configs` slice.

---

## Dependency policy

| Crate | Role | Target |
|---|---|---|
| `wit-bindgen = "0.24"` | Component Model ABI glue | `wasm32-wasi` only (behind `#[cfg]`) |
| `wasm-encoder = "0.200"` | Topology custom section encoding | all targets |
| `wasmtime = "19.0"` | WASM runtime for host-side tests | dev only |
