# ephemeral_lifecycle — MODULE 7

Fixes flaw #4 from `verified_flaw_manifest.json`: *No memory zeroing on dissolution*.

## Overview

| Type | Role |
|---|---|
| `SecureWeightMatrix` | `Vec<f64>` weight store; derives `Zeroize + ZeroizeOnDrop` |
| `CryptoEphemeralNet` | Multi-layer net; explicit `zeroize()` call in `Drop` |

## Key properties

* `SecureWeightMatrix` is zeroed both explicitly (called from `CryptoEphemeralNet::drop`) and
  automatically (its own `ZeroizeOnDrop`-derived `Drop` runs when the `Vec` field is freed).
* `CryptoEphemeralNet::drop` sets an `Arc<AtomicBool>` probe **after** zeroing, giving tests a
  safe, UB-free way to assert that zeroing occurred.
* Bias vector is wrapped in `secrecy::Secret<Vec<f64>>` and never printed or leaked.
* Layer-count comparison uses `subtle::ConstantTimeEq` to resist timing side-channels.

## Running tests

```bash
cargo test
```
