# pq_transport — MODULE 8

Post-quantum authenticated key-exchange channel using **Kyber1024** (KEM) and **Dilithium5** (signatures).

## Overview

| Type | Role |
|---|---|
| `PostQuantumChannel` | Holds a Kyber1024 + Dilithium5 keypair; drives encapsulate/decapsulate |
| `EncapsulationBundle` | Wire-format: Kyber ciphertext + Dilithium-signed ciphertext |
| `ChannelError` | Typed errors: bad signature, ciphertext mismatch, invalid bytes |

## Protocol

```
Alice                                        Bob
  |  encapsulate(bob.kem_pk)  →  bundle      |
  |  sign(ciphertext, alice.sign_sk)         |
  |                                          |  decapsulate(bundle, alice.sign_pk)
  |                                          |  verify signature
  |                                          |  kyber_decapsulate → shared_secret
  |  session_key = blake3(shared_secret)     |  session_key = blake3(shared_secret)
```

## Running tests

```bash
cargo test
```
