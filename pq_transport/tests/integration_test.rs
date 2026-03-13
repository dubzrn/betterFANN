use pq_transport::PostQuantumChannel;

#[test]
fn test_kem_roundtrip() {
    let alice = PostQuantumChannel::new();
    let bob = PostQuantumChannel::new();

    // Alice encapsulates a shared secret for Bob, signing with her Dilithium key.
    let (bundle, alice_session_key) = alice.encapsulate(bob.kem_public_key());

    // Bob decapsulates and verifies Alice's signature.
    let bob_session_key = bob
        .decapsulate(&bundle, alice.sign_public_key())
        .expect("decapsulation must succeed");

    assert_eq!(
        alice_session_key, bob_session_key,
        "session keys must match after roundtrip"
    );
}

#[test]
fn test_tampered_ciphertext_rejected() {
    let alice = PostQuantumChannel::new();
    let bob = PostQuantumChannel::new();

    let (mut bundle, _) = alice.encapsulate(bob.kem_public_key());
    // Flip a byte in the ciphertext to simulate tampering.
    bundle.ciphertext[0] ^= 0xFF;

    let result = bob.decapsulate(&bundle, alice.sign_public_key());
    assert!(result.is_err(), "tampered ciphertext must be rejected");
}

#[test]
fn test_wrong_sender_key_rejected() {
    let alice = PostQuantumChannel::new();
    let bob = PostQuantumChannel::new();
    let eve = PostQuantumChannel::new();

    let (bundle, _) = alice.encapsulate(bob.kem_public_key());

    // Bob tries to verify with Eve's public key — must fail.
    let result = bob.decapsulate(&bundle, eve.sign_public_key());
    assert!(result.is_err(), "wrong signing key must be rejected");
}
