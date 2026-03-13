use ephemeral_lifecycle::{CryptoEphemeralNet, SecureWeightMatrix};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[test]
fn test_zero_residual_after_drop() {
    let zeroed = Arc::new(AtomicBool::new(false));

    {
        let layer = SecureWeightMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let net = CryptoEphemeralNet::new(vec![layer], vec![0.1, 0.2])
            .with_drop_probe(Arc::clone(&zeroed));

        // Confirm the net works before drop.
        let out = net.forward(&[1.0, 1.0]);
        assert_eq!(out.len(), 2);
        assert!(!zeroed.load(Ordering::SeqCst), "flag must not be set before drop");
    }
    // `net` is dropped here; Drop zeroes weights then sets the flag.
    assert!(
        zeroed.load(Ordering::SeqCst),
        "zeroed_flag must be true after drop, proving explicit zeroize() ran"
    );
}

#[test]
fn test_same_depth_constant_time() {
    let a = CryptoEphemeralNet::new(
        vec![SecureWeightMatrix::new(1, 1, vec![1.0])],
        vec![],
    );
    let b = CryptoEphemeralNet::new(
        vec![SecureWeightMatrix::new(1, 1, vec![2.0])],
        vec![],
    );
    let c = CryptoEphemeralNet::new(vec![], vec![]);

    assert_eq!(bool::from(a.same_depth(&b)), true);
    assert_eq!(bool::from(a.same_depth(&c)), false);
}
