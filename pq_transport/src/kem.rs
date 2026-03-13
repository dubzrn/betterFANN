use pqcrypto_kyber::kyber1024;
use pqcrypto_traits::kem::{Ciphertext, PublicKey, SecretKey, SharedSecret};

pub struct KyberKeyPair {
    pub public_key: kyber1024::PublicKey,
    pub secret_key: kyber1024::SecretKey,
}

impl KyberKeyPair {
    pub fn generate() -> Self {
        let (pk, sk) = kyber1024::keypair();
        Self { public_key: pk, secret_key: sk }
    }
}

pub fn encapsulate(
    public_key: &kyber1024::PublicKey,
) -> (kyber1024::SharedSecret, kyber1024::Ciphertext) {
    kyber1024::encapsulate(public_key)
}

pub fn decapsulate(
    ciphertext: &kyber1024::Ciphertext,
    secret_key: &kyber1024::SecretKey,
) -> kyber1024::SharedSecret {
    kyber1024::decapsulate(ciphertext, secret_key)
}

pub fn ciphertext_from_bytes(bytes: &[u8]) -> Result<kyber1024::Ciphertext, String> {
    kyber1024::Ciphertext::from_bytes(bytes).map_err(|e| e.to_string())
}

pub fn shared_secret_bytes(ss: &kyber1024::SharedSecret) -> &[u8] {
    ss.as_bytes()
}
