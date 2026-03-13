use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey, SecretKey, SignedMessage};

pub struct DilithiumKeyPair {
    pub public_key: dilithium5::PublicKey,
    pub secret_key: dilithium5::SecretKey,
}

impl DilithiumKeyPair {
    pub fn generate() -> Self {
        let (pk, sk) = dilithium5::keypair();
        Self { public_key: pk, secret_key: sk }
    }
}

pub fn sign(message: &[u8], secret_key: &dilithium5::SecretKey) -> dilithium5::SignedMessage {
    dilithium5::sign(message, secret_key)
}

pub fn verify(
    signed_message: &dilithium5::SignedMessage,
    public_key: &dilithium5::PublicKey,
) -> Result<Vec<u8>, String> {
    dilithium5::open(signed_message, public_key).map_err(|e| e.to_string())
}

pub fn signed_message_from_bytes(bytes: &[u8]) -> Result<dilithium5::SignedMessage, String> {
    dilithium5::SignedMessage::from_bytes(bytes).map_err(|e| e.to_string())
}

pub fn signed_message_as_bytes(sm: &dilithium5::SignedMessage) -> &[u8] {
    sm.as_bytes()
}
