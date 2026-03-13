use pqcrypto_traits::kem::{Ciphertext, PublicKey as KemPublicKey, SharedSecret};
use pqcrypto_traits::sign::{PublicKey as SignPublicKey, SignedMessage};

use crate::kem::{self, KyberKeyPair};
use crate::sign::{self, DilithiumKeyPair};

#[derive(Debug)]
pub enum ChannelError {
    SignatureVerificationFailed(String),
    CiphertextMismatch,
    InvalidBytes(String),
}

impl std::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelError::SignatureVerificationFailed(e) => {
                write!(f, "Signature verification failed: {e}")
            }
            ChannelError::CiphertextMismatch => {
                write!(f, "Ciphertext mismatch after signature verification")
            }
            ChannelError::InvalidBytes(e) => write!(f, "Invalid bytes: {e}"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Raw bytes to send over the wire.
pub struct EncapsulationBundle {
    /// Kyber1024 ciphertext bytes.
    pub ciphertext: Vec<u8>,
    /// Dilithium5 signed(ciphertext) bytes — authenticates the ciphertext.
    pub signed_ciphertext: Vec<u8>,
}

pub struct PostQuantumChannel {
    kem_keypair: KyberKeyPair,
    sign_keypair: DilithiumKeyPair,
}

impl PostQuantumChannel {
    pub fn new() -> Self {
        Self {
            kem_keypair: KyberKeyPair::generate(),
            sign_keypair: DilithiumKeyPair::generate(),
        }
    }

    pub fn kem_public_key(&self) -> &pqcrypto_kyber::kyber1024::PublicKey {
        &self.kem_keypair.public_key
    }

    pub fn sign_public_key(&self) -> &pqcrypto_dilithium::dilithium5::PublicKey {
        &self.sign_keypair.public_key
    }

    /// KEM-encapsulate using `recipient_kem_pk`, sign the ciphertext.
    /// Returns `(bundle, session_key)`. The session_key is kept locally;
    /// only the bundle is transmitted to the recipient.
    pub fn encapsulate(
        &self,
        recipient_kem_pk: &pqcrypto_kyber::kyber1024::PublicKey,
    ) -> (EncapsulationBundle, [u8; 32]) {
        let (shared_secret, ciphertext) = kem::encapsulate(recipient_kem_pk);
        let ct_bytes = ciphertext.as_bytes().to_vec();

        let signed = sign::sign(&ct_bytes, &self.sign_keypair.secret_key);
        let signed_bytes = sign::signed_message_as_bytes(&signed).to_vec();

        let session_key = derive_session_key(kem::shared_secret_bytes(&shared_secret));

        (
            EncapsulationBundle {
                ciphertext: ct_bytes,
                signed_ciphertext: signed_bytes,
            },
            session_key,
        )
    }

    /// Verify the bundle signature, decapsulate, and derive the session key.
    pub fn decapsulate(
        &self,
        bundle: &EncapsulationBundle,
        sender_sign_pk: &pqcrypto_dilithium::dilithium5::PublicKey,
    ) -> Result<[u8; 32], ChannelError> {
        let signed_msg = sign::signed_message_from_bytes(&bundle.signed_ciphertext)
            .map_err(ChannelError::InvalidBytes)?;

        let verified_ct = sign::verify(&signed_msg, sender_sign_pk)
            .map_err(ChannelError::SignatureVerificationFailed)?;

        if verified_ct != bundle.ciphertext {
            return Err(ChannelError::CiphertextMismatch);
        }

        let ct = kem::ciphertext_from_bytes(&bundle.ciphertext)
            .map_err(ChannelError::InvalidBytes)?;

        let shared_secret = kem::decapsulate(&ct, &self.kem_keypair.secret_key);
        Ok(derive_session_key(kem::shared_secret_bytes(&shared_secret)))
    }
}

impl Default for PostQuantumChannel {
    fn default() -> Self {
        Self::new()
    }
}

fn derive_session_key(shared_secret: &[u8]) -> [u8; 32] {
    *blake3::hash(shared_secret).as_bytes()
}
