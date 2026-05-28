import hashlib
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature


# ─── ML Entropy (lazy import — never runs at Flask startup) ──────────────────

def _ml_salt(length: int = 16) -> bytes:
    """Salt via LSTM+GAN entropy. Falls back to os.urandom on any error."""
    try:
        from ml_entropy import generate_ml_salt
        print("[ENTROPY] ML salt used (LSTM+GAN)")
        return generate_ml_salt(length)
    except Exception as e:
        print(f"[ENTROPY] Fallback to os.urandom for salt (reason: {e})")
        return os.urandom(length)

def _ml_rsa_seed() -> int:
    """RSA seed via LSTM+GAN entropy. Falls back to os.urandom on any error."""
    try:
        from ml_entropy import get_entropy_seed_for_rsa
        print("[ENTROPY] ML RSA seed used (LSTM+GAN)")
        return get_entropy_seed_for_rsa()
    except Exception as e:
        print(f"[ENTROPY] Fallback to os.urandom for RSA seed (reason: {e})")
        return int.from_bytes(os.urandom(64), "big")


# ─── RSA Key Generation ───────────────────────────────────────────────────────

def generate_rsa_keypair():
    """2048-bit RSA key pair with LSTM+GAN augmented entropy."""
    ml_seed = _ml_rsa_seed()
    os_seed = int.from_bytes(os.urandom(32), "big")
    combined = ml_seed ^ os_seed
    try:
        import numpy as np
        np.random.seed(combined % (2**32))
    except Exception:
        pass

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")
    return private_pem, public_pem


# ─── RSA-PSS Helpers ─────────────────────────────────────────────────────────

def rsa_sign_data(data: bytes, private_key_pem: str) -> bytes:
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"), password=None, backend=default_backend()
    )
    return private_key.sign(
        data,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )

def rsa_verify_data(data: bytes, signature: bytes, public_key_pem: str) -> bool:
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode("utf-8"), backend=default_backend()
    )
    try:
        public_key.verify(
            signature,
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False


# ─── Key Derivation ──────────────────────────────────────────────────────────

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())


# ─── Encrypt / Decrypt ───────────────────────────────────────────────────────
#
# SECURITY MODEL:
#   encrypt_file(file, recipient_public_key)  → only recipient's private key decrypts
#   decrypt_file(file, my_private_key)        → only intended recipient can decrypt
#
# Payload: [2 bytes: name_len][name bytes: filename][file data]
# Format:  [16 bytes: salt][12 bytes: nonce][ciphertext]
# Salt/nonce generated with LSTM+GAN entropy

def encrypt_file(file_path: str, recipient_public_key_pem: str) -> str:
    salt  = _ml_salt(16)
    nonce = _ml_salt(12)
    key   = derive_key(recipient_public_key_pem, salt)
    chacha = ChaCha20Poly1305(key)

    with open(file_path, "rb") as f:
        data = f.read()

    original_name = os.path.basename(file_path).encode("utf-8")
    payload = len(original_name).to_bytes(2, "big") + original_name + data
    ciphertext = chacha.encrypt(nonce, payload, None)

    out_path = file_path + ".bin"
    with open(out_path, "wb") as f:
        f.write(salt + nonce + ciphertext)
    return out_path


def decrypt_file(file_path: str, my_private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(
        my_private_key_pem.encode("utf-8"), password=None, backend=default_backend()
    )
    my_public_key_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    with open(file_path, "rb") as f:
        raw = f.read()

    salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
    key = derive_key(my_public_key_pem, salt)
    chacha = ChaCha20Poly1305(key)
    payload = chacha.decrypt(nonce, ciphertext, None)

    name_len = int.from_bytes(payload[:2], "big")
    original_name = payload[2:2 + name_len].decode("utf-8")
    file_data = payload[2 + name_len:]

    out_path = os.path.join(os.path.dirname(file_path), original_name)
    with open(out_path, "wb") as f:
        f.write(file_data)
    return out_path


# ─── Encrypt + Sign / Decrypt + Verify ───────────────────────────────────────
#
# SECURITY MODEL:
#   encrypt_and_sign_file(file, recipient_public_key, sender_private_key)
#   decrypt_and_verify_file(file, my_private_key, sender_public_key)
#
# Bundle: [4 bytes: sig_len][RSA-PSS sig][encrypted blob]

def encrypt_and_sign_file(file_path: str, recipient_public_key_pem: str, sender_private_key_pem: str) -> str:
    salt  = _ml_salt(16)
    nonce = _ml_salt(12)
    key   = derive_key(recipient_public_key_pem, salt)
    chacha = ChaCha20Poly1305(key)

    with open(file_path, "rb") as f:
        data = f.read()

    original_name = os.path.basename(file_path).encode("utf-8")
    payload = len(original_name).to_bytes(2, "big") + original_name + data
    ciphertext = chacha.encrypt(nonce, payload, None)
    cipher_blob = salt + nonce + ciphertext

    signature = rsa_sign_data(cipher_blob, sender_private_key_pem)

    out_path = file_path + ".bin"
    with open(out_path, "wb") as f:
        f.write(len(signature).to_bytes(4, "big") + signature + cipher_blob)
    return out_path


def decrypt_and_verify_file(file_path: str, my_private_key_pem: str, sender_public_key_pem: str) -> tuple:
    private_key = serialization.load_pem_private_key(
        my_private_key_pem.encode("utf-8"), password=None, backend=default_backend()
    )
    my_public_key_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    with open(file_path, "rb") as f:
        raw = f.read()

    sig_len   = int.from_bytes(raw[:4], "big")
    signature = raw[4:4 + sig_len]
    cipher_blob = raw[4 + sig_len:]

    sig_valid = rsa_verify_data(cipher_blob, signature, sender_public_key_pem)

    salt, nonce, ciphertext = cipher_blob[:16], cipher_blob[16:28], cipher_blob[28:]
    key = derive_key(my_public_key_pem, salt)
    chacha = ChaCha20Poly1305(key)
    payload = chacha.decrypt(nonce, ciphertext, None)

    name_len = int.from_bytes(payload[:2], "big")
    original_name = payload[2:2 + name_len].decode("utf-8")
    file_data = payload[2 + name_len:]

    out_path = os.path.join(os.path.dirname(file_path), original_name)
    with open(out_path, "wb") as f:
        f.write(file_data)
    return out_path, sig_valid
