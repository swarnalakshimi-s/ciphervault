import hashlib
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

# ─── RSA Key Generation ───────────────────────────────────────────────────────

def generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
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
    private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None, backend=default_backend())
    return private_key.sign(data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())

def rsa_verify_data(data: bytes, signature: bytes, public_key_pem: str) -> bool:
    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"), backend=default_backend())
    try:
        public_key.verify(signature, data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        return True
    except InvalidSignature:
        return False


# ─── Key Derivation ──────────────────────────────────────────────────────────

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
    return kdf.derive(password.encode())


# ─── Encrypt / Decrypt ───────────────────────────────────────────────────────
#
# SECURITY MODEL:
#   encrypt_file(file, recipient_public_key)  → only recipient's private key can decrypt
#   decrypt_file(file, my_private_key)        → only the intended recipient can decrypt
#
# Payload format: [2 bytes: name_len][name_len bytes: filename][file data]
# File format:    [16 bytes: salt][12 bytes: nonce][ciphertext]

def encrypt_file(file_path: str, recipient_public_key_pem: str) -> str:
    """Encrypt with RECIPIENT's public key — only they can decrypt with their private key."""
    salt = os.urandom(16)
    key = derive_key(recipient_public_key_pem, salt)
    nonce = os.urandom(12)
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
    """Decrypt using MY OWN private key — only works if file was encrypted for me."""
    # Derive the symmetric key using my public key (derived from my private key)
    private_key = serialization.load_pem_private_key(my_private_key_pem.encode("utf-8"), password=None, backend=default_backend())
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
#     → encrypted for recipient only, signed by sender
#   decrypt_and_verify_file(file, my_private_key, sender_public_key)
#     → only recipient can decrypt, verify sender's signature
#
# Bundle format: [4 bytes: sig_len][sig_len bytes: RSA-PSS sig][encrypted blob]

def encrypt_and_sign_file(file_path: str, recipient_public_key_pem: str, sender_private_key_pem: str) -> str:
    """Encrypt with RECIPIENT's public key, sign with SENDER's private key."""
    salt = os.urandom(16)
    key = derive_key(recipient_public_key_pem, salt)
    nonce = os.urandom(12)
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
    """Decrypt with MY private key, verify signature with SENDER's public key."""
    private_key = serialization.load_pem_private_key(my_private_key_pem.encode("utf-8"), password=None, backend=default_backend())
    my_public_key_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    with open(file_path, "rb") as f:
        raw = f.read()
    sig_len = int.from_bytes(raw[:4], "big")
    signature = raw[4:4 + sig_len]
    cipher_blob = raw[4 + sig_len:]

    # Verify sender's signature
    sig_valid = rsa_verify_data(cipher_blob, signature, sender_public_key_pem)

    # Decrypt using my own public key (derived from my private key)
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
