import hashlib
import os
import json
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


# ─── Legacy Scramble ──────────────────────────────────────────────────────────

def rotl(val, shift):
    return ((val << shift) & 0xFF) | (val >> (8 - shift))

def rotr(val, shift):
    return (val >> shift) | ((val << (8 - shift)) & 0xFF)

def derive_subkeys(key, rounds=3):
    return [hashlib.sha256((key + str(i)).encode("utf-8")).digest() for i in range(rounds)]

def scramble(data, key, rounds=3):
    subkeys = derive_subkeys(key, rounds)
    result = bytearray(data)
    for r in range(rounds):
        key_bytes = subkeys[r]
        for i in range(len(result)):
            k = key_bytes[i % len(key_bytes)]
            val = result[i] ^ k
            val = rotl(val, k % 8)
            val ^= (k & 0xAA)
            result[i] = val
    return result

def descramble(data, key, rounds=3):
    subkeys = derive_subkeys(key, rounds)
    result = bytearray(data)
    for r in reversed(range(rounds)):
        key_bytes = subkeys[r]
        for i in range(len(result)):
            k = key_bytes[i % len(key_bytes)]
            val = result[i] ^ (k & 0xAA)
            val = rotr(val, k % 8)
            val ^= k
            result[i] = val
    return result


# ─── HMAC Signature ──────────────────────────────────────────────────────────

def sign_data(data: bytes, private_key: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    return hashlib.sha256((digest + private_key).encode("utf-8")).hexdigest()

def sign_file(file_path: str, private_key: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    signature = sign_data(data, private_key)
    out_path = file_path + ".sig"
    with open(out_path, "w") as f:
        f.write(signature)
    return out_path

def verify_signature(data: bytes, signature: str, private_key: str) -> bool:
    digest = hashlib.sha256(data).hexdigest()
    expected = hashlib.sha256((digest + private_key).encode("utf-8")).hexdigest()
    return signature == expected


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
# Payload format: [2 bytes: name_len][name_len bytes: original filename][file data]
# Output format:  [16 bytes: salt][12 bytes: nonce][ciphertext]
# File extension: .bin  (Gmail/phone compatible)

def encrypt_file(file_path: str, password: str) -> str:
    salt = os.urandom(16)
    key = derive_key(password, salt)
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


def decrypt_file(file_path: str, password: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()
    salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
    key = derive_key(password, salt)
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
# Bundle format: [4 bytes: sig_len][sig_len bytes: RSA sig][encrypted blob]

def encrypt_and_sign_file(file_path: str, public_key_pem: str, private_key_pem: str) -> str:
    salt = os.urandom(16)
    key = derive_key(public_key_pem, salt)
    nonce = os.urandom(12)
    chacha = ChaCha20Poly1305(key)
    with open(file_path, "rb") as f:
        data = f.read()
    original_name = os.path.basename(file_path).encode("utf-8")
    payload = len(original_name).to_bytes(2, "big") + original_name + data
    ciphertext = chacha.encrypt(nonce, payload, None)
    cipher_blob = salt + nonce + ciphertext
    signature = rsa_sign_data(cipher_blob, private_key_pem)
    out_path = file_path + ".bin"
    with open(out_path, "wb") as f:
        f.write(len(signature).to_bytes(4, "big") + signature + cipher_blob)
    return out_path


def decrypt_and_verify_file(file_path: str, public_key_pem: str) -> tuple:
    with open(file_path, "rb") as f:
        raw = f.read()
    sig_len = int.from_bytes(raw[:4], "big")
    signature = raw[4:4 + sig_len]
    cipher_blob = raw[4 + sig_len:]
    sig_valid = rsa_verify_data(cipher_blob, signature, public_key_pem)
    salt, nonce, ciphertext = cipher_blob[:16], cipher_blob[16:28], cipher_blob[28:]
    key = derive_key(public_key_pem, salt)
    chacha = ChaCha20Poly1305(key)
    payload = chacha.decrypt(nonce, ciphertext, None)
    name_len = int.from_bytes(payload[:2], "big")
    original_name = payload[2:2 + name_len].decode("utf-8")
    file_data = payload[2 + name_len:]
    out_path = os.path.join(os.path.dirname(file_path), original_name)
    with open(out_path, "wb") as f:
        f.write(file_data)
    return out_path, sig_valid
