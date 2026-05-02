import hashlib
import os
import json
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

# ─── RSA Key Generation (called at registration) ─────────────────────────────

def generate_rsa_keypair():
    """Generate a 2048-bit RSA key pair.
    Returns (private_key_pem: str, public_key_pem: str)
    """
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


# ─── Core Custom Scramble (legacy) ───────────────────────────────────────────

def rotl(val, shift):
    return ((val << shift) & 0xFF) | (val >> (8 - shift))

def rotr(val, shift):
    return (val >> shift) | ((val << (8 - shift)) & 0xFF)

def derive_subkeys(key, rounds=3):
    subkeys = []
    for i in range(rounds):
        digest = hashlib.sha256((key + str(i)).encode("utf-8")).digest()
        subkeys.append(digest)
    return subkeys

def scramble(data, key, rounds=3):
    subkeys = derive_subkeys(key, rounds)
    result = bytearray(data)
    for r in range(rounds):
        key_bytes = subkeys[r]
        key_len = len(key_bytes)
        for i in range(len(result)):
            k = key_bytes[i % key_len]
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
        key_len = len(key_bytes)
        for i in range(len(result)):
            k = key_bytes[i % key_len]
            val = result[i] ^ (k & 0xAA)
            val = rotr(val, k % 8)
            val ^= k
            result[i] = val
    return result


# ─── HMAC Signature (used for Sign / Verify tabs) ────────────────────────────

def sign_data(data: bytes, private_key: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    signature = hashlib.sha256((digest + private_key).encode("utf-8")).hexdigest()
    return signature

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


# ─── RSA-PSS Signature helpers ───────────────────────────────────────────────

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


# ─── ChaCha20-Poly1305 Encrypt / Decrypt ─────────────────────────────────────
#     Plain Encrypt/Decrypt tabs: password = user's public key PEM string

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt_file(file_path: str, password: str) -> str:
    salt = os.urandom(16)
    key = derive_key(password, salt)
    nonce = os.urandom(12)
    chacha = ChaCha20Poly1305(key)
    with open(file_path, "rb") as f:
        data = f.read()
    ciphertext = chacha.encrypt(nonce, data, None)
    out_path = file_path + ".enc"
    with open(out_path, "wb") as f:
        f.write(salt + nonce + ciphertext)
    return out_path

def decrypt_file(file_path: str, password: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()
    salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
    key = derive_key(password, salt)
    chacha = ChaCha20Poly1305(key)
    plaintext = chacha.decrypt(nonce, ciphertext, None)
    out_path = file_path + ".dec"
    with open(out_path, "wb") as f:
        f.write(plaintext)
    return out_path


# ─── Encrypt + Sign (combined) ───────────────────────────────────────────────

def encrypt_and_sign_file(file_path: str, public_key_pem: str, private_key_pem: str) -> str:
    """
    1. Encrypt file with ChaCha20-Poly1305 using public_key_pem as the password.
    2. RSA-PSS sign the ciphertext blob with the private key.
    3. Bundle: [4-byte sig_len][RSA signature][cipher_blob] → .encsig
    """
    salt = os.urandom(16)
    key = derive_key(public_key_pem, salt)
    nonce = os.urandom(12)
    chacha = ChaCha20Poly1305(key)
    with open(file_path, "rb") as f:
        data = f.read()
    ciphertext = chacha.encrypt(nonce, data, None)
    cipher_blob = salt + nonce + ciphertext

    signature = rsa_sign_data(cipher_blob, private_key_pem)
    sig_len = len(signature).to_bytes(4, "big")

    out_path = file_path + ".encsig"
    with open(out_path, "wb") as f:
        f.write(sig_len + signature + cipher_blob)
    return out_path


def decrypt_and_verify_file(file_path: str, public_key_pem: str) -> tuple:
    """
    1. Parse [sig_len][RSA signature][cipher_blob].
    2. Verify RSA-PSS signature using the public key.
    3. Decrypt cipher_blob using public key as password.
    Returns (out_path: str, sig_valid: bool)
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    sig_len = int.from_bytes(raw[:4], "big")
    signature = raw[4:4 + sig_len]
    cipher_blob = raw[4 + sig_len:]

    sig_valid = rsa_verify_data(cipher_blob, signature, public_key_pem)

    salt, nonce, ciphertext = cipher_blob[:16], cipher_blob[16:28], cipher_blob[28:]
    key = derive_key(public_key_pem, salt)
    chacha = ChaCha20Poly1305(key)
    plaintext = chacha.decrypt(nonce, ciphertext, None)

    out_path = file_path + ".dec"
    with open(out_path, "wb") as f:
        f.write(plaintext)

    return out_path, sig_valid
