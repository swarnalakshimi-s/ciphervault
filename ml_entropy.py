"""
ml_entropy.py  —  LSTM + GAN Entropy Source
Called only during encryption operations, never at Flask startup.
"""

import os
import hashlib
import struct
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# LSTM GENERATOR
# ═══════════════════════════════════════════════════════════════════

class LSTMCell:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / (input_size + hidden_size))
        n = input_size + hidden_size
        self.Wf = rng.randn(n, hidden_size) * s
        self.Wi = rng.randn(n, hidden_size) * s
        self.Wg = rng.randn(n, hidden_size) * s
        self.Wo = rng.randn(n, hidden_size) * s
        self.bf = np.ones(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def forward(self, x, h, c):
        combined = np.concatenate([h, x])
        f = self.sigmoid(combined @ self.Wf + self.bf)
        i = self.sigmoid(combined @ self.Wi + self.bi)
        g = np.tanh(combined @ self.Wg + self.bg)
        o = self.sigmoid(combined @ self.Wo + self.bo)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new


class LSTMGenerator:
    def __init__(self, input_size=8, hidden_size=64):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm1 = LSTMCell(input_size, hidden_size, seed=1)
        self.lstm2 = LSTMCell(hidden_size, hidden_size, seed=2)
        rng = np.random.RandomState(3)
        self.W_out = rng.randn(hidden_size, 256) * 0.1
        self.b_out = np.zeros(256)

    def _to_bits(self, b):
        return np.array([(b >> i) & 1 for i in range(self.input_size)],
                        dtype=np.float32) * 2.0 - 1.0

    def _softmax(self, x, temp=1.0):
        x = x / temp - np.max(x / temp)
        e = np.exp(x)
        return e / e.sum()

    def train(self, data: bytes, epochs=3, lr=0.005):
        data = list(data)
        for epoch in range(epochs):
            h1 = np.zeros(self.hidden_size)
            c1 = np.zeros(self.hidden_size)
            h2 = np.zeros(self.hidden_size)
            c2 = np.zeros(self.hidden_size)
            for i in range(len(data) - 1):
                x = self._to_bits(data[i])
                h1, c1 = self.lstm1.forward(x, h1, c1)
                h2, c2 = self.lstm2.forward(h1, h2, c2)
                logits = h2 @ self.W_out + self.b_out
                probs = self._softmax(logits)
                dL = probs.copy()
                dL[data[i + 1]] -= 1.0
                dL = np.clip(dL, -1.0, 1.0)
                self.W_out -= lr * np.outer(h2, dL)
                self.b_out -= lr * dL
            lr *= 0.85

    def generate(self, seed: bytes, length: int) -> bytes:
        h1 = np.zeros(self.hidden_size)
        c1 = np.zeros(self.hidden_size)
        h2 = np.zeros(self.hidden_size)
        c2 = np.zeros(self.hidden_size)
        for b in seed:
            x = self._to_bits(b)
            h1, c1 = self.lstm1.forward(x, h1, c1)
            h2, c2 = self.lstm2.forward(h1, h2, c2)
        result = []
        current = seed[-1] if seed else 128
        for _ in range(length):
            x = self._to_bits(current)
            h1, c1 = self.lstm1.forward(x, h1, c1)
            h2, c2 = self.lstm2.forward(h1, h2, c2)
            probs = self._softmax(h2 @ self.W_out + self.b_out, temp=1.1)
            current = int(np.random.choice(256, p=probs))
            result.append(current)
        return bytes(result)


# ═══════════════════════════════════════════════════════════════════
# GAN DISCRIMINATOR
# ═══════════════════════════════════════════════════════════════════

class GANDiscriminator:
    def __init__(self, block_size=32):
        self.block_size = block_size
        inp = block_size * 8
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(inp, 128) * np.sqrt(2.0 / inp)
        self.b1 = np.zeros(128)
        self.W2 = rng.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(64)
        self.W3 = rng.randn(64, 1) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(1)

    def _to_bits(self, data: bytes) -> np.ndarray:
        bits = []
        for b in data[:self.block_size]:
            bits.extend([(b >> i) & 1 for i in range(8)])
        # Pad if data shorter than block_size
        while len(bits) < self.block_size * 8:
            bits.append(0)
        return np.array(bits[:self.block_size * 8], dtype=np.float32) * 2.0 - 1.0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def forward(self, x):
        a1 = np.maximum(0, x @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        return float(self._sigmoid(a2 @ self.W3 + self.b3)[0])

    def train_step(self, real: bytes, fake: bytes, lr=0.001):
        x_real = self._to_bits(real)
        x_fake = self._to_bits(fake)
        for x, label in [(x_real, 1.0), (x_fake, 0.0)]:
            a1 = np.maximum(0, x @ self.W1 + self.b1)
            a2 = np.maximum(0, a1 @ self.W2 + self.b2)
            pred = float(self._sigmoid(a2 @ self.W3 + self.b3)[0])
            err = pred - label
            d3 = np.clip(np.array([err * pred * (1 - pred)]), -1, 1)
            d2 = np.clip((self.W3 @ d3.T).flatten() * (a2 > 0), -1, 1)
            d1 = np.clip((self.W2 @ d2.T).flatten() * (a1 > 0), -1, 1)
            self.W3 -= lr * np.outer(a2, d3)
            self.W2 -= lr * np.outer(a1, d2)
            self.W1 -= lr * np.outer(x, d1)

    def score(self, data: bytes) -> float:
        return self.forward(self._to_bits(data))


# ═══════════════════════════════════════════════════════════════════
# COMBINED SYSTEM — created fresh per use, no global state at import
# ═══════════════════════════════════════════════════════════════════

def _build_and_train_system():
    """Build and train a fresh LSTM+GAN system. Called lazily."""
    gen = LSTMGenerator(input_size=8, hidden_size=64)
    dis = GANDiscriminator(block_size=32)

    # Train LSTM on random bytes
    gen.train(os.urandom(512), epochs=3, lr=0.005)

    # Adversarial training
    for _ in range(8):
        real = os.urandom(32)
        fake = gen.generate(os.urandom(16), 32)
        dis.train_step(real, fake, lr=0.001)

    return gen, dis


def generate_ml_entropy(num_bytes: int = 64) -> bytes:
    """
    Generate ML-augmented entropy.
    Trains a fresh LSTM+GAN system, generates bytes, XORs with os.urandom.
    """
    try:
        gen_size = max(num_bytes, 32)
        gen, dis = _build_and_train_system()

        best = None
        best_score = -1
        for _ in range(3):
            seed = os.urandom(16)
            candidate = gen.generate(seed, gen_size)
            score = dis.score(candidate[:32])
            if score > best_score:
                best_score = score
                best = candidate

        ml_bytes = (best or os.urandom(gen_size))[:num_bytes]

        # XOR with fresh OS entropy — guaranteed at least as strong as os.urandom
        os_bytes = os.urandom(num_bytes)
        combined = bytes(a ^ b for a, b in zip(ml_bytes, os_bytes))

        # SHA-512 for uniform distribution
        result = b""
        counter = 0
        while len(result) < num_bytes:
            result += hashlib.sha512(combined + struct.pack(">I", counter)).digest()
            counter += 1
        return result[:num_bytes]

    except Exception:
        # Any failure → safe fallback
        return os.urandom(num_bytes)


def generate_ml_salt(length: int = 16) -> bytes:
    return generate_ml_entropy(length)


def get_entropy_seed_for_rsa() -> int:
    return int.from_bytes(generate_ml_entropy(64), "big")


def test_entropy_quality(num_bytes: int = 512) -> dict:
    """Statistical quality test — use in project presentation."""
    data = generate_ml_entropy(num_bytes)
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts / num_bytes
    probs_nz = probs[probs > 0]
    shannon = float(-np.sum(probs_nz * np.log2(probs_nz)))
    chi_sq = float(np.sum((counts - num_bytes / 256) ** 2 / (num_bytes / 256)))
    corr = float(np.corrcoef(arr[:-1].astype(float), arr[1:].astype(float))[0, 1])
    bits = np.unpackbits(arr)
    return {
        "shannon_entropy_bits": round(shannon, 4),
        "chi_square": round(chi_sq, 2),
        "serial_correlation": round(corr, 4),
        "bit_balance_percent": round(float(np.mean(bits)) * 100, 2),
        "assessment": "PASS" if shannon > 7.0 and abs(corr) < 0.1 else "FAIL",
        "bytes_tested": num_bytes
    }


def compare_entropy_sources(num_bytes: int = 512) -> dict:
    """Compare ML vs OS entropy — for project presentation."""
    results = {}
    for name, data in [("os_urandom", os.urandom(num_bytes)),
                        ("lstm_gan", generate_ml_entropy(num_bytes))]:
        arr = np.frombuffer(data, dtype=np.uint8)
        counts = np.bincount(arr, minlength=256)
        probs = counts / num_bytes
        probs_nz = probs[probs > 0]
        results[name] = {
            "shannon_entropy": round(float(-np.sum(probs_nz * np.log2(probs_nz))), 4),
            "chi_square": round(float(np.sum((counts - num_bytes/256)**2 / (num_bytes/256))), 2),
            "serial_correlation": round(abs(float(np.corrcoef(
                arr[:-1].astype(float), arr[1:].astype(float))[0, 1])), 4),
        }
    return results
