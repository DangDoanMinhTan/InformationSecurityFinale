import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MatchingConfig
from .encoding import IrisCode
from .matching import MatchResult, shift_range

FIELD_BITS = 7
FIELD_SIZE = 1 << FIELD_BITS
FIELD_POLY = 0x83
CODEWORD_SYMBOLS = 32
BLOCK_BITS = 64
VAULT_BITS = CODEWORD_SYMBOLS * BLOCK_BITS


def build_field() -> Tuple[List[int], List[int]]:
    exp = [0] * (2 * FIELD_SIZE)
    log = [0] * FIELD_SIZE
    value = 1
    for power in range(FIELD_SIZE - 1):
        exp[power] = value
        log[value] = power
        value <<= 1
        if value & FIELD_SIZE:
            value ^= FIELD_POLY
    if value != 1:
        raise RuntimeError("field polynomial is not primitive")
    for power in range(FIELD_SIZE - 1, 2 * FIELD_SIZE):
        exp[power] = exp[power - (FIELD_SIZE - 1)]
    return exp, log


EXP, LOG = build_field()


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    if a == 0:
        return 0
    return EXP[(LOG[a] - LOG[b]) % (FIELD_SIZE - 1)]


def gf_pow(a: int, power: int) -> int:
    if a == 0:
        return 0 if power else 1
    return EXP[(LOG[a] * power) % (FIELD_SIZE - 1)]


def poly_mul(p: Sequence[int], q: Sequence[int]) -> List[int]:
    result = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            result[i + j] ^= gf_mul(a, b)
    return result


def poly_eval(poly: Sequence[int], x: int) -> int:
    value = 0
    for coefficient in poly:
        value = gf_mul(value, x) ^ coefficient
    return value


class ReedSolomon:
    def __init__(self, length: int, dimension: int):
        if not 0 < dimension < length <= FIELD_SIZE - 1:
            raise ValueError("invalid Reed-Solomon parameters")
        self.length = length
        self.dimension = dimension
        self.parity = length - dimension
        generator = [1]
        for power in range(self.parity):
            generator = poly_mul(generator, [1, EXP[power]])
        self.generator = generator

    @property
    def correctable(self) -> int:
        return self.parity // 2

    def encode(self, message: Sequence[int]) -> List[int]:
        if len(message) != self.dimension:
            raise ValueError("message length mismatch")
        remainder = list(message) + [0] * self.parity
        for index in range(self.dimension):
            coefficient = remainder[index]
            if coefficient:
                for offset in range(1, len(self.generator)):
                    remainder[index + offset] ^= gf_mul(self.generator[offset], coefficient)
        return list(message) + remainder[self.dimension:]

    def syndromes(self, received: Sequence[int]) -> List[int]:
        return [poly_eval(received, EXP[power]) for power in range(self.parity)]

    @staticmethod
    def solve(matrix: List[List[int]], rhs: List[int]) -> Optional[List[int]]:
        size = len(rhs)
        rows = [list(matrix[i]) + [rhs[i]] for i in range(size)]
        for column in range(size):
            pivot = next((r for r in range(column, size) if rows[r][column]), None)
            if pivot is None:
                return None
            rows[column], rows[pivot] = rows[pivot], rows[column]
            inverse = gf_div(1, rows[column][column])
            rows[column] = [gf_mul(value, inverse) for value in rows[column]]
            for r in range(size):
                if r != column and rows[r][column]:
                    factor = rows[r][column]
                    rows[r] = [value ^ gf_mul(factor, rows[column][c]) for c, value in enumerate(rows[r])]
        return [rows[i][size] for i in range(size)]

    def decode(self, received: Sequence[int]) -> Optional[List[int]]:
        if len(received) != self.length:
            raise ValueError("word length mismatch")
        word = list(received)
        syndromes = self.syndromes(word)
        if not any(syndromes):
            return word[: self.dimension]
        for errors in range(self.correctable, 0, -1):
            matrix = [[syndromes[i + m] for m in range(errors)] for i in range(errors)]
            rhs = [syndromes[i + errors] for i in range(errors)]
            coefficients = self.solve(matrix, rhs)
            if coefficients is None:
                continue
            locator = [1] + list(reversed(coefficients))
            positions = []
            for index in range(self.length):
                if poly_eval(locator, EXP[(self.length - 1 - index) % (FIELD_SIZE - 1)]) == 0:
                    positions.append(index)
            if len(positions) != errors:
                continue
            vandermonde = [
                [EXP[((self.length - 1 - index) * power) % (FIELD_SIZE - 1)] for index in positions]
                for power in range(errors)
            ]
            magnitudes = self.solve(vandermonde, syndromes[:errors])
            if magnitudes is None:
                continue
            for index, magnitude in zip(positions, magnitudes):
                word[index] ^= magnitude
            if any(self.syndromes(word)):
                return None
            return word[: self.dimension]
        return None


def hadamard_matrix(order: int) -> np.ndarray:
    matrix = np.array([[1.0]])
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


HADAMARD = hadamard_matrix(BLOCK_BITS)
BLOCK_INDEX_BITS = np.array(
    [[(index >> bit) & 1 for bit in range(FIELD_BITS - 1)] for index in range(BLOCK_BITS)], dtype=np.uint8
)


def hadamard_encode(symbol: int) -> np.ndarray:
    constant = symbol & 1
    coefficients = np.array([(symbol >> (bit + 1)) & 1 for bit in range(FIELD_BITS - 1)], dtype=np.uint8)
    word = (BLOCK_INDEX_BITS @ coefficients + constant) & 1
    return word.astype(np.uint8)


def hadamard_decode(bits: np.ndarray, reliable: np.ndarray) -> Tuple[int, float]:
    signal = (1.0 - 2.0 * bits.astype(np.float64)) * reliable.astype(np.float64)
    correlation = HADAMARD @ signal
    best = int(np.argmax(np.abs(correlation)))
    constant = 1 if correlation[best] < 0 else 0
    total = max(float(reliable.sum()), 1.0)
    return (best << 1) | constant, float(abs(correlation[best]) / total)


def interleaving() -> np.ndarray:
    stream = hashlib.shake_256(b"iris-vault-interleave-v1").digest(VAULT_BITS * 4)
    values = np.frombuffer(stream, dtype=np.uint32)
    return np.argsort(values, kind="stable")


INTERLEAVE = interleaving()
DEINTERLEAVE = np.argsort(INTERLEAVE)


def flatten_code(code: IrisCode) -> Tuple[np.ndarray, np.ndarray]:
    bits = code.bits.reshape(-1).astype(np.uint8)
    mask = code.mask.reshape(-1).astype(bool)
    if bits.size != VAULT_BITS:
        raise ValueError(f"vault needs a {VAULT_BITS}-bit code")
    return bits[INTERLEAVE], mask[INTERLEAVE]


def pack(bits: np.ndarray) -> str:
    return base64.b64encode(np.packbits(bits.astype(np.uint8)).tobytes()).decode("ascii")


def unpack(text: str, total: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))[:total]


@dataclass(frozen=True)
class VaultRecord:
    helper: str
    mask: str
    salt: str
    check: str
    key_bits: int
    shape: Tuple[int, int, int]

    def to_dict(self) -> Dict:
        return {
            "helper": self.helper,
            "mask": self.mask,
            "salt": self.salt,
            "check": self.check,
            "key_bits": self.key_bits,
            "shape": list(self.shape),
        }

    @staticmethod
    def from_dict(payload: Dict) -> "VaultRecord":
        return VaultRecord(
            helper=payload["helper"],
            mask=payload["mask"],
            salt=payload["salt"],
            check=payload["check"],
            key_bits=int(payload["key_bits"]),
            shape=tuple(payload["shape"]),
        )


class FuzzyVault:
    def __init__(self, key_symbols: int = 20):
        self.outer = ReedSolomon(CODEWORD_SYMBOLS, key_symbols)
        self.key_symbols = key_symbols

    @property
    def key_bits(self) -> int:
        return self.key_symbols * FIELD_BITS

    def codeword(self, symbols: Sequence[int]) -> np.ndarray:
        blocks = [hadamard_encode(symbol) for symbol in self.outer.encode(symbols)]
        return np.concatenate(blocks)

    @staticmethod
    def symbols_to_key(symbols: Sequence[int]) -> bytes:
        accumulator = 0
        for symbol in symbols:
            accumulator = (accumulator << FIELD_BITS) | symbol
        length = (len(symbols) * FIELD_BITS + 7) // 8
        return accumulator.to_bytes(length, "big")

    @staticmethod
    def digest(key: bytes, salt: bytes) -> str:
        return hashlib.sha256(b"iris-vault-check|" + salt + key).hexdigest()

    def lock(self, code: IrisCode, key_symbols: Optional[Sequence[int]] = None) -> Tuple[VaultRecord, bytes]:
        symbols = list(key_symbols) if key_symbols is not None else [secrets.randbelow(FIELD_SIZE) for _ in range(self.key_symbols)]
        key = self.symbols_to_key(symbols)
        bits, mask = flatten_code(code)
        helper = self.codeword(symbols) ^ bits
        salt = secrets.token_bytes(16)
        record = VaultRecord(
            helper=pack(helper),
            mask=pack(mask),
            salt=base64.b64encode(salt).decode("ascii"),
            check=self.digest(key, salt),
            key_bits=self.key_bits,
            shape=tuple(code.shape),
        )
        return record, key

    def attempt(self, record: VaultRecord, noisy: np.ndarray, reliable: np.ndarray) -> Optional[bytes]:
        signal = ((1.0 - 2.0 * noisy.astype(np.float64)) * reliable.astype(np.float64)).reshape(CODEWORD_SYMBOLS, BLOCK_BITS)
        correlation = signal @ HADAMARD
        best = np.argmax(np.abs(correlation), axis=1)
        negative = correlation[np.arange(CODEWORD_SYMBOLS), best] < 0
        received = ((best << 1) | negative.astype(np.int64)).tolist()
        decoded = self.outer.decode(received)
        if decoded is None:
            return None
        key = self.symbols_to_key(decoded)
        salt = base64.b64decode(record.salt)
        if hmac.compare_digest(self.digest(key, salt), record.check):
            return key
        return None

    def unlock(self, record: VaultRecord, probe: IrisCode, matching: MatchingConfig) -> Optional[bytes]:
        helper = unpack(record.helper, VAULT_BITS)
        enrolled_mask = unpack(record.mask, VAULT_BITS).astype(bool)
        for shift in shift_range(matching.rotation_shifts):
            shifted = IrisCode(np.roll(probe.bits, shift, axis=1), np.roll(probe.mask, shift, axis=1))
            bits, mask = flatten_code(shifted)
            reliable = mask & enrolled_mask
            if int(reliable.sum()) < matching.min_overlap_bits:
                continue
            key = self.attempt(record, helper ^ bits, reliable)
            if key is not None:
                return key
        return None


def keystream(secret: bytes, label: bytes, length: int) -> bytes:
    return hashlib.shake_256(b"iris-cancelable|" + label + b"|" + secret).digest(length)


@dataclass(frozen=True)
class Transform:
    permutation: Optional[np.ndarray]
    flip: Optional[np.ndarray]

    @staticmethod
    def identity() -> "Transform":
        return Transform(None, None)

    @staticmethod
    def derive(secret: bytes, user_id: str, salt: bytes, version: int, size: int = VAULT_BITS) -> "Transform":
        label = f"{user_id}|{version}|".encode("utf-8") + salt
        stream = keystream(secret, label, size * 4 + (size + 7) // 8)
        order = np.argsort(np.frombuffer(stream[: size * 4], dtype=np.uint32), kind="stable")
        flip = np.unpackbits(np.frombuffer(stream[size * 4:], dtype=np.uint8))[:size].astype(bool)
        return Transform(order, flip)

    @property
    def is_identity(self) -> bool:
        return self.permutation is None

    def apply(self, code: IrisCode) -> IrisCode:
        if self.is_identity:
            return code
        shape = code.bits.shape
        bits = code.bits.reshape(-1)[self.permutation] ^ self.flip
        mask = code.mask.reshape(-1)[self.permutation]
        return IrisCode(bits.reshape(shape), mask.reshape(shape))

    def distance(self, probe: IrisCode, stored: IrisCode, matching: MatchingConfig) -> MatchResult:
        best = MatchResult(distance=1.0, shift=0, overlap=0)
        for shift in shift_range(matching.rotation_shifts):
            rolled = IrisCode(np.roll(probe.bits, shift, axis=1), np.roll(probe.mask, shift, axis=1))
            transformed = self.apply(rolled)
            usable = transformed.mask & stored.mask
            overlap = int(usable.sum())
            if overlap < matching.min_overlap_bits:
                continue
            differing = int(((transformed.bits ^ stored.bits) & usable).sum())
            distance = differing / overlap
            if distance < best.distance:
                best = MatchResult(distance=distance, shift=shift, overlap=overlap)
        return best
