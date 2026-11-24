import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .config import MatchingConfig
from .encoding import IrisCode


@dataclass(frozen=True)
class MatchResult:
    distance: float
    shift: int
    overlap: int


def shift_range(rotation_shifts: int) -> range:
    half = max(0, rotation_shifts // 2)
    return range(-half, half + 1)


def hamming_distance(probe: IrisCode, reference: IrisCode, config: MatchingConfig) -> MatchResult:
    best = MatchResult(distance=1.0, shift=0, overlap=0)
    for shift in shift_range(config.rotation_shifts):
        shifted_bits = np.roll(reference.bits, shift, axis=1)
        shifted_mask = np.roll(reference.mask, shift, axis=1)
        usable = probe.mask & shifted_mask
        overlap = int(usable.sum())
        if overlap < config.min_overlap_bits:
            continue
        differing = int(((probe.bits ^ shifted_bits) & usable).sum())
        distance = differing / overlap
        if distance < best.distance:
            best = MatchResult(distance=distance, shift=shift, overlap=overlap)
    return best


def hamming_matrix(codes: Sequence[IrisCode], config: MatchingConfig) -> np.ndarray:
    count = len(codes)
    bits = np.stack([code.bits for code in codes]).astype(np.float32)
    masks = np.stack([code.mask for code in codes]).astype(np.float32)
    flat_bits = bits.reshape(count, -1)
    flat_masks = masks.reshape(count, -1)
    masked_bits = flat_bits * flat_masks
    best = np.ones((count, count), dtype=np.float32)
    for shift in shift_range(config.rotation_shifts):
        shifted_masks = np.roll(masks, shift, axis=2).reshape(count, -1)
        shifted_bits = np.roll(bits, shift, axis=2).reshape(count, -1)
        shifted_masked_bits = shifted_bits * shifted_masks
        overlap = flat_masks @ shifted_masks.T
        differing = masked_bits @ shifted_masks.T + flat_masks @ shifted_masked_bits.T - 2.0 * (masked_bits @ shifted_masked_bits.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = np.where(overlap >= config.min_overlap_bits, differing / np.maximum(overlap, 1.0), 1.0)
        best = np.minimum(best, distance.astype(np.float32))
    return best


def false_accept_probability(threshold: float, config: MatchingConfig) -> float:
    n = config.degrees_of_freedom
    limit = int(math.floor(threshold * n + 1e-9))
    if limit < 0:
        return 0.0
    total = sum(math.comb(n, k) for k in range(limit + 1))
    single = total / (2 ** n)
    return min(1.0, single * max(1, config.rotation_shifts))


def threshold_for_database(size: int, config: MatchingConfig, target_far: float) -> float:
    n = config.degrees_of_freedom
    max_index = int(math.floor(config.threshold * n + 1e-9))
    chosen = 0.0
    for index in range(max_index + 1):
        candidate = index / n
        if max(1, size) * false_accept_probability(candidate, config) <= target_far:
            chosen = candidate
        else:
            break
    return chosen
