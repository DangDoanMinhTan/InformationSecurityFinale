import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import PadConfig
from .segmentation import Segmentation


@dataclass(frozen=True)
class PadResult:
    passed: bool
    reasons: Tuple[str, ...]
    spectral_peak: float
    pupil_constriction: Optional[float]
    detail_ratio: Optional[float] = None


def jpeg_exclusion(size: int) -> np.ndarray:
    frequencies = np.fft.fftshift(np.fft.fftfreq(size))
    horizontal = frequencies[None, :]
    vertical = frequencies[:, None]
    excluded = np.zeros((size, size), dtype=bool)
    for harmonic in range(1, 5):
        line = harmonic / 8.0
        excluded |= np.abs(np.abs(horizontal) - line) < 0.006
        excluded |= np.abs(np.abs(vertical) - line) < 0.006
    return excluded


def spectral_peak_score(segmentation: Segmentation, config: PadConfig) -> float:
    image = segmentation.image.astype(np.float32)
    iris = segmentation.iris
    half = int(max(24, min(0.7 * iris.r, 80)))
    cx = int(round(iris.x))
    cy = int(round(iris.y))
    height, width = image.shape
    half = min(half, cx, cy, width - cx - 1, height - cy - 1)
    if half < 16:
        return 0.0
    size = 2 * half
    patch = image[cy - half: cy + half, cx - half: cx + half]
    patch = patch - patch.mean()
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(patch * window))
    log_power = np.log10(np.abs(spectrum) ** 2 + 1e-3)
    freq_y, freq_x = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(size)), np.fft.fftshift(np.fft.fftfreq(size)), indexing="ij"
    )
    radius = np.hypot(freq_x, freq_y)
    bins = np.minimum((radius * size).astype(int), size)
    residual = np.zeros_like(log_power)
    for index in np.unique(bins):
        selected = bins == index
        residual[selected] = log_power[selected] - np.median(log_power[selected])
    low, high = config.spectral_band
    band = (radius >= low) & (radius <= high) & ~jpeg_exclusion(size)
    if not band.any():
        return 0.0
    return float(residual[band].max())


def detail_ratio_score(segmentation: Segmentation, config: PadConfig) -> Optional[float]:
    image = segmentation.image.astype(np.float32)
    iris = segmentation.iris
    half = int(max(24, min(0.7 * iris.r, 80)))
    cx = int(round(iris.x))
    cy = int(round(iris.y))
    height, width = image.shape
    half = min(half, cx, cy, width - cx - 1, height - cy - 1)
    if half < 16:
        return None
    size = 2 * half
    patch = image[cy - half: cy + half, cx - half: cx + half]
    patch = patch - patch.mean()
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(patch * window))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(size))
    radius = np.hypot(frequencies[None, :], frequencies[:, None])
    fine = (radius >= config.detail_fine_band[0]) & (radius <= config.detail_fine_band[1])
    coarse = (radius >= config.detail_coarse_band[0]) & (radius <= config.detail_coarse_band[1])
    if not fine.any() or not coarse.any():
        return None
    return float(np.log10(power[fine].mean() + 1e-6) - np.log10(power[coarse].mean() + 1e-6))


class PresentationAttackDetector:
    def __init__(self, config: PadConfig):
        self.config = config

    def assess(self, primary: Segmentation, stimulated: Optional[Segmentation] = None) -> PadResult:
        reasons = []
        peak = spectral_peak_score(primary, self.config)
        if peak > self.config.max_spectral_peak:
            reasons.append("periodic print or screen pattern detected")
        detail: Optional[float] = None
        if self.config.min_detail_ratio is not None:
            detail = detail_ratio_score(primary, self.config)
            if detail is not None and detail < self.config.min_detail_ratio:
                reasons.append("image detail too low for a live iris (re-captured print or screen)")
        constriction: Optional[float] = None
        if stimulated is not None:
            before = primary.pupil.r / primary.iris.r
            after = stimulated.pupil.r / stimulated.iris.r
            constriction = 1.0 - after / before
            if constriction < self.config.min_pupil_constriction:
                reasons.append("no pupil reaction to light")
        elif self.config.require_pupil_reaction:
            reasons.append("pupil reaction frame missing")
        return PadResult(
            passed=not reasons,
            reasons=tuple(reasons),
            spectral_peak=peak,
            pupil_constriction=constriction,
            detail_ratio=detail,
        )
