import base64
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from .config import EncodingConfig
from .normalization import NormalizedIris


@dataclass(frozen=True)
class IrisCode:
    bits: np.ndarray
    mask: np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.bits.shape)

    @property
    def size(self) -> int:
        return int(self.bits.size)

    def usable_fraction(self) -> float:
        return float(self.mask.mean())

    def to_hex(self) -> str:
        return np.packbits(self.bits.reshape(-1)).tobytes().hex()

    def serialize(self) -> Dict[str, str]:
        return {
            "bits": base64.b64encode(np.packbits(self.bits.reshape(-1)).tobytes()).decode("ascii"),
            "mask": base64.b64encode(np.packbits(self.mask.reshape(-1)).tobytes()).decode("ascii"),
        }

    @staticmethod
    def deserialize(payload: Dict[str, str], shape: Tuple[int, int, int]) -> "IrisCode":
        total = int(np.prod(shape))
        bits = np.unpackbits(np.frombuffer(base64.b64decode(payload["bits"]), dtype=np.uint8))[:total]
        mask = np.unpackbits(np.frombuffer(base64.b64decode(payload["mask"]), dtype=np.uint8))[:total]
        return IrisCode(bits.reshape(shape).astype(bool), mask.reshape(shape).astype(bool))


def gabor_kernels(wavelength: float, config: EncodingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_t = config.bandwidth_ratio * wavelength
    sigma_r = config.radial_sigma
    half_t = int(math.ceil(3 * sigma_t))
    half_r = int(math.ceil(3 * sigma_r))
    t_axis = np.arange(-half_t, half_t + 1, dtype=np.float64)
    r_axis = np.arange(-half_r, half_r + 1, dtype=np.float64)
    grid_t, grid_r = np.meshgrid(t_axis, r_axis)
    envelope = np.exp(-(grid_t ** 2) / (2 * sigma_t ** 2) - (grid_r ** 2) / (2 * sigma_r ** 2))
    omega = 2.0 * math.pi / wavelength
    real = envelope * np.cos(omega * grid_t)
    imag = envelope * np.sin(omega * grid_t)
    real = real - envelope * (real.sum() / envelope.sum())
    envelope = envelope / envelope.sum()
    return real.astype(np.float32), imag.astype(np.float32), envelope.astype(np.float32)


def pad_polar(image: np.ndarray, half_r: int, half_t: int) -> np.ndarray:
    radial_padded = np.pad(image, ((half_r, half_r), (0, 0)), mode="reflect")
    return np.pad(radial_padded, ((0, 0), (half_t, half_t)), mode="wrap")


def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    half_r = kernel.shape[0] // 2
    half_t = kernel.shape[1] // 2
    padded = pad_polar(image, half_r, half_t)
    filtered = cv2.filter2D(padded, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
    return filtered[half_r: half_r + image.shape[0], half_t: half_t + image.shape[1]]


def encode(normalized: NormalizedIris, config: EncodingConfig) -> IrisCode:
    valid = normalized.valid
    image = normalized.image.astype(np.float32)
    height, width = image.shape
    if valid.sum() < 200:
        raise ValueError("not enough valid iris pixels to encode")
    mean = float(image[valid].mean())
    std = float(image[valid].std()) + 1e-6
    standardized = np.where(valid, (image - mean) / std, 0.0).astype(np.float32)
    valid_float = valid.astype(np.float32)
    bands = config.radial_bands
    samples = config.angular_samples
    rows = np.minimum(((np.arange(bands) + 0.5) * height / bands).astype(int), height - 1)
    columns = np.minimum(((np.arange(samples) + 0.5) * width / samples).astype(int), width - 1)
    bits = np.zeros((bands, samples, 2), dtype=bool)
    mask = np.zeros((bands, samples, 2), dtype=bool)
    for scale_index, wavelength in enumerate(config.wavelengths):
        real_kernel, imag_kernel, envelope = gabor_kernels(wavelength, config)
        real_response = apply_filter(standardized, real_kernel)
        imag_response = apply_filter(standardized, imag_kernel)
        coverage = apply_filter(valid_float, envelope)
        magnitude = np.hypot(real_response, imag_response)
        reference = float(np.median(magnitude[valid])) if valid.any() else 0.0
        floor = config.min_magnitude_ratio * reference
        for band in range(scale_index, bands, len(config.wavelengths)):
            row = rows[band]
            real_samples = real_response[row, columns]
            imag_samples = imag_response[row, columns]
            reliable = (coverage[row, columns] >= config.min_valid_fraction) & (magnitude[row, columns] >= floor)
            bits[band, :, 0] = real_samples > 0
            bits[band, :, 1] = imag_samples > 0
            mask[band, :, 0] = reliable
            mask[band, :, 1] = reliable
    return IrisCode(bits=bits, mask=mask)
