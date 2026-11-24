from dataclasses import dataclass

import cv2
import numpy as np

from .config import NormalizationConfig
from .segmentation import Segmentation


@dataclass
class NormalizedIris:
    image: np.ndarray
    valid: np.ndarray
    inside_fraction: float


def normalize(segmentation: Segmentation, config: NormalizationConfig) -> NormalizedIris:
    height = config.radial_resolution
    width = config.angular_resolution
    theta = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False)
    radial = config.radial_start + (config.radial_end - config.radial_start) * (np.arange(height) + 0.5) / height
    cosines = np.cos(theta)
    sines = np.sin(theta)
    pupil = segmentation.pupil
    iris = segmentation.iris
    pupil_x = pupil.x + pupil.r * cosines
    pupil_y = pupil.y + pupil.r * sines
    iris_x = iris.x + iris.r * cosines
    iris_y = iris.y + iris.r * sines
    map_x = ((1.0 - radial)[:, None] * pupil_x[None, :] + radial[:, None] * iris_x[None, :]).astype(np.float32)
    map_y = ((1.0 - radial)[:, None] * pupil_y[None, :] + radial[:, None] * iris_y[None, :]).astype(np.float32)
    source = segmentation.image.astype(np.float32)
    unwrapped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    noise = cv2.remap(
        segmentation.noise_mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=1.0,
    )
    inside = cv2.remap(
        np.ones_like(source), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    inside_mask = inside > 0.99
    valid = (noise < 0.3) & inside_mask
    return NormalizedIris(image=unwrapped, valid=valid, inside_fraction=float(inside_mask.mean()))
