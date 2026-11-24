from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .config import QualityConfig
from .normalization import NormalizedIris
from .segmentation import Segmentation


@dataclass(frozen=True)
class QualityReport:
    sharpness: float
    usable_fraction: float
    pupil_iris_ratio: float
    inside_fraction: float
    limbus_strength: float
    passed: bool
    failures: Tuple[str, ...]


def measure_sharpness(segmentation: Segmentation) -> float:
    image = cv2.GaussianBlur(segmentation.image, (0, 0), 1.0).astype(np.float32)
    laplacian = np.abs(cv2.Laplacian(image, cv2.CV_32F, ksize=3))
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    iris = segmentation.iris
    pupil = segmentation.pupil
    annulus = (np.hypot(xx - iris.x, yy - iris.y) <= iris.r) & (np.hypot(xx - pupil.x, yy - pupil.y) >= pupil.r * 1.1)
    usable = annulus & ~segmentation.noise_mask
    if usable.sum() < 50:
        return 0.0
    return float(laplacian[usable].mean())


def assess_quality(segmentation: Segmentation, normalized: NormalizedIris, config: QualityConfig) -> QualityReport:
    sharpness = measure_sharpness(segmentation)
    usable_fraction = float(normalized.valid.mean())
    ratio = segmentation.pupil.r / segmentation.iris.r
    failures = []
    if sharpness < config.min_sharpness:
        failures.append("image out of focus")
    if usable_fraction < config.min_usable_fraction:
        failures.append("iris too occluded")
    if not config.min_pupil_iris_ratio <= ratio <= config.max_pupil_iris_ratio:
        failures.append("pupil dilation out of range")
    if normalized.inside_fraction < config.min_inside_fraction:
        failures.append("iris cut off by image border")
    if segmentation.limbus_strength < config.min_limbus_strength:
        failures.append("weak iris boundary")
    return QualityReport(
        sharpness=sharpness,
        usable_fraction=usable_fraction,
        pupil_iris_ratio=ratio,
        inside_fraction=normalized.inside_fraction,
        limbus_strength=segmentation.limbus_strength,
        passed=not failures,
        failures=tuple(failures),
    )
