from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import SystemConfig
from .encoding import IrisCode
from .evaluation import equal_error_rate, error_curves
from .matching import hamming_matrix
from .pipeline import IrisPipeline
from .segmentation import SegmentationError
from .webcam import ASPECT, guide_rect, process_frame

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
GUIDE_FRACTION = 0.35


@dataclass(frozen=True)
class CameraProfile:
    name: str
    title: str
    scale: float
    blur: float
    noise: float
    jpeg_quality: int
    gradient: float
    glare: float
    offset: float
    gamma: float


PROFILES: Dict[str, CameraProfile] = {
    "good": CameraProfile("good", "Good webcam (close, steady, even light)", 1.0, 0.8, 3.0, 85, 0.05, 0.0, 0.01, 1.0),
    "typical": CameraProfile("typical", "Typical webcam (arm's length, room light)", 0.8, 1.3, 5.0, 75, 0.2, 0.02, 0.04, 1.15),
    "poor": CameraProfile("poor", "Poor webcam (far, dim, noisy, window glare)", 0.6, 2.0, 9.0, 55, 0.4, 0.06, 0.07, 1.4),
}

CHANNELS = ("visible", "nir_proxy")


def select_channel(image: np.ndarray, channel: str) -> np.ndarray:
    if channel == "visible":
        return image
    if channel == "nir_proxy":
        red = image[:, :, 2]
        return np.repeat(red[:, :, None], 3, axis=2)
    raise ValueError(f"unknown channel: {channel}")


def lighting_field(height: int, width: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(0, 2 * np.pi))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (xx / width - 0.5) * np.cos(angle) + (yy / height - 0.5) * np.sin(angle)
    return 1.0 + strength * ramp.astype(np.float32)


def glare_spot(height: int, width: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return np.zeros((height, width), dtype=np.float32)
    cx = float(rng.uniform(0.25, 0.75)) * width
    cy = float(rng.uniform(0.2, 0.7)) * height
    radius = float(rng.uniform(0.05, 0.12)) * width
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return (255.0 * strength * 4.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2)))).astype(np.float32)


def camera_frame(image: np.ndarray, profile: CameraProfile, rng: np.random.Generator) -> np.ndarray:
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x, y, w, h = guide_rect((FRAME_HEIGHT, FRAME_WIDTH), GUIDE_FRACTION)
    fitted = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (95, 120, 150), dtype=np.float32)
    shrunk_w = max(8, int(round(w * profile.scale)))
    shrunk_h = max(8, int(round(h * profile.scale)))
    shrunk = cv2.resize(fitted, (shrunk_w, shrunk_h), interpolation=cv2.INTER_AREA)
    shift_x = int(round(float(rng.uniform(-1, 1)) * profile.offset * w))
    shift_y = int(round(float(rng.uniform(-1, 1)) * profile.offset * h))
    left = x + (w - shrunk_w) // 2 + shift_x
    top = y + (h - shrunk_h) // 2 + shift_y
    left = int(np.clip(left, 0, FRAME_WIDTH - shrunk_w))
    top = int(np.clip(top, 0, FRAME_HEIGHT - shrunk_h))
    canvas[top: top + shrunk_h, left: left + shrunk_w] = shrunk
    canvas = cv2.GaussianBlur(canvas, (0, 0), profile.blur)
    canvas *= lighting_field(FRAME_HEIGHT, FRAME_WIDTH, profile.gradient, rng)[:, :, None]
    canvas = 255.0 * np.power(np.clip(canvas, 0, 255) / 255.0, profile.gamma)
    canvas += glare_spot(FRAME_HEIGHT, FRAME_WIDTH, profile.glare, rng)[:, :, None]
    canvas += rng.normal(0.0, profile.noise, canvas.shape).astype(np.float32)
    frame = np.clip(canvas, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), profile.jpeg_quality])
    if not ok:
        raise ValueError("cannot encode frame")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def webcam_capture(image: np.ndarray, profile: CameraProfile, seed: int, channel: str = "visible") -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = camera_frame(select_channel(image, channel), profile, rng)
    return process_frame(frame, GUIDE_FRACTION)


@dataclass
class WebcamResult:
    profile: str
    channel: str
    images: int
    acquired: int
    genuine: int
    impostor: int
    far_at_threshold: float
    frr_at_threshold: float
    threshold: float
    eer: float
    eer_threshold: float
    acquisition_rate: float
    genuine_mean: float
    impostor_mean: float

    def summary(self) -> str:
        return (
            f"{self.profile:8s} {self.channel:9s} acquired {self.acquired}/{self.images} ({self.acquisition_rate * 100:.1f}%)  "
            f"genuine pairs {self.genuine} mean HD {self.genuine_mean:.3f}  impostor pairs {self.impostor} mean HD {self.impostor_mean:.3f}  "
            f"at {self.threshold:.2f}: FRR {self.frr_at_threshold * 100:.1f}% FAR {self.far_at_threshold * 100:.4f}%  EER {self.eer * 100:.2f}% at {self.eer_threshold:.3f}"
        )


def evaluate_webcam(
    images: Sequence[Tuple[str, np.ndarray]],
    config: SystemConfig,
    profile: CameraProfile,
    channel: str,
    seed: int = 11,
) -> WebcamResult:
    pipeline = IrisPipeline(config)
    codes: List[IrisCode] = []
    labels: List[str] = []
    for index, (label, image) in enumerate(images):
        capture = webcam_capture(image, profile, seed * 10007 + index, channel)
        try:
            features = pipeline.extract(capture)
        except (SegmentationError, ValueError):
            continue
        if not features.quality.passed:
            continue
        codes.append(features.code)
        labels.append(label)
    if len(codes) < 2:
        raise ValueError("not enough usable captures")
    distances = hamming_matrix(codes, config.matching)
    array = np.array(labels)
    same = array[:, None] == array[None, :]
    upper = np.triu_indices(len(codes), k=1)
    pair_same = same[upper]
    genuine = distances[upper][pair_same]
    impostor = distances[upper][~pair_same]
    thresholds, far, frr = error_curves(genuine, impostor)
    eer, eer_threshold = equal_error_rate(thresholds, far, frr)
    threshold = config.matching.threshold
    return WebcamResult(
        profile=profile.name,
        channel=channel,
        images=len(images),
        acquired=len(codes),
        genuine=int(genuine.size),
        impostor=int(impostor.size),
        far_at_threshold=float((impostor <= threshold).mean()),
        frr_at_threshold=float((genuine > threshold).mean()),
        threshold=threshold,
        eer=eer,
        eer_threshold=eer_threshold,
        acquisition_rate=len(codes) / len(images),
        genuine_mean=float(genuine.mean()),
        impostor_mean=float(impostor.mean()),
    )
