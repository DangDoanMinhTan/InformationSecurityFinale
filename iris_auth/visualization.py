from typing import Optional, Sequence

import cv2
import numpy as np

from .encoding import IrisCode
from .normalization import NormalizedIris
from .pipeline import IrisFeatures
from .segmentation import Segmentation


def draw_segmentation(segmentation: Segmentation) -> np.ndarray:
    canvas = cv2.cvtColor(segmentation.image, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(canvas)
    tint[:, :] = (0, 0, 255)
    masked = segmentation.noise_mask
    canvas[masked] = (0.65 * canvas[masked] + 0.35 * tint[masked]).astype(np.uint8)
    pupil = segmentation.pupil
    iris = segmentation.iris
    cv2.circle(canvas, (int(round(pupil.x)), int(round(pupil.y))), int(round(pupil.r)), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, (int(round(iris.x)), int(round(iris.y))), int(round(iris.r)), (255, 160, 0), 1, cv2.LINE_AA)
    width = canvas.shape[1]
    xs = np.arange(0, width, dtype=np.float32)
    for eyelid in (segmentation.upper_eyelid, segmentation.lower_eyelid):
        if eyelid is None:
            continue
        ys = eyelid.boundary(xs)
        keep = (np.abs(xs - iris.x) <= iris.r * 1.3) & (ys >= 0) & (ys < canvas.shape[0])
        points = np.stack([xs[keep], ys[keep]], axis=1).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_normalized(normalized: NormalizedIris, target_width: int = 560) -> np.ndarray:
    image = np.clip(normalized.image, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    invalid = ~normalized.valid
    color[invalid] = (0.5 * color[invalid] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    scale = target_width / color.shape[1]
    return cv2.resize(color, (target_width, int(color.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)


def render_code(code: IrisCode, target_width: int = 560, row_height: int = 10) -> np.ndarray:
    bands, samples, _ = code.bits.shape
    grid = code.bits.reshape(bands, samples * 2)
    usable = code.mask.reshape(bands, samples * 2)
    pixels = np.where(grid, 255, 0).astype(np.uint8)
    pixels[~usable] = 128
    color = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    return cv2.resize(color, (target_width, bands * row_height), interpolation=cv2.INTER_NEAREST)


def put_lines(canvas: np.ndarray, lines: Sequence[str], origin_x: int, origin_y: int) -> int:
    y = origin_y
    for line in lines:
        cv2.putText(canvas, line, (origin_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    return y


def compose_panel(features: IrisFeatures, title_lines: Optional[Sequence[str]] = None) -> np.ndarray:
    overlay = draw_segmentation(features.segmentation)
    normalized = render_normalized(features.normalized)
    code = render_code(features.code)
    margin = 16
    right_width = normalized.shape[1]
    height_left = overlay.shape[0]
    height_right = 24 + normalized.shape[0] + 24 + code.shape[0] + 24
    header = 20 * len(title_lines or []) + 12
    canvas_height = header + max(height_left, height_right) + margin
    canvas_width = overlay.shape[1] + right_width + 3 * margin
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    if title_lines:
        put_lines(canvas, title_lines, margin, 22)
    top = header
    canvas[top: top + overlay.shape[0], margin: margin + overlay.shape[1]] = overlay
    x = overlay.shape[1] + 2 * margin
    y = top
    put_lines(canvas, ["Normalized iris (red = masked)"], x, y + 14)
    y += 24
    canvas[y: y + normalized.shape[0], x: x + right_width] = normalized
    y += normalized.shape[0]
    put_lines(canvas, ["IrisCode 2048 bits (gray = masked)"], x, y + 18)
    y += 24
    canvas[y: y + code.shape[0], x: x + right_width] = code
    return canvas
