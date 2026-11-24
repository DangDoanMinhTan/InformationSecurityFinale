import sys
from typing import Optional, Tuple

import cv2
import numpy as np

from .io_utils import stretch_channel

OUTPUT_WIDTH = 320
OUTPUT_HEIGHT = 280
ASPECT = OUTPUT_WIDTH / OUTPUT_HEIGHT
PREVIEW_WIDTH = 640


class Camera:
    def __init__(self, index: int = 0):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            for _ in range(5):
                self.capture.read()

    def is_open(self) -> bool:
        return bool(self.capture.isOpened())

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self) -> None:
        self.capture.release()


def guide_rect(shape: Tuple[int, ...], fraction: float) -> Tuple[int, int, int, int]:
    height, width = shape[:2]
    box_w = int(width * fraction)
    box_h = int(round(box_w / ASPECT))
    if box_h > height * 0.95:
        box_h = int(height * 0.95)
        box_w = int(round(box_h * ASPECT))
    return (width - box_w) // 2, (height - box_h) // 2, box_w, box_h


def crop_guide(frame: np.ndarray, fraction: float) -> np.ndarray:
    x, y, w, h = guide_rect(frame.shape, fraction)
    return frame[y: y + h, x: x + w]


def to_iris_gray(crop: np.ndarray) -> np.ndarray:
    channel = crop.max(axis=2) if crop.ndim == 3 else crop
    method = cv2.INTER_AREA if channel.shape[1] >= OUTPUT_WIDTH else cv2.INTER_CUBIC
    resized = cv2.resize(channel, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=method)
    return cv2.GaussianBlur(stretch_channel(resized), (3, 3), 0)


def process_frame(frame: np.ndarray, fraction: float) -> np.ndarray:
    return to_iris_gray(crop_guide(frame, fraction))


def preview_frame(frame: np.ndarray, fraction: float, mirror: bool = True) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = PREVIEW_WIDTH / width
    small = cv2.resize(frame, (PREVIEW_WIDTH, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    if mirror:
        small = cv2.flip(small, 1)
    x, y, w, h = guide_rect(small.shape, fraction)
    cv2.rectangle(small, (x, y), (x + w, y + h), (80, 220, 80), 2)
    center = (x + w // 2, y + h // 2)
    cv2.circle(small, center, int(0.27 * w), (60, 200, 255), 2, cv2.LINE_AA)
    cv2.line(small, (center[0] - 10, center[1]), (center[0] + 10, center[1]), (60, 200, 255), 1)
    cv2.line(small, (center[0], center[1] - 10), (center[0], center[1] + 10), (60, 200, 255), 1)
    return small
