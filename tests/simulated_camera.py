import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic_eyes as synthetic
from iris_auth.webcam import guide_rect

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def visible_frame(name: str, session: int, fraction: float = 0.35) -> np.ndarray:
    seed = synthetic.subject_seed(name)
    rng = np.random.default_rng(seed + 500 + session)
    capture = synthetic.random_capture(rng)
    gray = synthetic.render_eye(seed, capture, int(rng.integers(0, 2 ** 31 - 1))).astype(np.float32)
    color = np.stack([gray * 0.45, gray * 0.65, gray * 0.95], axis=2)
    small = cv2.resize(color, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    color = cv2.resize(small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    color += rng.normal(0.0, 2.5, color.shape).astype(np.float32)
    ok, encoded = cv2.imencode(".jpg", np.clip(color, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    x, y, w, h = guide_rect((FRAME_HEIGHT, FRAME_WIDTH), fraction)
    canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (95, 120, 150), dtype=np.uint8)
    canvas[y: y + h, x: x + w] = cv2.resize(color, (w, h), interpolation=cv2.INTER_CUBIC)
    return canvas


class StubCamera:
    def __init__(self, name: str, session_start: int = 1, opened: bool = True):
        self.name = name
        self.session = session_start
        self.opened = opened
        self.released = False

    def is_open(self) -> bool:
        return self.opened

    def read(self):
        return visible_frame(self.name, self.session)

    def release(self) -> None:
        self.released = True

    def next_session(self) -> None:
        self.session += 1
