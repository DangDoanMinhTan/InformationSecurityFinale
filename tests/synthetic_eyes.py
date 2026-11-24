import math
import sys
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris_auth.io_utils import write_image

WIDTH = 320
HEIGHT = 280


@dataclass(frozen=True)
class Capture:
    iris_radius: float
    pupil_ratio: float
    center: Tuple[float, float]
    pupil_offset: Tuple[float, float]
    rotation: float
    upper_lid: float
    lower_lid: float
    gain: float
    gradient: float
    gradient_angle: float
    blur: float
    noise: float
    jitter: float
    lash_count: int


def subject_seed(name: str, base_seed: int = 2024) -> int:
    return (zlib.crc32(name.encode("utf-8")) ^ (base_seed * 2654435761)) & 0x7FFFFFFF


def wrap_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    half = int(math.ceil(3 * sigma))
    axis = np.arange(-half, half + 1)
    kernel = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.concatenate([values[-half:], values, values[:half]])
    return np.convolve(padded, kernel, mode="valid")


def wrap_blur(array: np.ndarray, sigma_r: float, sigma_t: float) -> np.ndarray:
    pad = int(math.ceil(3 * sigma_t)) + 1
    padded = np.pad(array, ((0, 0), (pad, pad)), mode="wrap")
    blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=sigma_t, sigmaY=sigma_r)
    return blurred[:, pad:-pad]


def standardize(array: np.ndarray) -> np.ndarray:
    return (array - array.mean()) / (array.std() + 1e-6)


def texture_field(rng: np.random.Generator, rows: int, cols: int, sigma_r: float, sigma_t: float) -> np.ndarray:
    noise = rng.standard_normal((rows, cols)).astype(np.float32)
    return standardize(wrap_blur(noise, sigma_r, sigma_t))


@lru_cache(maxsize=64)
def subject_texture(seed: int, rows: int = 160, cols: int = 720) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = (
        1.0 * texture_field(rng, rows, cols, 16.0, 1.8)
        + 0.8 * texture_field(rng, rows, cols, 6.0, 4.0)
        + 0.5 * texture_field(rng, rows, cols, 1.6, 30.0)
        + 0.4 * texture_field(rng, rows, cols, 3.0, 2.0)
    )
    grid_r, grid_t = np.mgrid[0:rows, 0:cols].astype(np.float32)
    for _ in range(int(rng.integers(8, 15))):
        center_r = rng.uniform(0.2, 0.9) * rows
        center_t = rng.uniform(0, cols)
        sigma_r = rng.uniform(3, 8)
        sigma_t = rng.uniform(6, 18)
        depth = rng.uniform(1.5, 3.0)
        delta_t = np.minimum(np.abs(grid_t - center_t), cols - np.abs(grid_t - center_t))
        texture -= depth * np.exp(-((grid_r - center_r) ** 2 / (2 * sigma_r ** 2) + delta_t ** 2 / (2 * sigma_t ** 2)))
    wobble = standardize(wrap_smooth_1d(rng.standard_normal(cols), 25.0))
    collarette = rng.uniform(0.3, 0.42) * rows + wobble * 4.0
    texture += 1.2 * np.exp(-((grid_r - collarette[None, :]) ** 2) / (2 * 4.0 ** 2))
    return standardize(texture).astype(np.float32)


def random_capture(rng: np.random.Generator, constriction: float = 1.0) -> Capture:
    return Capture(
        iris_radius=float(rng.uniform(88, 100)),
        pupil_ratio=float(rng.uniform(0.34, 0.50)) * constriction,
        center=(float(WIDTH / 2 + rng.uniform(-10, 10)), float(HEIGHT / 2 + rng.uniform(-8, 8))),
        pupil_offset=(float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))),
        rotation=float(rng.uniform(-5, 5)),
        upper_lid=float(rng.uniform(0.72, 0.95)),
        lower_lid=float(rng.uniform(0.88, 1.0)),
        gain=float(rng.uniform(0.88, 1.12)),
        gradient=float(rng.uniform(0.0, 0.18)),
        gradient_angle=float(rng.uniform(0, 2 * math.pi)),
        blur=float(rng.uniform(0.9, 1.5)),
        noise=float(rng.uniform(2.5, 5.0)),
        jitter=float(rng.uniform(0.2, 0.4)),
        lash_count=int(rng.integers(18, 40)),
    )


def lid_polyline(cx: float, cy: float, half_width: float, depth: float, upper: bool) -> np.ndarray:
    xs = np.linspace(cx - half_width, cx + half_width, 120)
    curve = depth / half_width ** 2 * (xs - cx) ** 2
    ys = cy - depth + curve if upper else cy + depth - curve
    return np.stack([xs, ys], axis=1)


def render_eye(seed: int, capture: Capture, noise_seed: int) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    texture = subject_texture(seed)
    rows, cols = texture.shape
    jitter = standardize(
        texture_field(rng, rows, cols, 16.0, 1.8) + 0.8 * texture_field(rng, rows, cols, 6.0, 4.0)
    )
    mixed = math.sqrt(1.0 - capture.jitter ** 2) * texture + capture.jitter * jitter
    mixed = np.roll(mixed, int(round(capture.rotation / 360.0 * cols)), axis=1)
    radial_samples, angular_samples = 400, 2400
    grid = cv2.resize(mixed.astype(np.float32), (angular_samples, radial_samples), interpolation=cv2.INTER_LINEAR)
    iris_values = np.clip(108.0 + 24.0 * grid, 45.0, 185.0)
    theta = np.linspace(0, 2 * math.pi, angular_samples, endpoint=False)
    radial = np.linspace(0, 1, radial_samples)
    cx, cy = capture.center
    px = cx + capture.pupil_offset[0]
    py = cy + capture.pupil_offset[1]
    pupil_radius = capture.pupil_ratio * capture.iris_radius
    iris_radius = capture.iris_radius
    pupil_x = px + pupil_radius * np.cos(theta)
    pupil_y = py + pupil_radius * np.sin(theta)
    iris_x = cx + iris_radius * np.cos(theta)
    iris_y = cy + iris_radius * np.sin(theta)
    map_x = (1 - radial)[:, None] * pupil_x[None, :] + radial[:, None] * iris_x[None, :]
    map_y = (1 - radial)[:, None] * pupil_y[None, :] + radial[:, None] * iris_y[None, :]
    ix = np.rint(map_x).astype(np.int64)
    iy = np.rint(map_y).astype(np.int64)
    inside = (ix >= 0) & (ix < WIDTH) & (iy >= 0) & (iy < HEIGHT)
    flat = iy[inside] * WIDTH + ix[inside]
    sums = np.bincount(flat, weights=iris_values[inside], minlength=WIDTH * HEIGHT)
    counts = np.bincount(flat, minlength=WIDTH * HEIGHT)
    iris_layer = (sums / np.maximum(counts, 1)).reshape(HEIGHT, WIDTH).astype(np.float32)
    iris_mask = (counts > 0).reshape(HEIGHT, WIDTH)
    iris_mask = cv2.morphologyEx(iris_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    half_width = 1.6 * iris_radius
    upper_depth = capture.upper_lid * iris_radius
    lower_depth = capture.lower_lid * iris_radius
    upper_boundary = cy - upper_depth + upper_depth / half_width ** 2 * (xx - cx) ** 2
    lower_boundary = cy + lower_depth - lower_depth / half_width ** 2 * (xx - cx) ** 2
    aperture = (yy > upper_boundary) & (yy < lower_boundary) & (np.abs(xx - cx) < half_width)

    skin = 148.0 + 6.0 * cv2.GaussianBlur(rng.standard_normal((HEIGHT, WIDTH)).astype(np.float32), (0, 0), 6.0) * 6.0
    sclera = 172.0 + 4.0 * cv2.GaussianBlur(rng.standard_normal((HEIGHT, WIDTH)).astype(np.float32), (0, 0), 8.0) * 8.0
    canvas = skin.copy()
    canvas[aperture] = sclera[aperture]
    visible_iris = iris_mask & aperture
    canvas[visible_iris] = iris_layer[visible_iris]
    pupil_mask = (np.hypot(xx - px, yy - py) <= pupil_radius) & aperture
    pupil_texture = 28.0 + 3.0 * cv2.GaussianBlur(rng.standard_normal((HEIGHT, WIDTH)).astype(np.float32), (0, 0), 2.0) * 4.0
    canvas[pupil_mask] = pupil_texture[pupil_mask]

    layer = np.clip(canvas, 0, 255).astype(np.uint8)
    for upper, depth in ((True, upper_depth), (False, lower_depth)):
        points = lid_polyline(cx, cy, half_width, depth, upper).astype(np.int32)
        cv2.polylines(layer, [points.reshape(-1, 1, 2)], False, 78, 2, cv2.LINE_AA)
    for _ in range(capture.lash_count):
        x = rng.uniform(cx - 0.8 * iris_radius, cx + 0.8 * iris_radius)
        y = cy - upper_depth + upper_depth / half_width ** 2 * (x - cx) ** 2
        length = rng.uniform(5, 22)
        angle = math.pi / 2 + rng.uniform(-0.5, 0.5)
        end = (int(x + length * math.cos(angle)), int(y + length * math.sin(angle)))
        cv2.line(layer, (int(x), int(y)), end, int(rng.uniform(35, 55)), 1, cv2.LINE_AA)
    for _ in range(capture.lash_count // 3):
        x = rng.uniform(cx - 0.8 * iris_radius, cx + 0.8 * iris_radius)
        y = cy + lower_depth - lower_depth / half_width ** 2 * (x - cx) ** 2
        length = rng.uniform(3, 10)
        angle = -math.pi / 2 + rng.uniform(-0.5, 0.5)
        end = (int(x + length * math.cos(angle)), int(y + length * math.sin(angle)))
        cv2.line(layer, (int(x), int(y)), end, int(rng.uniform(40, 60)), 1, cv2.LINE_AA)

    highlight = (int(px - 0.35 * pupil_radius), int(py - 0.30 * pupil_radius))
    cv2.circle(layer, highlight, 4, 255, -1, cv2.LINE_AA)
    cv2.circle(layer, (highlight[0] + 9, highlight[1] + 3), 2, 255, -1, cv2.LINE_AA)

    image = layer.astype(np.float32)
    direction = (xx - cx) * math.cos(capture.gradient_angle) + (yy - cy) * math.sin(capture.gradient_angle)
    image *= capture.gain * (1.0 + capture.gradient * direction / WIDTH)
    image = cv2.GaussianBlur(image, (0, 0), capture.blur)
    image += rng.normal(0.0, capture.noise, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def render_subject(name: str, count: int, base_seed: int = 2024, constriction: float = 1.0) -> List[np.ndarray]:
    seed = subject_seed(name, base_seed)
    rng = np.random.default_rng(seed + 17)
    images = []
    for index in range(count):
        capture = random_capture(rng, constriction)
        images.append(render_eye(seed, capture, int(rng.integers(0, 2 ** 31 - 1))))
    return images


def render_live_pair(name: str, session: int, base_seed: int = 2024) -> Tuple[np.ndarray, np.ndarray]:
    seed = subject_seed(name, base_seed)
    rng = np.random.default_rng(seed + 1000 + session)
    capture = random_capture(rng)
    constricted = Capture(**{**capture.__dict__, "pupil_ratio": capture.pupil_ratio * 0.72})
    dark = render_eye(seed, capture, int(rng.integers(0, 2 ** 31 - 1)))
    bright = render_eye(seed, constricted, int(rng.integers(0, 2 ** 31 - 1)))
    return dark, bright


def simulate_print_attack(image: np.ndarray, seed: int, period: float = 5.0, blur: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = image.shape
    upsampled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    yy, xx = np.mgrid[0:height * 2, 0:width * 2].astype(np.float32)
    angle = math.radians(45.0)
    u = xx * math.cos(angle) + yy * math.sin(angle)
    v = -xx * math.sin(angle) + yy * math.cos(angle)
    screen = 0.5 + 0.25 * (np.cos(2 * math.pi * u / period) + np.cos(2 * math.pi * v / period))
    source = np.clip((upsampled / 255.0 - 0.1) / 0.85, 0, 1)
    printed = 35.0 + 190.0 * (source > screen).astype(np.float32)
    captured = cv2.resize(printed, (width, height), interpolation=cv2.INTER_AREA)
    captured = cv2.GaussianBlur(captured, (0, 0), blur)
    captured += rng.normal(0.0, 3.0, captured.shape).astype(np.float32)
    return np.clip(captured, 0, 255).astype(np.uint8)


def generate_subjects(out_dir, names: Sequence[str], count: int, base_seed: int = 2024) -> List[Path]:
    root = Path(out_dir)
    written: List[Path] = []
    for name in names:
        for index, image in enumerate(render_subject(name, count, base_seed), start=1):
            path = root / name / f"{name}_{index:02d}.png"
            write_image(path, image)
            written.append(path)
    return written


def generate_dataset(
    out_dir,
    users: Sequence[str],
    outsiders: Sequence[str],
    images_per_user: int = 8,
    images_per_outsider: int = 3,
    base_seed: int = 2024,
) -> Dict[str, List[Path]]:
    root = Path(out_dir)
    written: Dict[str, List[Path]] = {"users": [], "outsiders": [], "probes": [], "attacks": []}
    written["users"] = generate_subjects(root / "users", users, images_per_user, base_seed)
    written["outsiders"] = generate_subjects(root / "outsiders", outsiders, images_per_outsider, base_seed)
    for name in users:
        dark, bright = render_live_pair(name, session=1, base_seed=base_seed)
        for label, frame in (("dark", dark), ("bright", bright)):
            path = root / "probes" / f"{name}_{label}.png"
            write_image(path, frame)
            written["probes"].append(path)
        seed = subject_seed(name, base_seed)
        for label, source_seed in (("dark", 1), ("bright", 2)):
            attack = simulate_print_attack(dark, seed + source_seed)
            path = root / "attacks" / f"{name}_print_{label}.png"
            write_image(path, attack)
            written["attacks"].append(path)
    return written
