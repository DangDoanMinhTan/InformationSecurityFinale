import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import SegmentationConfig


class SegmentationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float


@dataclass(frozen=True)
class Eyelid:
    apex_y: float
    center_x: float
    curvature: float
    upper: bool
    score: float

    def boundary(self, xs: np.ndarray) -> np.ndarray:
        offset = self.curvature * (xs - self.center_x) ** 2
        return self.apex_y + offset if self.upper else self.apex_y - offset


@dataclass
class Segmentation:
    image: np.ndarray
    scale: float
    pupil: Circle
    iris: Circle
    upper_eyelid: Optional[Eyelid]
    lower_eyelid: Optional[Eyelid]
    noise_mask: np.ndarray
    pupil_strength: float
    limbus_strength: float


def prepare_image(gray: np.ndarray, config: SegmentationConfig) -> Tuple[np.ndarray, float]:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    height, width = gray.shape
    scale = 1.0
    if width > config.max_width:
        scale = config.max_width / width
        gray = cv2.resize(gray, (config.max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    return gray, scale


def remove_specular(gray: np.ndarray, config: SegmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    bright = (gray >= config.specular_threshold).astype(np.uint8)
    spots = np.zeros_like(bright)
    if bright.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        limit = config.max_specular_area_ratio * gray.size
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] <= limit:
                spots[labels == index] = 1
    if not spots.any():
        return gray.copy(), np.zeros(gray.shape, dtype=bool)
    spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.inpaint(gray, spots, 3, cv2.INPAINT_TELEA), spots.astype(bool)


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape
    xs = np.clip(xs, 0, width - 1.001)
    ys = np.clip(ys, 0, height - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = xs - x0
    fy = ys - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def gaussian_kernel(sigma: float) -> np.ndarray:
    half = max(1, int(math.ceil(3 * sigma)))
    axis = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def integro_differential(
    image: np.ndarray,
    centers: Sequence[Tuple[float, float]],
    radii: np.ndarray,
    angles: np.ndarray,
    sigma: float,
) -> Tuple[float, float, float, float]:
    cosines = np.cos(angles)
    sines = np.sin(angles)
    kernel = gaussian_kernel(sigma)
    margin = int(math.ceil(2 * sigma))
    best = (-math.inf, centers[0][0], centers[0][1], float(radii[0]))
    for cx, cy in centers:
        xs = cx + radii[:, None] * cosines[None, :]
        ys = cy + radii[:, None] * sines[None, :]
        profile = bilinear_sample(image, xs, ys).mean(axis=1)
        derivative = np.convolve(np.diff(profile), kernel, mode="same")
        if derivative.size <= 2 * margin:
            continue
        window = derivative[margin: derivative.size - margin]
        index = int(np.argmax(window)) + margin
        value = float(derivative[index])
        if value > best[0]:
            radius = float(radii[index] + (radii[1] - radii[0]) / 2.0)
            best = (value, float(cx), float(cy), radius)
    return best


def ellipse_kernel(size: int) -> np.ndarray:
    size = max(3, size | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def fill_holes(binary: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


ANGLES = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
COS = np.cos(ANGLES)
SIN = np.sin(ANGLES)


def ring_samples(small: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float) -> np.ndarray:
    height, width = small.shape
    px = np.clip(np.rint(xs[..., None] + radius * COS).astype(int), 0, width - 1)
    py = np.clip(np.rint(ys[..., None] + radius * SIN).astype(int), 0, height - 1)
    return small[py, px]


def edge_score(small: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float) -> np.ndarray:
    diff = ring_samples(small, xs, ys, radius + 2.5) - ring_samples(small, xs, ys, radius - 2.5)
    sectors = diff.reshape(diff.shape[:-1] + (8, 8)).mean(axis=-1)
    return np.sort(sectors, axis=-1)[..., 1]


def edge_prepare(image: np.ndarray, shrink: int) -> np.ndarray:
    small = cv2.resize(image, None, fx=1.0 / shrink, fy=1.0 / shrink, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.2)


def search_pupil_edge(image: np.ndarray, config: SegmentationConfig, shrink: int = 2) -> Tuple[Circle, float]:
    small = edge_prepare(image, shrink)
    height, width = small.shape
    side = min(height, width)
    r_min = max(4, int(config.min_pupil_ratio * side))
    r_max = int(config.max_pupil_ratio * side)
    ys = np.arange(int(0.3 * height), int(0.7 * height) + 1, 2)
    xs = np.arange(int(0.3 * width), int(0.7 * width) + 1, 2)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    best = (0.0, 0.0, 0.0)
    best_score = -1e9
    for radius in range(r_min, r_max + 1):
        score = edge_score(small, grid_x, grid_y, float(radius))
        index = np.unravel_index(np.argmax(score), score.shape)
        if score[index] > best_score:
            best_score = float(score[index])
            best = (float(xs[index[1]] * shrink), float(ys[index[0]] * shrink), float(radius * shrink))
    return Circle(*best), best_score


def circle_edge_score(image: np.ndarray, circle: Circle, shrink: int = 2) -> float:
    small = edge_prepare(image, shrink)
    xs = np.array([circle.x / shrink])
    ys = np.array([circle.y / shrink])
    return float(edge_score(small, xs, ys, circle.r / shrink)[0])


def border_mask(smooth: np.ndarray) -> np.ndarray:
    dark = (smooth <= 12).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    height, width = smooth.shape
    border = np.zeros(smooth.shape, dtype=np.uint8)
    for index in range(1, count):
        x = stats[index, cv2.CC_STAT_LEFT]
        y = stats[index, cv2.CC_STAT_TOP]
        w = stats[index, cv2.CC_STAT_WIDTH]
        h = stats[index, cv2.CC_STAT_HEIGHT]
        touches = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches and stats[index, cv2.CC_STAT_AREA] >= 0.01 * height * width:
            border[labels == index] = 1
    if not border.any():
        return border.astype(bool)
    border = cv2.dilate(border, ellipse_kernel(9))
    return border.astype(bool)


def locate_pupil(image: np.ndarray, config: SegmentationConfig) -> Circle:
    height, width = image.shape
    side = min(height, width)
    r_min = config.min_pupil_ratio * side
    r_max = config.max_pupil_ratio * side
    smooth = cv2.GaussianBlur(image, (0, 0), 3.0)
    kernel = ellipse_kernel(7)
    close_kernel = ellipse_kernel(15)
    half_diagonal = math.hypot(height, width) / 2.0
    best: Optional[Circle] = None
    best_score = 0.0
    outside = border_mask(smooth)
    valid = ~outside
    values = smooth[valid] if valid.any() else smooth.ravel()
    for percentile in config.pupil_percentiles:
        threshold = np.percentile(values, percentile)
        binary = ((smooth <= threshold) & valid).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        binary = fill_holes(binary)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for index in range(1, count):
            area = float(stats[index, cv2.CC_STAT_AREA])
            box_w = float(stats[index, cv2.CC_STAT_WIDTH])
            box_h = float(stats[index, cv2.CC_STAT_HEIGHT])
            radius = math.sqrt(area / math.pi)
            if not r_min <= radius <= r_max:
                continue
            aspect = min(box_w, box_h) / max(box_w, box_h)
            fill = area / (box_w * box_h)
            if aspect < 0.7 or fill < 0.65:
                continue
            cx, cy = float(centroids[index][0]), float(centroids[index][1])
            component = (labels == index).astype(np.uint8)
            outer = cv2.dilate(component, ellipse_kernel(int(2 * math.ceil(0.6 * radius)) + 1))
            inner = cv2.dilate(component, ellipse_kernel(5))
            ring = (outer > 0) & (inner == 0)
            if not ring.any():
                continue
            contrast = float(smooth[ring].mean()) - float(smooth[component > 0].mean())
            if contrast <= 0:
                continue
            shape_quality = aspect * max(0.0, 1.0 - abs(fill - math.pi / 4) / (math.pi / 4))
            offset = math.hypot(cx - width / 2.0, cy - height / 2.0) / half_diagonal
            score = contrast * shape_quality * (1.0 - 0.5 * offset)
            if score > best_score:
                best_score = score
                best = Circle(cx, cy, radius)
    edge, edge_value = search_pupil_edge(image, config)
    if best is None:
        if edge_value < config.min_pupil_edge:
            raise SegmentationError("pupil not found")
        return edge
    disagree = math.hypot(best.x - edge.x, best.y - edge.y) > 0.5 * best.r or abs(best.r - edge.r) > 0.4 * best.r
    if disagree and edge_value > config.edge_override_ratio * circle_edge_score(image, best):
        return edge
    return best


def refine_pupil(image: np.ndarray, initial: Circle, config: SegmentationConfig) -> Tuple[Circle, float]:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0).astype(np.float32)
    span = max(3, int(round(0.2 * initial.r)))
    centers = [
        (initial.x + dx, initial.y + dy)
        for dx in range(-span, span + 1)
        for dy in range(-span, span + 1)
    ]
    radii = np.arange(max(3.0, initial.r * 0.7), initial.r * 1.3, 0.5)
    angles = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    value, cx, cy, radius = integro_differential(blurred, centers, radii, angles, config.pupil_sigma)
    if not math.isfinite(value):
        raise SegmentationError("pupil boundary refinement failed")
    return Circle(cx, cy, radius), value


def locate_iris(image: np.ndarray, pupil: Circle, config: SegmentationConfig) -> Tuple[Circle, float]:
    height, width = image.shape
    side = min(height, width)
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5).astype(np.float32)
    r_low = max(pupil.r * 1.35, config.min_iris_ratio * side)
    r_high = config.max_iris_ratio * side
    if r_high <= r_low + 4:
        raise SegmentationError("iris radius range is empty")
    radii = np.arange(r_low, r_high, 1.0)
    sector = math.radians(config.limbus_sector_degrees)
    angles = np.concatenate(
        [np.linspace(-sector, sector, 24), np.linspace(math.pi - sector, math.pi + sector, 24)]
    )
    span = max(3, int(round(config.center_search_ratio * side)))
    coarse = [
        (pupil.x + dx, pupil.y + dy)
        for dx in range(-span, span + 1, 2)
        for dy in range(-span, span + 1, 2)
    ]
    _, cx, cy, _ = integro_differential(blurred, coarse, radii, angles, config.limbus_sigma)
    fine_offsets = np.arange(-1.5, 1.6, 0.5)
    fine = [(cx + dx, cy + dy) for dx in fine_offsets for dy in fine_offsets]
    value, cx, cy, radius = integro_differential(blurred, fine, radii, angles, config.limbus_sigma)
    if not math.isfinite(value):
        raise SegmentationError("iris boundary not found")
    iris = Circle(cx, cy, radius)
    if iris.r < pupil.r * 1.3:
        raise SegmentationError("iris boundary inconsistent with pupil")
    return iris, value


def detect_eyelid(
    image: np.ndarray, pupil: Circle, iris: Circle, upper: bool, config: SegmentationConfig
) -> Optional[Eyelid]:
    height, width = image.shape
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0).astype(np.float32)
    gradient = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    left = max(0, int(math.floor(iris.x - 0.85 * iris.r)))
    right = min(width - 1, int(math.ceil(iris.x + 0.85 * iris.r)))
    if right - left < 10:
        return None
    xs = np.arange(left, right + 1)
    if upper:
        apexes = np.arange(iris.y - 1.05 * iris.r, iris.y - 0.5 * pupil.r, 1.0)
    else:
        apexes = np.arange(iris.y + 0.5 * pupil.r, iris.y + 1.05 * iris.r, 1.0)
    if apexes.size == 0:
        return None
    sign = -1.0 if upper else 1.0
    curvatures = np.linspace(config.eyelid_curvature_range[0], config.eyelid_curvature_range[1], 12)
    shifts = (-0.25 * iris.r, 0.0, 0.25 * iris.r)
    best_score = -math.inf
    best: Optional[Tuple[float, float, float]] = None
    for shift in shifts:
        center_x = iris.x + shift
        squared = (xs - center_x) ** 2
        for curvature in curvatures:
            offset = curvature * squared
            ys = apexes[:, None] + offset[None, :] if upper else apexes[:, None] - offset[None, :]
            rows = np.clip(np.rint(ys).astype(np.int32), 0, height - 1)
            values = gradient[rows, xs[None, :]]
            scores = sign * values.mean(axis=1)
            index = int(np.argmax(scores))
            if scores[index] > best_score:
                best_score = float(scores[index])
                best = (float(apexes[index]), center_x, float(curvature))
    if best is None or best_score < config.eyelid_min_score:
        return None
    return Eyelid(apex_y=best[0], center_x=best[1], curvature=best[2], upper=upper, score=best_score)


def build_noise_mask(
    image: np.ndarray,
    specular: np.ndarray,
    pupil: Circle,
    iris: Circle,
    upper: Optional[Eyelid],
    lower: Optional[Eyelid],
    config: SegmentationConfig,
) -> np.ndarray:
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    mask = specular.copy()
    if upper is not None:
        mask |= yy < upper.boundary(xx) + config.eyelid_margin
    if lower is not None:
        mask |= yy > lower.boundary(xx) - config.eyelid_margin
    iris_distance = np.hypot(xx - iris.x, yy - iris.y)
    pupil_distance = np.hypot(xx - pupil.x, yy - pupil.y)
    annulus = (iris_distance <= iris.r) & (pupil_distance >= pupil.r)
    clean_region = annulus & ~mask
    if clean_region.sum() > 100:
        iris_level = float(np.median(image[clean_region]))
        pupil_level = float(np.percentile(image[pupil_distance < pupil.r * 0.8], 10))
        threshold = pupil_level + config.lash_darkness * (iris_level - pupil_level)
        lashes = (annulus & (image < threshold)).astype(np.uint8)
        lashes = cv2.dilate(lashes, np.ones((3, 3), np.uint8))
        mask |= lashes.astype(bool)
    return mask


def segment(gray: np.ndarray, config: SegmentationConfig) -> Segmentation:
    working, scale = prepare_image(gray, config)
    clean, specular = remove_specular(working, config)
    initial = locate_pupil(clean, config)
    pupil, pupil_strength = refine_pupil(clean, initial, config)
    iris, limbus_strength = locate_iris(clean, pupil, config)
    upper = detect_eyelid(clean, pupil, iris, True, config)
    lower = detect_eyelid(clean, pupil, iris, False, config)
    noise = build_noise_mask(clean, specular, pupil, iris, upper, lower, config)
    return Segmentation(
        image=clean,
        scale=scale,
        pupil=pupil,
        iris=iris,
        upper_eyelid=upper,
        lower_eyelid=lower,
        noise_mask=noise,
        pupil_strength=pupil_strength,
        limbus_strength=limbus_strength,
    )
