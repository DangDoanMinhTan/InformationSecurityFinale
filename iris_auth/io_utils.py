from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def read_gray(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"cannot read image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unsupported or corrupt image: {path}")
    return image


def write_image(path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extension = target.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(str(target))


def list_images(root) -> List[Path]:
    base = Path(root)
    if base.is_file():
        return [base]
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def class_label(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return "_".join(relative.parts[:-1])
    stem = path.stem
    end = len(stem)
    while end > 0 and stem[end - 1].isdigit():
        end -= 1
    label = stem[:end].rstrip("_-. ")
    return label if label else stem


def load_dataset(roots: Iterable) -> List[Tuple[str, Path]]:
    records: List[Tuple[str, Path]] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            raise FileNotFoundError(f"dataset path not found: {base}")
        for path in list_images(base):
            records.append((class_label(base if base.is_dir() else base.parent, path), path))
    return records


def stretch_channel(channel: np.ndarray) -> np.ndarray:
    low, high = np.percentile(channel, (1.0, 99.7))
    scaled = (channel.astype(np.float32) - low) * (255.0 / max(float(high - low), 1.0))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def decode_visible(data: bytes, source: str = "upload") -> np.ndarray:
    buffer = np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"empty image: {source}")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unsupported or corrupt image: {source}")
    return stretch_channel(image.max(axis=2))


def read_visible(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"cannot read image: {path}")
    return decode_visible(data.tobytes(), str(path))
