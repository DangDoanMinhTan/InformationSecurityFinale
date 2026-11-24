import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .authenticator import IrisAuthenticator, Outcome
from .config import SystemConfig
from .io_utils import read_visible
from .pad import PresentationAttackDetector
from .pipeline import IrisPipeline
from .segmentation import Segmentation, SegmentationError
from .templatestore import TemplateStore

ATTACK_KINDS = ("halftone_print", "screen_replay", "photo_print")

ATTACK_TITLES = {
    "halftone_print": "Inkjet/laser halftone print",
    "screen_replay": "Screen replay (phone or tablet)",
    "photo_print": "Continuous-tone photo print",
}


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def recapture(image: np.ndarray, rng: np.random.Generator, blur: float, contrast: float, noise: float) -> np.ndarray:
    height, width = image.shape
    angle = float(rng.uniform(-2.0, 2.0))
    scale = float(rng.uniform(0.97, 1.03))
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    softened = cv2.GaussianBlur(warped, (0, 0), blur)
    mean = float(softened.mean())
    adjusted = (softened - mean) * contrast + mean * float(rng.uniform(0.9, 1.05))
    adjusted += rng.normal(0.0, noise, adjusted.shape).astype(np.float32)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def halftone_print(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray = to_gray(image)
    height, width = gray.shape
    factor = 2
    upsampled = cv2.resize(gray, (width * factor, height * factor), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    period = float(rng.uniform(5.0, 12.0))
    angle = math.radians(float(rng.choice([15.0, 45.0, 75.0])))
    yy, xx = np.mgrid[0:height * factor, 0:width * factor].astype(np.float32)
    u = xx * math.cos(angle) + yy * math.sin(angle)
    v = -xx * math.sin(angle) + yy * math.cos(angle)
    screen = 0.5 + 0.25 * (np.cos(2 * math.pi * u / period) + np.cos(2 * math.pi * v / period))
    source = np.clip((upsampled / 255.0 - 0.08) / 0.88, 0.0, 1.0)
    paper = float(rng.uniform(200.0, 235.0))
    ink = float(rng.uniform(25.0, 50.0))
    printed = ink + (paper - ink) * (source > screen).astype(np.float32)
    captured = cv2.resize(printed, (width, height), interpolation=cv2.INTER_AREA)
    return recapture(captured, rng, float(rng.uniform(0.7, 1.4)), float(rng.uniform(0.85, 1.0)), float(rng.uniform(1.5, 4.0)))


def screen_replay(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray = to_gray(image)
    height, width = gray.shape
    factor = 3
    upsampled = cv2.resize(gray, (width * factor, height * factor), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    pitch = float(rng.uniform(5.0, 9.0))
    yy, xx = np.mgrid[0:height * factor, 0:width * factor].astype(np.float32)
    grid = (0.55 + 0.45 * np.cos(math.pi * xx / pitch) ** 2) * (0.55 + 0.45 * np.cos(math.pi * yy / pitch) ** 2)
    backlit = np.clip(upsampled * grid * float(rng.uniform(1.05, 1.25)), 0, 255)
    small = cv2.resize(backlit, (int(width * rng.uniform(0.9, 1.0)), int(height * rng.uniform(0.9, 1.0))), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return recapture(restored, rng, float(rng.uniform(0.5, 1.1)), float(rng.uniform(0.9, 1.05)), float(rng.uniform(1.5, 4.0)))


def photo_print(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray = to_gray(image).astype(np.float32)
    return recapture(gray, rng, float(rng.uniform(0.9, 1.6)), float(rng.uniform(0.78, 0.92)), float(rng.uniform(2.0, 5.0)))


def static_second_frame(image: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shifted = np.roll(image.astype(np.float32), (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))), axis=(0, 1))
    shifted = shifted * float(rng.uniform(1.0, 1.08)) + rng.normal(0.0, 2.5, shifted.shape).astype(np.float32)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def simulate_pupil_response(image: np.ndarray, segmentation: Segmentation, constriction: float) -> np.ndarray:
    gray = to_gray(image)
    height, width = gray.shape
    scale = segmentation.scale
    cx = segmentation.pupil.x / scale
    cy = segmentation.pupil.y / scale
    pupil = segmentation.pupil.r / scale
    iris = segmentation.iris.r / scale
    reduced = pupil * (1.0 - constriction)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    radius = np.hypot(dx, dy)
    inner = radius * (pupil / reduced)
    outer = pupil + (radius - reduced) * (iris - pupil) / (iris - reduced)
    source = np.where(radius <= reduced, inner, np.where(radius <= iris, outer, radius))
    factor = np.where(radius > 1e-6, source / np.maximum(radius, 1e-6), 1.0).astype(np.float32)
    mapX = (cx + dx * factor).astype(np.float32)
    mapY = (cy + dy * factor).astype(np.float32)
    return cv2.remap(gray, mapX, mapY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


ATTACKS: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "halftone_print": halftone_print,
    "screen_replay": screen_replay,
    "photo_print": photo_print,
}


def simulate_attack(kind: str, image: np.ndarray, seed: int) -> np.ndarray:
    if kind not in ATTACKS:
        raise ValueError(f"unknown attack kind: {kind}")
    return ATTACKS[kind](image, np.random.default_rng(seed))


@dataclass
class AttackReport:
    eyes: int
    genuine_images: int
    genuine_pad_rejected: int
    genuine_accepted_without_pad: int
    genuine_accepted_with_pad: int
    genuine_accepted_with_reflex: int
    genuine_trials: int
    per_attack: Dict[str, Dict[str, float]]
    spectral_genuine: List[float]
    spectral_attacks: Dict[str, List[float]]
    skipped: List[str]

    def summary(self) -> List[str]:
        lines = [
            f"Eyes enrolled: {self.eyes}   genuine photos checked by PAD: {self.genuine_images}   rejected by PAD: {self.genuine_pad_rejected} ({self.rate(self.genuine_pad_rejected, self.genuine_images):.1f}%)",
            f"Genuine verification: accepted {self.genuine_accepted_without_pad}/{self.genuine_trials} without PAD, {self.genuine_accepted_with_pad}/{self.genuine_trials} with PAD, {self.genuine_accepted_with_reflex}/{self.genuine_trials} with PAD plus simulated pupil reflex",
        ]
        for kind, stats in self.per_attack.items():
            lines.append(
                f"{ATTACK_TITLES[kind]}: {int(stats['trials'])} attempts, not acquired {int(stats['not_acquired'])}, "
                f"accepted without PAD {stats['accepted_without_pad'] * 100:.1f}%, accepted with PAD {stats['accepted_with_pad'] * 100:.1f}%, accepted with PAD plus reflex {stats['accepted_with_reflex'] * 100:.1f}%, "
                f"PAD miss rate {stats['pad_miss'] * 100:.1f}%"
            )
        return lines

    @staticmethod
    def rate(count: int, total: int) -> float:
        return 100.0 * count / total if total else 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AttackReport":
        return cls(**data)

    @classmethod
    def merge(cls, parts: Sequence["AttackReport"]) -> "AttackReport":
        if not parts:
            raise ValueError("nothing to merge")
        per_attack: Dict[str, Dict[str, float]] = {}
        spectral_attacks: Dict[str, List[float]] = {kind: [] for kind in ATTACK_KINDS}
        for kind in ATTACK_KINDS:
            totals = {"trials": 0.0, "not_acquired": 0.0, "acquired": 0.0, "accepted_without_pad": 0.0, "accepted_with_pad": 0.0, "accepted_with_reflex": 0.0, "pad_miss": 0.0}
            for part in parts:
                stats = part.per_attack[kind]
                trials = stats["trials"]
                acquired = stats["acquired"]
                totals["trials"] += trials
                totals["not_acquired"] += stats["not_acquired"]
                totals["acquired"] += acquired
                totals["accepted_without_pad"] += stats["accepted_without_pad"] * trials
                totals["accepted_with_pad"] += stats["accepted_with_pad"] * trials
                totals["accepted_with_reflex"] += stats["accepted_with_reflex"] * trials
                totals["pad_miss"] += stats["pad_miss"] * acquired
                spectral_attacks[kind].extend(part.spectral_attacks[kind])
            totals["accepted_without_pad"] /= max(totals["trials"], 1.0)
            totals["accepted_with_pad"] /= max(totals["trials"], 1.0)
            totals["accepted_with_reflex"] /= max(totals["trials"], 1.0)
            totals["pad_miss"] /= max(totals["acquired"], 1.0)
            per_attack[kind] = totals
        return cls(
            eyes=sum(part.eyes for part in parts),
            genuine_images=sum(part.genuine_images for part in parts),
            genuine_pad_rejected=sum(part.genuine_pad_rejected for part in parts),
            genuine_accepted_without_pad=sum(part.genuine_accepted_without_pad for part in parts),
            genuine_accepted_with_pad=sum(part.genuine_accepted_with_pad for part in parts),
            genuine_accepted_with_reflex=sum(part.genuine_accepted_with_reflex for part in parts),
            genuine_trials=sum(part.genuine_trials for part in parts),
            per_attack=per_attack,
            spectral_genuine=[value for part in parts for value in part.spectral_genuine],
            spectral_attacks=spectral_attacks,
            skipped=[item for part in parts for item in part.skipped],
        )


def evaluate_attacks(
    records: Sequence[Tuple[str, Path]],
    config: SystemConfig,
    store_path: Path,
    passphrase: str,
    enrol_count: int = 2,
    repeats: int = 2,
    seed: int = 7,
    reader: Callable[[Path], np.ndarray] = read_visible,
    progress: Optional[Callable[[int, int], None]] = None,
    offset: int = 0,
    limit: Optional[int] = None,
) -> AttackReport:
    groups: Dict[str, List[Path]] = {}
    for label, path in records:
        groups.setdefault(label, []).append(Path(path))
    labels = [label for label in sorted(groups) if len(groups[label]) > enrol_count]
    labels = labels[offset: None if limit is None else offset + limit]
    secure = replace(config.security, max_failed_attempts=10 ** 6, max_failed_attempts_with_pad=10 ** 6, expose_diagnostics=True)
    plain = replace(config, security=secure, pad=replace(config.pad, enabled=False))
    guarded = replace(config, security=secure, pad=replace(config.pad, enabled=True, require_pupil_reaction=False))
    store = TemplateStore(store_path, passphrase)
    enroller = IrisAuthenticator(store, plain)
    undefended = IrisAuthenticator(store, plain)
    defended = IrisAuthenticator(store, guarded)
    reflexive = IrisAuthenticator(store, replace(guarded, pad=replace(guarded.pad, require_pupil_reaction=True)))
    detector = PresentationAttackDetector(guarded.pad)
    pipeline = IrisPipeline(guarded)
    skipped: List[str] = []
    enrolled: List[str] = []
    for label in labels:
        paths = sorted(groups[label])
        try:
            enroller.enroll(label, [reader(path) for path in paths[:enrol_count]], replace=True)
        except Exception as error:
            skipped.append(f"{label}: {error}")
            continue
        enrolled.append(label)
    spectral_genuine: List[float] = []
    genuine_images = 0
    genuine_pad_rejected = 0
    genuine_trials = 0
    genuine_without = 0
    genuine_with = 0
    genuine_reflex = 0
    per_attack: Dict[str, Dict[str, float]] = {
        kind: {"trials": 0, "not_acquired": 0, "accepted_without_pad": 0, "accepted_with_pad": 0, "accepted_with_reflex": 0, "pad_miss": 0, "acquired": 0}
        for kind in ATTACK_KINDS
    }
    spectral_attacks: Dict[str, List[float]] = {kind: [] for kind in ATTACK_KINDS}
    total = len(enrolled)
    for index, label in enumerate(enrolled, start=1):
        paths = sorted(groups[label])
        images = [reader(path) for path in paths]
        for image in images:
            try:
                features = pipeline.extract(image)
            except (SegmentationError, ValueError):
                continue
            result = detector.assess(features.segmentation)
            spectral_genuine.append(result.spectral_peak)
            genuine_images += 1
            if not result.passed:
                genuine_pad_rejected += 1
        probe = images[enrol_count]
        genuine_trials += 1
        if undefended.verify(label, probe).accepted:
            genuine_without += 1
        if defended.verify(label, probe).accepted:
            genuine_with += 1
        rng = np.random.default_rng(seed * 7919 + index)
        try:
            live = simulate_pupil_response(probe, pipeline.extract(probe).segmentation, float(rng.uniform(0.15, 0.35)))
            if reflexive.verify(label, probe, live).accepted:
                genuine_reflex += 1
        except (SegmentationError, ValueError):
            pass
        for kind in ATTACK_KINDS:
            stats = per_attack[kind]
            for repeat in range(repeats):
                attack = simulate_attack(kind, probe, seed * 100003 + index * 131 + repeat * 17 + ATTACK_KINDS.index(kind))
                stats["trials"] += 1
                try:
                    features = pipeline.extract(attack)
                except (SegmentationError, ValueError):
                    stats["not_acquired"] += 1
                    continue
                if not features.quality.passed:
                    stats["not_acquired"] += 1
                    continue
                stats["acquired"] += 1
                assessment = detector.assess(features.segmentation)
                spectral_attacks[kind].append(assessment.spectral_peak)
                if assessment.passed:
                    stats["pad_miss"] += 1
                if undefended.verify(label, attack).accepted:
                    stats["accepted_without_pad"] += 1
                if defended.verify(label, attack).accepted:
                    stats["accepted_with_pad"] += 1
                bright = static_second_frame(attack, seed * 31 + index * 13 + repeat)
                if reflexive.verify(label, attack, bright).accepted:
                    stats["accepted_with_reflex"] += 1
        if progress:
            progress(index, total)
    for kind, stats in per_attack.items():
        trials = max(stats["trials"], 1)
        acquired = max(stats["acquired"], 1)
        stats["accepted_without_pad"] = stats["accepted_without_pad"] / trials
        stats["accepted_with_pad"] = stats["accepted_with_pad"] / trials
        stats["accepted_with_reflex"] = stats["accepted_with_reflex"] / trials
        stats["pad_miss"] = stats["pad_miss"] / acquired
    return AttackReport(
        eyes=len(enrolled),
        genuine_images=genuine_images,
        genuine_pad_rejected=genuine_pad_rejected,
        genuine_accepted_without_pad=genuine_without,
        genuine_accepted_with_pad=genuine_with,
        genuine_accepted_with_reflex=genuine_reflex,
        genuine_trials=genuine_trials,
        per_attack=per_attack,
        spectral_genuine=spectral_genuine,
        spectral_attacks=spectral_attacks,
        skipped=skipped,
    )
