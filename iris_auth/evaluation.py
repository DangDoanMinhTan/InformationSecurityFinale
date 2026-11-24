import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SystemConfig
from .io_utils import read_gray
from .matching import false_accept_probability, hamming_matrix
from .pipeline import IrisPipeline
from .segmentation import SegmentationError

OPERATING_THRESHOLDS = (0.26, 0.30, 0.32, 0.33, 0.35)


@dataclass
class EvaluationResult:
    images: int
    acquired: int
    failures: List[str]
    genuine: np.ndarray
    impostor: np.ndarray
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    eer: float
    eer_threshold: float
    operating_points: List[Dict[str, float]]


def error_curves(genuine: np.ndarray, impostor: np.ndarray, steps: int = 1001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.linspace(0.0, 1.0, steps)
    far = np.array([(impostor <= t).mean() for t in thresholds])
    frr = np.array([(genuine > t).mean() for t in thresholds])
    return thresholds, far, frr


def equal_error_rate(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray) -> Tuple[float, float]:
    index = int(np.argmin(np.abs(far - frr)))
    return float((far[index] + frr[index]) / 2.0), float(thresholds[index])


def evaluate_records(
    records: Sequence[Tuple[str, Path]],
    config: SystemConfig,
    progress: Optional[Callable[[int, int], None]] = None,
    reader: Callable[[Path], np.ndarray] = read_gray,
) -> EvaluationResult:
    pipeline = IrisPipeline(config)
    codes = []
    labels = []
    failures: List[str] = []
    for index, (label, path) in enumerate(records, start=1):
        try:
            features = pipeline.extract(reader(path))
        except (SegmentationError, ValueError, FileNotFoundError) as error:
            failures.append(f"{path}: {error}")
            continue
        if not features.quality.passed:
            failures.append(f"{path}: {'; '.join(features.quality.failures)}")
            continue
        codes.append(features.code)
        labels.append(label)
        if progress:
            progress(index, len(records))
    if len(codes) < 2:
        raise ValueError("not enough usable images to evaluate")
    distances = hamming_matrix(codes, config.matching)
    label_array = np.array(labels)
    same = label_array[:, None] == label_array[None, :]
    upper = np.triu_indices(len(codes), k=1)
    pair_same = same[upper]
    genuine = distances[upper][pair_same]
    impostor = distances[upper][~pair_same]
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("need at least two classes with two images each")
    thresholds, far, frr = error_curves(genuine, impostor)
    eer, eer_threshold = equal_error_rate(thresholds, far, frr)
    points = []
    for threshold in OPERATING_THRESHOLDS:
        points.append(
            {
                "threshold": threshold,
                "far": float((impostor <= threshold).mean()),
                "frr": float((genuine > threshold).mean()),
                "model_far": false_accept_probability(threshold, config.matching),
            }
        )
    return EvaluationResult(
        images=len(records),
        acquired=len(codes),
        failures=failures,
        genuine=genuine,
        impostor=impostor,
        thresholds=thresholds,
        far=far,
        frr=frr,
        eer=eer,
        eer_threshold=eer_threshold,
        operating_points=points,
    )


def plot_evaluation(result: EvaluationResult, config: SystemConfig, path: Path, dpi: int = 130) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    bins = np.linspace(0.0, 0.8, 81)
    axes[0].hist(result.genuine, bins=bins, density=True, alpha=0.75, color="#2a9d8f", label=f"Genuine (n={result.genuine.size})")
    axes[0].hist(result.impostor, bins=bins, density=True, alpha=0.75, color="#e76f51", label=f"Impostor (n={result.impostor.size})")
    axes[0].axvline(config.matching.threshold, color="black", linestyle="--", label=f"Threshold {config.matching.threshold:.2f}")
    axes[0].set_xlabel("Hamming distance")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Score distributions")
    axes[0].legend()
    axes[1].plot(result.thresholds, result.far * 100, color="#e76f51", label="FAR")
    axes[1].plot(result.thresholds, result.frr * 100, color="#2a9d8f", label="FRR")
    axes[1].scatter([result.eer_threshold], [result.eer * 100], color="black", zorder=5, label=f"EER {result.eer * 100:.2f}% @ {result.eer_threshold:.3f}")
    axes[1].axvline(config.matching.threshold, color="black", linestyle="--")
    axes[1].set_xlim(0.15, 0.55)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Error rate (%)")
    axes[1].set_title("FAR and FRR versus threshold")
    axes[1].legend()
    floor = 1.0 / max(1, result.impostor.size)
    axes[2].loglog(np.maximum(result.far, floor / 10), np.maximum(result.frr, floor / 10), color="#264653")
    axes[2].set_xlabel("FAR")
    axes[2].set_ylabel("FRR")
    axes[2].set_title("DET curve (floor applied at zero)")
    axes[2].grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def save_report(result: EvaluationResult, config: SystemConfig, directory: Path) -> Dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    figure_path = directory / "evaluation.png"
    json_path = directory / "evaluation.json"
    plot_evaluation(result, config, figure_path)
    summary = {
        "images": result.images,
        "acquired": result.acquired,
        "rejected": len(result.failures),
        "genuine_pairs": int(result.genuine.size),
        "impostor_pairs": int(result.impostor.size),
        "genuine_mean": float(result.genuine.mean()),
        "genuine_std": float(result.genuine.std()),
        "impostor_mean": float(result.impostor.mean()),
        "impostor_std": float(result.impostor.std()),
        "eer": result.eer,
        "eer_threshold": result.eer_threshold,
        "operating_points": result.operating_points,
        "rejected_files": result.failures,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"figure": figure_path, "json": json_path}
