import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SystemConfig
from .encoding import IrisCode
from .io_utils import read_visible
from .pipeline import IrisPipeline
from .protection import FuzzyVault
from .segmentation import SegmentationError


@dataclass
class VaultResult:
    key_symbols: int
    key_bits: int
    eyes: int
    genuine_trials: int
    genuine_released: int
    impostor_trials: int
    impostor_released: int

    @property
    def frr(self) -> float:
        return 1.0 - self.genuine_released / self.genuine_trials if self.genuine_trials else 0.0

    @property
    def far(self) -> float:
        return self.impostor_released / self.impostor_trials if self.impostor_trials else 0.0

    def summary(self) -> str:
        return (
            f"key {self.key_bits} bits ({self.key_symbols} symbols): eyes {self.eyes}, genuine released {self.genuine_released}/{self.genuine_trials} "
            f"(FRR {self.frr * 100:.1f}%), impostor released {self.impostor_released}/{self.impostor_trials} (FAR {self.far * 100:.4f}%)"
        )


def extract_codes(
    records: Sequence[Tuple[str, Path]],
    config: SystemConfig,
    reader: Callable[[Path], np.ndarray] = read_visible,
) -> Dict[str, List[IrisCode]]:
    pipeline = IrisPipeline(config)
    codes: Dict[str, List[IrisCode]] = {}
    for label, path in sorted(records):
        try:
            features = pipeline.extract(reader(path))
        except (SegmentationError, ValueError):
            continue
        if features.quality.passed:
            codes.setdefault(label, []).append(features.code)
    return codes


def evaluate_vault(
    codes: Dict[str, List[IrisCode]],
    config: SystemConfig,
    key_symbols: int,
    enrol_count: int = 2,
    impostors_per_eye: Optional[int] = None,
    seed: int = 5,
) -> VaultResult:
    vault = FuzzyVault(key_symbols)
    rng = np.random.default_rng(seed)
    eligible = [label for label, items in codes.items() if len(items) > enrol_count]
    genuine_trials = 0
    genuine_released = 0
    impostor_trials = 0
    impostor_released = 0
    for label in eligible:
        symbols = [secrets.randbelow(128) for _ in range(key_symbols)]
        locked = [vault.lock(code, symbols) for code in codes[label][:enrol_count]]
        records = [record for record, _ in locked]
        key = locked[0][1]
        for probe in codes[label][enrol_count:]:
            genuine_trials += 1
            for record in records:
                if vault.unlock(record, probe, config.matching) == key:
                    genuine_released += 1
                    break
        others = [(other, probe) for other in codes if other != label for probe in codes[other]]
        if impostors_per_eye is not None and len(others) > impostors_per_eye:
            picks = rng.choice(len(others), impostors_per_eye, replace=False)
            others = [others[int(index)] for index in picks]
        for _, probe in others:
            impostor_trials += 1
            for record in records:
                if vault.unlock(record, probe, config.matching) is not None:
                    impostor_released += 1
                    break
    return VaultResult(
        key_symbols=key_symbols,
        key_bits=vault.key_bits,
        eyes=len(eligible),
        genuine_trials=genuine_trials,
        genuine_released=genuine_released,
        impostor_trials=impostor_trials,
        impostor_released=impostor_released,
    )
