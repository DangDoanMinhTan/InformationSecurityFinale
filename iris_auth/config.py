from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class SegmentationConfig:
    max_width: int = 320
    min_pupil_ratio: float = 0.06
    max_pupil_ratio: float = 0.27
    min_iris_ratio: float = 0.20
    max_iris_ratio: float = 0.47
    specular_threshold: int = 235
    max_specular_area_ratio: float = 0.004
    pupil_percentiles: Tuple[int, ...] = (2, 4, 6, 9, 12, 16, 20, 25, 30)
    pupil_sigma: float = 1.5
    limbus_sigma: float = 1.5
    limbus_sector_degrees: float = 38.0
    center_search_ratio: float = 0.04
    eyelid_min_score: float = 3.0
    eyelid_curvature_range: Tuple[float, float] = (0.0005, 0.012)
    lash_darkness: float = 0.35
    eyelid_margin: float = 3.0
    min_pupil_edge: float = 12.0
    edge_override_ratio: float = 1.5


@dataclass(frozen=True)
class NormalizationConfig:
    radial_resolution: int = 64
    angular_resolution: int = 512
    radial_start: float = 0.08
    radial_end: float = 0.96


@dataclass(frozen=True)
class EncodingConfig:
    radial_bands: int = 8
    angular_samples: int = 128
    wavelengths: Tuple[float, ...] = (12.0, 24.0)
    bandwidth_ratio: float = 0.56
    radial_sigma: float = 4.0
    min_valid_fraction: float = 0.65
    min_magnitude_ratio: float = 0.15


@dataclass(frozen=True)
class MatchingConfig:
    threshold: float = 0.32
    rotation_shifts: int = 7
    min_overlap_bits: int = 512
    degrees_of_freedom: int = 249


@dataclass(frozen=True)
class QualityConfig:
    min_sharpness: float = 2.5
    min_usable_fraction: float = 0.40
    min_pupil_iris_ratio: float = 0.15
    max_pupil_iris_ratio: float = 0.80
    min_inside_fraction: float = 0.85
    min_limbus_strength: float = 2.0


@dataclass(frozen=True)
class PadConfig:
    enabled: bool = True
    max_spectral_peak: float = 1.8
    require_pupil_reaction: bool = False
    min_pupil_constriction: float = 0.08
    spectral_band: Tuple[float, float] = (0.12, 0.5)
    min_detail_ratio: Optional[float] = None
    detail_fine_band: Tuple[float, float] = (0.28, 0.5)
    detail_coarse_band: Tuple[float, float] = (0.04, 0.14)


@dataclass(frozen=True)
class SecurityConfig:
    max_failed_attempts: int = 5
    max_failed_attempts_with_pad: int = 10
    lockout_seconds: int = 300
    target_system_far: float = 1e-6
    reject_duplicate_enrollment: bool = True
    expose_diagnostics: bool = False
    cancelable_templates: bool = True
    vault_key_symbols: int = 20


@dataclass(frozen=True)
class SystemConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    pad: PadConfig = field(default_factory=PadConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


def real_eye_config() -> SystemConfig:
    base = SystemConfig()
    return SystemConfig(
        segmentation=replace(
            base.segmentation, max_specular_area_ratio=0.01, eyelid_min_score=2.0, max_iris_ratio=0.5, lash_darkness=0.2
        ),
        normalization=base.normalization,
        encoding=base.encoding,
        matching=replace(base.matching, threshold=0.36, rotation_shifts=11),
        quality=replace(base.quality, min_sharpness=1.5, min_usable_fraction=0.30, min_limbus_strength=1.0),
        pad=replace(base.pad, enabled=False, max_spectral_peak=1.6, min_detail_ratio=-2.47),
        security=base.security,
    )
