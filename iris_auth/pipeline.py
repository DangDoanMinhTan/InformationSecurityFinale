from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SystemConfig
from .encoding import IrisCode, encode
from .io_utils import read_gray
from .normalization import NormalizedIris, normalize
from .quality import QualityReport, assess_quality
from .segmentation import Segmentation, SegmentationError, segment


@dataclass
class IrisFeatures:
    code: IrisCode
    segmentation: Segmentation
    normalized: NormalizedIris
    quality: QualityReport


class IrisPipeline:
    def __init__(self, config: SystemConfig = SystemConfig()):
        self.config = config

    def extract(self, image: np.ndarray) -> IrisFeatures:
        segmentation = segment(image, self.config.segmentation)
        normalized = normalize(segmentation, self.config.normalization)
        quality = assess_quality(segmentation, normalized, self.config.quality)
        try:
            code = encode(normalized, self.config.encoding)
        except ValueError as error:
            raise SegmentationError(str(error)) from error
        return IrisFeatures(code=code, segmentation=segmentation, normalized=normalized, quality=quality)

    def extract_path(self, path) -> IrisFeatures:
        return self.extract(read_gray(Path(path)))
