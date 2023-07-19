from dataclasses import dataclass

import numpy as np

from .utils import similarity_matrix


@dataclass
class Result:
    angle: float
    tvec: tuple[float, float]
    scale: float
    dangle: float
    dscale: float
    dt: float

    @property
    def transformation(self) -> np.ndarray:
        affine = similarity_matrix(self.scale, self.angle, self.tvec)

        return affine
