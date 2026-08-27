from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(slots=True)
class SensorData:
    positions: np.ndarray | None = None
    sample_times: np.ndarray | None = None
    angles: tuple[Rotation, ...] | None = None
    spatial_averager: object | None = None
    spatial_dims: np.ndarray | None = None
