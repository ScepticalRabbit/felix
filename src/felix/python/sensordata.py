# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
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
