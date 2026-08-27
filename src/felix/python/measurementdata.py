# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Measurement data container for multi-sensor datasets."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from felix.python.sensordescriptor import SensorDescriptor


@dataclass(slots=True)
class MeasurementData:
    """Measurement container wrapping values, times, and sensor metadata."""

    values: np.ndarray
    sample_times: np.ndarray
    positions: np.ndarray
    components: tuple[str, ...]
    units: str | None = None
    descriptor: SensorDescriptor | None = None

    def get_values(self) -> np.ndarray:
        return self.values

    def get_sample_times(self) -> np.ndarray:
        return self.sample_times

    def get_positions(self) -> np.ndarray:
        return self.positions

    def get_components(self) -> tuple[str, ...]:
        return self.components
