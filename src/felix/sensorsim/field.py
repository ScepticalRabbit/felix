# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation


class IField(ABC):
    @abstractmethod
    def get_sim_data(self) -> object:
        pass

    @abstractmethod
    def get_time_steps(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_all_components(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def get_component_index(self, comp_key: str) -> int:
        pass

    @abstractmethod
    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        pass
