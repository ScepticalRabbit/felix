from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from pyvale.dataio.meshconv import enforce_mesh_convention
from pyvale.dataio.simdata import SimData
from scipy.spatial.transform import Rotation

from felix.cython import felix as fc
from felix.python.enums import EDim


class IField:
    """Compatibility adapter implemented by Felix field configurations."""


@dataclass(slots=True)
class FieldScalar(IField):
    _sim_data: SimData
    _comp_key: str
    _spatial_dims: EDim

    def __init__(
        self,
        sim_data: SimData,
        comp_key: str,
        spatial_dims: EDim,
    ) -> None:
        self._comp_key = comp_key
        self._spatial_dims = spatial_dims
        self.set_sim_data(sim_data)

    def set_sim_data(self, sim_data: SimData) -> None:
        self._sim_data = _enforce_mesh(sim_data)

    def get_sim_data(self) -> SimData:
        return self._sim_data

    def get_time_steps(self) -> np.ndarray:
        return _get_time_steps(self._sim_data)

    def get_all_components(self) -> tuple[str, ...]:
        return (self._comp_key,)

    def get_component_index(self, comp_key: str) -> int:
        if comp_key != self._comp_key:
            raise ValueError(f"Unknown field component: {comp_key}")
        return 0

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        return fc.sample_field_config(self, points, times, angles)[0]


@dataclass(slots=True)
class FieldVector(IField):
    _sim_data: SimData
    _comp_keys: tuple[str, ...]
    _spatial_dims: EDim

    def __init__(
        self,
        sim_data: SimData,
        comp_keys: tuple[str, ...],
        spatial_dims: EDim,
    ) -> None:
        self._comp_keys = comp_keys
        self._spatial_dims = spatial_dims
        self.set_sim_data(sim_data)

    def set_sim_data(self, sim_data: SimData) -> None:
        self._sim_data = _enforce_mesh(sim_data)

    def get_sim_data(self) -> SimData:
        return self._sim_data

    def get_time_steps(self) -> np.ndarray:
        return _get_time_steps(self._sim_data)

    def get_all_components(self) -> tuple[str, ...]:
        return self._comp_keys

    def get_component_index(self, comp_key: str) -> int:
        return self._comp_keys.index(comp_key)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        return fc.sample_field_config(self, points, times, angles)[0]


@dataclass(slots=True)
class FieldTensor(IField):
    _sim_data: SimData
    _norm_comp_keys: tuple[str, ...]
    _dev_comp_keys: tuple[str, ...]
    _spatial_dims: EDim

    def __init__(
        self,
        sim_data: SimData,
        norm_comp_keys: tuple[str, ...],
        dev_comp_keys: tuple[str, ...],
        spatial_dims: EDim,
    ) -> None:
        self._norm_comp_keys = norm_comp_keys
        self._dev_comp_keys = dev_comp_keys
        self._spatial_dims = spatial_dims
        self.set_sim_data(sim_data)

    def set_sim_data(self, sim_data: SimData) -> None:
        self._sim_data = _enforce_mesh(sim_data)

    def get_sim_data(self) -> SimData:
        return self._sim_data

    def get_time_steps(self) -> np.ndarray:
        return _get_time_steps(self._sim_data)

    def get_all_components(self) -> tuple[str, ...]:
        return self._norm_comp_keys + self._dev_comp_keys

    def get_component_index(self, comp_key: str) -> int:
        return self.get_all_components().index(comp_key)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        return fc.sample_field_config(self, points, times, angles)[0]


FieldSpec: TypeAlias = FieldScalar | FieldVector | FieldTensor


def _enforce_mesh(sim_data: SimData) -> SimData:
    if sim_data.connect is None:
        return sim_data
    return enforce_mesh_convention(sim_data)


def _get_time_steps(sim_data: SimData) -> np.ndarray:
    if sim_data.time is None:
        return np.array([0.0], dtype=np.float64)
    return sim_data.time
