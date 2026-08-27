# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from dataclasses import dataclass
from typing import Sequence, TypeAlias

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

    def get_visualiser(self) -> object:
        return _create_visualiser(self._sim_data, self._spatial_dims)

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

    def get_visualiser(self) -> object:
        return _create_visualiser(self._sim_data, self._spatial_dims)

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
        norm_comp_keys: tuple[str, ...] | Sequence[str] | None = None,
        dev_comp_keys: tuple[str, ...] | Sequence[str] | EDim | None = None,
        spatial_dims: EDim | None = None,
        comp_keys: tuple[str, ...] | Sequence[str] | None = None,
    ) -> None:
        if comp_keys is not None:
            norm_comp_keys = comp_keys
        all_keys = tuple(norm_comp_keys) if norm_comp_keys is not None else ()
        if isinstance(dev_comp_keys, EDim):
            s_dims = dev_comp_keys
            if s_dims == EDim.TWOD:
                self._norm_comp_keys = all_keys[:2]
                self._dev_comp_keys = all_keys[2:]
            else:
                self._norm_comp_keys = all_keys[:3]
                self._dev_comp_keys = all_keys[3:]
            self._spatial_dims = s_dims
        elif spatial_dims is not None:
            if dev_comp_keys is None:
                if spatial_dims == EDim.TWOD:
                    self._norm_comp_keys = all_keys[:2]
                    self._dev_comp_keys = all_keys[2:]
                else:
                    self._norm_comp_keys = all_keys[:3]
                    self._dev_comp_keys = all_keys[3:]
            else:
                self._norm_comp_keys = all_keys
                self._dev_comp_keys = tuple(dev_comp_keys)
            self._spatial_dims = spatial_dims
        else:
            self._norm_comp_keys = all_keys[:2]
            self._dev_comp_keys = all_keys[2:]
            self._spatial_dims = EDim.TWOD
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

    def get_visualiser(self) -> object:
        return _create_visualiser(self._sim_data, self._spatial_dims)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        return fc.sample_field_config(self, points, times, angles)[0]


FieldSpec: TypeAlias = FieldScalar | FieldVector | FieldTensor


def _create_visualiser(sim_data: SimData, spatial_dims: EDim) -> object:
    try:
        import pyvale.sensorsim.enums as pe
        from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis

        is_2d = (
            spatial_dims == EDim.TWOD
            or getattr(spatial_dims, "value", None) == 2
            or spatial_dims == 2
        )
        py_dim = pe.EDim.TWOD if is_2d else pe.EDim.THREED
        return simdata_to_pyvista_vis(sim_data, py_dim)
    except Exception:
        return None


def _enforce_mesh(sim_data: SimData) -> SimData:
    if sim_data.connect is None:
        return sim_data
    return enforce_mesh_convention(sim_data)


def _get_time_steps(sim_data: SimData) -> np.ndarray:
    if sim_data.time is None:
        return np.array([0.0], dtype=np.float64)
    return sim_data.time
