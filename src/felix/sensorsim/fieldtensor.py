# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.dataio.simdata import SimData
from pyvale.dataio.meshconv import enforce_mesh_convention
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from felix.sensorsim.enums import EDim
from felix.sensorsim.field import IField
from felix.sensorsim.simtools import sample_simdata_field


class FieldTensor(IField):
    __slots__ = (
        "_norm_comp_keys",
        "_dev_comp_keys",
        "_spatial_dims",
        "_sim_data",
        "_visualiser",
    )

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
        self._sim_data = None
        self._visualiser = None
        self.set_sim_data(sim_data)

    @property
    def sim_data(self) -> SimData:
        return self._sim_data

    def set_sim_data(self, sim_data: SimData) -> None:
        if sim_data.connect is not None:
            sim_data = enforce_mesh_convention(sim_data)
        self._sim_data = sim_data
        try:
            import pyvale.sensorsim.enums as pe

            is_2d = (
                self._spatial_dims == EDim.TWOD
                or getattr(self._spatial_dims, "value", None) == 2
                or self._spatial_dims == 2
            )
            py_dim = pe.EDim.TWOD if is_2d else pe.EDim.THREED
            self._visualiser = simdata_to_pyvista_vis(sim_data, py_dim)
        except Exception:
            self._visualiser = None

    def get_sim_data(self) -> SimData:
        return self._sim_data

    def get_visualiser(self) -> object:
        return self._visualiser

    def get_time_steps(self) -> np.ndarray:
        if self._sim_data.time is None:
            return np.array([0.0], dtype=np.float64)
        return self._sim_data.time

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
        comp_keys = self._norm_comp_keys + self._dev_comp_keys
        truth, _, _, _, _ = sample_simdata_field(
            sim_data=self._sim_data,
            comp_keys=comp_keys,
            spatial_dims=self._spatial_dims,
            points=points,
            times=times,
            angles=angles,
            is_tensor=True,
            error_specs=None,
        )
        return truth
