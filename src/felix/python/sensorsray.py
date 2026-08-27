# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Ray-casting sensor array (LIDAR, pyrometer, line-of-sight) in Felix."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from pyvale.dataio.simdata import SimData
from felix.python.enums import EDim, ERayMode
from felix.python.errgraph import ErrGraph
from felix.python.errspecs import ErrIntOpts, IErrSimulator
from felix.python.fieldspecs import IField
from felix.python.sensordata import SensorData
from felix.python.sensordescriptor import SensorDescriptor
from felix.python.sensorspoint import ErrIntegrator, ISensorArray
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis


class SensorsRay(ISensorArray):
    """Ray-casting sensor array utilizing mesh surface intersection."""

    __slots__ = (
        "_sim_data",
        "_field",
        "_disp_field",
        "_ray_origins",
        "_ray_directions",
        "_max_distance",
        "_sample_times",
        "_mode",
        "_descriptor",
        "_sensor_data",
        "_truth",
        "_measurements",
        "_error_integrator",
    )

    def __init__(
        self,
        sim_data: SimData,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        field: IField | None = None,
        disp_field: IField | None = None,
        max_distance: float = 1000.0,
        sample_times: np.ndarray | None = None,
        mode: ERayMode = ERayMode.DISTANCE,
        descriptor: SensorDescriptor | None = None,
    ) -> None:
        self._sim_data = sim_data
        self._field = field
        self._disp_field = disp_field

        origins = np.asarray(ray_origins, dtype=np.float64)
        if origins.ndim == 1:
            origins = origins[np.newaxis, :]
        self._ray_origins = origins

        dirs = np.asarray(ray_directions, dtype=np.float64)
        if dirs.ndim == 1:
            dirs = dirs[np.newaxis, :]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._ray_directions = dirs / norms

        self._max_distance = float(max_distance)

        if sample_times is None:
            if field is not None:
                sample_times = field.get_time_steps()
            elif sim_data.time is not None:
                sample_times = sim_data.time
            else:
                sample_times = np.array([0.0])
        self._sample_times = sample_times

        self._mode = mode

        if descriptor is None:
            tag = "LIDAR" if mode == ERayMode.DISTANCE else "RAY"
            descriptor = SensorDescriptor(name="Ray Sensor", tag=tag)
        self._descriptor = descriptor

        self._sensor_data = SensorData(
            positions=self._ray_origins,
            sample_times=self._sample_times,
        )

        self._error_integrator = None
        self._truth = None
        self._measurements = None

    def get_ray_origins(self) -> np.ndarray:
        return self._ray_origins

    def get_ray_directions(self) -> np.ndarray:
        return self._ray_directions

    def get_mode(self) -> ERayMode:
        return self._mode

    def get_descriptor(self) -> SensorDescriptor:
        return self._descriptor

    def get_field(self) -> IField | None:
        return self._field

    def get_sensor_data(self) -> SensorData:
        return self._sensor_data

    def get_sensor_data_nominal(self) -> SensorData:
        return self._sensor_data

    def get_all_components(self) -> tuple[str, ...]:
        if self._mode == ERayMode.DISTANCE:
            return ("standoff_distance",)
        if self._mode == ERayMode.SURFACE_FIELD:
            if self._field is not None:
                return self._field.get_all_components()
            return ("surface_value",)
        return ("path_integral",)

    def get_measurement_shape(self) -> tuple[int, int, int]:
        n_rays = self._ray_origins.shape[0]
        n_comps = len(self.get_all_components())
        n_times = self._sample_times.shape[0]
        return (n_rays, n_comps, n_times)

    def get_sample_times(self) -> np.ndarray:
        return self._sample_times

    def _build_surface_grid(self) -> pv.PolyData:
        is_3d = self._sim_data.coords.shape[1] == 3 and np.any(
            np.abs(self._sim_data.coords[:, 2]) > 1e-12
        )
        s_dim = 3 if is_3d else 2
        grid = simdata_to_pyvista_vis(self._sim_data, spatial_dims=s_dim)
        if isinstance(grid, pv.UnstructuredGrid):
            return grid.extract_surface(algorithm="dataset_surface")
        if isinstance(grid, pv.PolyData):
            return grid
        return pv.PolyData(self._sim_data.coords)

    def calc_truth(self) -> np.ndarray:
        n_rays = self._ray_origins.shape[0]
        n_times = self._sample_times.shape[0]
        n_comps = len(self.get_all_components())

        surface = self._build_surface_grid()
        base_points = np.array(surface.points, copy=True)

        truth = np.zeros((n_rays, n_comps, n_times), dtype=np.float64)

        for tt, t_val in enumerate(self._sample_times):
            if self._disp_field is not None:
                t_arr = np.array([t_val])
                disp_samples = self._disp_field.sample_field(
                    base_points, times=t_arr
                )
                n_disp = disp_samples.shape[1]
                deformed_points = np.array(base_points, copy=True)
                deformed_points[:, :n_disp] += disp_samples[:, :n_disp, 0]
                surface.points = deformed_points

            for rr in range(n_rays):
                p0 = self._ray_origins[rr]
                d = self._ray_directions[rr]
                p1 = p0 + self._max_distance * d

                intersection_pts, _ = surface.ray_trace(
                    p0, p1, first_point=True
                )

                if intersection_pts.size >= 3:
                    hit_pt = (
                        intersection_pts[:3]
                        if intersection_pts.ndim == 1
                        else intersection_pts[0]
                    )
                    dist = float(np.linalg.norm(hit_pt - p0))

                    if self._mode == ERayMode.DISTANCE:
                        truth[rr, 0, tt] = dist
                    elif self._mode == ERayMode.SURFACE_FIELD:
                        if self._field is not None:
                            val = self._field.sample_field(
                                hit_pt.reshape(1, 3),
                                times=np.array([t_val]),
                            )
                            truth[rr, :, tt] = val[0, :, 0]
                        else:
                            truth[rr, 0, tt] = dist
                    else:
                        if self._field is not None:
                            n_quad = 10
                            s_nodes = np.linspace(0.0, dist, n_quad)
                            ds = (
                                dist / (n_quad - 1)
                                if n_quad > 1
                                else dist
                            )
                            seg_pts = (
                                p0.reshape(1, 3)
                                + s_nodes[:, np.newaxis]
                                * d.reshape(1, 3)
                            )
                            seg_vals = self._field.sample_field(
                                seg_pts, times=np.array([t_val])
                            )
                            truth[rr, 0, tt] = (
                                np.sum(seg_vals[:, 0, 0]) * ds
                            )
                        else:
                            truth[rr, 0, tt] = dist
                else:
                    truth[rr, :, tt] = (
                        self._max_distance
                        if self._mode == ERayMode.DISTANCE
                        else np.nan
                    )

        self._truth = truth
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth()
        return self._truth

    def set_error_chain(
        self,
        err_chain: list[IErrSimulator] | ErrGraph | None,
        err_int_opts: ErrIntOpts | None = None,
    ) -> None:
        if err_chain is None:
            self._error_integrator = None
            return None

        if isinstance(err_chain, ErrGraph):
            self._error_integrator = err_chain
            return None

        self._error_integrator = ErrIntegrator(
            err_chain=err_chain,
            sensor_data_initial=self.get_sensor_data(),
            meas_shape=self.get_measurement_shape(),
            err_int_opts=err_int_opts,
        )
        return None

    def set_error_graph(self, err_graph: ErrGraph | None) -> None:
        self._error_integrator = err_graph

    def sim_measurements(self) -> np.ndarray:
        truth = self.get_truth()
        if self._error_integrator is None:
            self._measurements = truth
            return self._measurements

        if isinstance(self._error_integrator, ErrGraph):
            total_err = self._error_integrator.calc_errors_from_graph(truth)
        else:
            total_err = self._error_integrator.calc_errors_from_chain(truth)

        self._measurements = truth + total_err
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.sim_measurements()
        return self._measurements

    def get_errors_systematic(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_systematic()

    def get_errors_random(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_random()

    def get_errors_total(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_total()
