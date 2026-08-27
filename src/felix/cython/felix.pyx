# cython: language_level=3
# --------------------------------------------------------------------------
# Felix: A High Performance Sensor Simulation Core
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint8_t, uint32_t, uint64_t
cimport numpy as cnp
import numpy as np

from felix.cython cimport felix as cf

cnp.import_array()


def get_last_error() -> str:
    """Return the last error string from the Felix Zig core."""
    cdef uint8_t out[512]
    cdef size_t msg_len = cf.felixGetLastError(out, 512)
    return bytes(out[:msg_len]).decode("utf-8", errors="replace")


def print_sensor_data(
    cnp.ndarray positions,
    object sample_times=None,
    object spatial_dims=None,
) -> None:
    cdef cnp.ndarray[double, ndim=2, mode="c"] pos_c = np.ascontiguousarray(
        positions, dtype=np.float64
    )
    cdef const double *pos_ptr = <const double *>pos_c.data
    cdef size_t pos_len = pos_c.size

    cdef cnp.ndarray[double, ndim=1, mode="c"] st_c
    cdef const double *st_ptr = NULL
    cdef size_t st_len = 0
    if sample_times is not None:
        st_c = np.ascontiguousarray(sample_times, dtype=np.float64)
        st_ptr = <const double *>st_c.data
        st_len = st_c.size

    cdef cnp.ndarray[double, ndim=1, mode="c"] sd_c
    cdef const double *sd_ptr = NULL
    cdef size_t sd_len = 0
    if spatial_dims is not None:
        sd_c = np.ascontiguousarray(spatial_dims, dtype=np.float64)
        sd_ptr = <const double *>sd_c.data
        sd_len = sd_c.size

    cf.felixPrintSensorData(
        pos_ptr,
        pos_len,
        st_ptr,
        st_len,
        sd_ptr,
        sd_len,
    )


def simulate_point_sensors(
    cnp.ndarray coords,
    cnp.ndarray connect,
    int elem_type,
    cnp.ndarray nodal_fields,
    cnp.ndarray sim_times,
    cnp.ndarray positions,
    object sample_times=None,
    object rot_matrices=None,
    int spatial_dims=3,
    bint is_tensor=False,
    list error_specs_list=None,
):
    """Run full point sensor simulation in Zig."""
    cdef cnp.ndarray[double, ndim=2, mode="c"] coords_c = np.ascontiguousarray(
        coords, dtype=np.float64
    )
    cdef cnp.ndarray[size_t, ndim=2, mode="c"] connect_c = np.ascontiguousarray(
        connect, dtype=np.uint64
    )
    cdef cnp.ndarray[double, ndim=3, mode="c"] fields_c = np.ascontiguousarray(
        nodal_fields, dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=1, mode="c"] sim_times_c = np.ascontiguousarray(
        sim_times, dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=2, mode="c"] pos_c = np.ascontiguousarray(
        positions, dtype=np.float64
    )

    cdef cf.SimMeshInput mesh_in
    mesh_in.coords_ptr = <const double *>coords_c.data
    mesh_in.num_nodes = coords_c.shape[0]
    mesh_in.connect_ptr = <const size_t *>connect_c.data
    mesh_in.num_elements = connect_c.shape[0]
    mesh_in.elem_type = <uint32_t>elem_type
    mesh_in.nodal_fields_ptr = <const double *>fields_c.data
    mesh_in.num_components = fields_c.shape[1]
    mesh_in.sim_times_ptr = <const double *>sim_times_c.data
    mesh_in.num_sim_times = sim_times_c.shape[0]

    cdef cnp.ndarray[double, ndim=1, mode="c"] st_c
    cdef cnp.ndarray[double, ndim=3, mode="c"] rot_c

    cdef cf.SensorArrayInput sensor_in
    sensor_in.positions_ptr = <const double *>pos_c.data
    sensor_in.num_sensors = pos_c.shape[0]
    if sample_times is not None:
        st_c = np.ascontiguousarray(sample_times, dtype=np.float64)
        sensor_in.sample_times_ptr = <const double *>st_c.data
        sensor_in.num_sample_times = st_c.shape[0]
    else:
        sensor_in.sample_times_ptr = NULL
        sensor_in.num_sample_times = 0

    if rot_matrices is not None:
        rot_c = np.ascontiguousarray(rot_matrices, dtype=np.float64)
        sensor_in.rot_matrices_ptr = <const double *>rot_c.data
        sensor_in.num_rot_matrices = rot_c.shape[0]
    else:
        sensor_in.rot_matrices_ptr = NULL
        sensor_in.num_rot_matrices = 0

    sensor_in.spatial_dims = <uint32_t>spatial_dims
    sensor_in.is_tensor = 1 if is_tensor else 0

    # Build ErrorSpec array
    cdef size_t num_errors = 0
    cdef cf.ErrorSpec *error_specs_ptr = NULL
    cdef list temp_tables = []
    cdef list temp_polys = []
    cdef cnp.ndarray[double, ndim=2, mode="c"] tbl_c
    cdef cnp.ndarray[double, ndim=1, mode="c"] poly_c

    if error_specs_list is not None and len(error_specs_list) > 0:
        num_errors = len(error_specs_list)
        error_specs_ptr = <cf.ErrorSpec *>malloc(num_errors * sizeof(cf.ErrorSpec))
        for ii, spec_dict in enumerate(error_specs_list):
            error_specs_ptr[ii].kind = <uint32_t>spec_dict.get("kind", 0)
            error_specs_ptr[ii].err_type = <uint32_t>spec_dict.get("err_type", 0)
            error_specs_ptr[ii].err_dep = <uint32_t>spec_dict.get("err_dep", 0)
            error_specs_ptr[ii].dist_type = <uint32_t>spec_dict.get("dist_type", 0)
            error_specs_ptr[ii].param0 = <double>spec_dict.get("param0", 0.0)
            error_specs_ptr[ii].param1 = <double>spec_dict.get("param1", 0.0)
            error_specs_ptr[ii].param2 = <double>spec_dict.get("param2", 0.0)
            error_specs_ptr[ii].seed = <uint64_t>spec_dict.get("seed", 0)
            error_specs_ptr[ii].has_seed = 1 if "seed" in spec_dict and spec_dict["seed"] is not None else 0

            if "table" in spec_dict and spec_dict["table"] is not None:
                tbl_c = np.ascontiguousarray(spec_dict["table"], dtype=np.float64)
                temp_tables.append(tbl_c)
                error_specs_ptr[ii].table_ptr = <const double *>tbl_c.data
                error_specs_ptr[ii].table_rows = tbl_c.shape[0]
            else:
                error_specs_ptr[ii].table_ptr = NULL
                error_specs_ptr[ii].table_rows = 0

            if "poly_coeffs" in spec_dict and spec_dict["poly_coeffs"] is not None:
                poly_c = np.ascontiguousarray(spec_dict["poly_coeffs"], dtype=np.float64)
                temp_polys.append(poly_c)
                error_specs_ptr[ii].poly_coeffs_ptr = <const double *>poly_c.data
                error_specs_ptr[ii].poly_coeffs_len = poly_c.shape[0]
            else:
                error_specs_ptr[ii].poly_coeffs_ptr = NULL
                error_specs_ptr[ii].poly_coeffs_len = 0

    cdef size_t n_sensors = sensor_in.num_sensors
    cdef size_t n_comps = mesh_in.num_components
    cdef size_t n_out_times = sensor_in.num_sample_times if sensor_in.num_sample_times > 0 else mesh_in.num_sim_times

    cdef cnp.ndarray[double, ndim=3, mode="c"] truth = np.zeros(
        (n_sensors, n_comps, n_out_times), dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=3, mode="c"] meas = np.zeros(
        (n_sensors, n_comps, n_out_times), dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=3, mode="c"] errs_sys = np.zeros(
        (n_sensors, n_comps, n_out_times), dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=3, mode="c"] errs_rand = np.zeros(
        (n_sensors, n_comps, n_out_times), dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=3, mode="c"] errs_total = np.zeros(
        (n_sensors, n_comps, n_out_times), dtype=np.float64
    )

    try:
        status = cf.felixSimulatePointSensors(
            &mesh_in,
            &sensor_in,
            error_specs_ptr,
            num_errors,
            <double *>truth.data,
            <double *>meas.data,
            <double *>errs_sys.data,
            <double *>errs_rand.data,
            <double *>errs_total.data,
        )
        if status != 0:
            err_msg = get_last_error()
            raise RuntimeError(f"Felix simulation failed: {err_msg}")
    finally:
        if error_specs_ptr != NULL:
            free(error_specs_ptr)

    return truth, meas, errs_sys, errs_rand, errs_total
