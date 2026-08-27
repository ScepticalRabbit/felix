# cython: language_level=3
# --------------------------------------------------------------------------
# Felix: A virtual sensor laboratory
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


def sample_field_config(
    object field,
    cnp.ndarray positions,
    object sample_times=None,
    object angles=None,
    list error_specs_list=None,
    size_t num_experiments=1,
    uint64_t experiment_seed=0,
):
    """Marshal a Felix field configuration and execute it in Zig."""
    sim_data = field.get_sim_data()
    connect = sim_data.connect
    if connect is None:
        raise ValueError("Felix field sampling requires mesh connectivity")
    if isinstance(connect, dict):
        connect_arr = np.vstack(tuple(connect.values()))
    else:
        connect_arr = connect

    spatial_dims = field._spatial_dims
    is_2d = getattr(spatial_dims, "value", spatial_dims) == 2
    num_nodes_per_elem = connect_arr.shape[1]
    if is_2d:
        if num_nodes_per_elem in (8, 9):
            connect_arr = connect_arr[:, :8]
            elem_type = 5
        elif num_nodes_per_elem == 6:
            connect_arr = connect_arr[:, :3]
            elem_type = 0
        elif num_nodes_per_elem == 4:
            elem_type = 1
        elif num_nodes_per_elem == 3:
            elem_type = 0
        else:
            raise ValueError("Unsupported 2D element node count")
    else:
        if num_nodes_per_elem in (20, 27):
            connect_arr = connect_arr[:, :20]
            elem_type = 6
        elif num_nodes_per_elem == 10:
            connect_arr = connect_arr[:, :4]
            elem_type = 2
        elif num_nodes_per_elem == 8:
            elem_type = 3
        elif num_nodes_per_elem == 4:
            elem_type = 2
        else:
            raise ValueError("Unsupported 3D element node count")

    comp_keys = field.get_all_components()
    sim_times = (
        sim_data.time
        if sim_data.time is not None
        else np.array([0.0], dtype=np.float64)
    )
    field_data = np.empty(
        (sim_data.coords.shape[0], len(comp_keys), sim_times.shape[0]),
        dtype=np.float64,
    )
    for cc, comp_key in enumerate(comp_keys):
        field_data[:, cc, :] = sim_data.node_vars[comp_key]

    rot_matrices = None
    if angles is not None:
        rot_matrices = np.array(
            [rotation.as_matrix().T for rotation in angles],
            dtype=np.float64,
        )

    is_tensor = hasattr(field, "_norm_comp_keys")
    return simulate_point_sensors(
        coords=sim_data.coords,
        connect=connect_arr,
        elem_type=elem_type,
        nodal_fields=field_data,
        sim_times=sim_times,
        positions=positions,
        sample_times=sample_times,
        rot_matrices=rot_matrices,
        spatial_dims=2 if is_2d else 3,
        is_tensor=is_tensor,
        error_specs_list=error_specs_list,
        num_experiments=num_experiments,
        experiment_seed=experiment_seed,
    )


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


def calc_experiment_stats(cnp.ndarray values):
    """Calculate population statistics over axis zero in Zig."""
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("values must contain at least one experiment")

    cdef cnp.ndarray[double, ndim=2, mode="c"] values_c = (
        np.ascontiguousarray(values, dtype=np.float64).reshape(
            values.shape[0], -1
        )
    )
    cdef size_t num_values = values_c.shape[1]
    cdef cnp.ndarray[double, ndim=1, mode="c"] mean = np.empty(num_values)
    cdef cnp.ndarray[double, ndim=1, mode="c"] std = np.empty(num_values)
    cdef cnp.ndarray[double, ndim=1, mode="c"] min_val = np.empty(num_values)
    cdef cnp.ndarray[double, ndim=1, mode="c"] max_val = np.empty(num_values)
    cdef cnp.ndarray[double, ndim=1, mode="c"] median = np.empty(num_values)
    cdef cnp.ndarray[double, ndim=1, mode="c"] var = np.empty(num_values)
    cdef cnp.ndarray[double, ndim=1, mode="c"] mad = np.empty(num_values)

    status = cf.felixCalcExperimentStats(
        <const double *>values_c.data,
        values_c.shape[0],
        num_values,
        <double *>mean.data,
        <double *>std.data,
        <double *>min_val.data,
        <double *>max_val.data,
        <double *>median.data,
        <double *>var.data,
        <double *>mad.data,
    )
    if status != 0:
        raise RuntimeError(f"Felix statistics failed: {get_last_error()}")

    out_shape = tuple(np.asarray(values).shape[1:])
    return tuple(
        arr.reshape(out_shape)
        for arr in (mean, std, min_val, max_val, median, var, mad)
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
    size_t num_experiments=1,
    uint64_t experiment_seed=0,
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
    cdef cnp.ndarray[double, ndim=2, mode="c"] work_pos = np.array(
        pos_c, dtype=np.float64, copy=True, order="C"
    )
    cdef cnp.ndarray[double, ndim=2, mode="c"] scratch_pos = np.array(
        pos_c, dtype=np.float64, copy=True, order="C"
    )

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

    cdef cnp.ndarray[double, ndim=1, mode="c"] work_times
    if sample_times is not None:
        work_times = np.array(st_c, dtype=np.float64, copy=True, order="C")
    else:
        work_times = np.array(
            sim_times_c, dtype=np.float64, copy=True, order="C"
        )
    cdef cnp.ndarray[double, ndim=1, mode="c"] scratch_times = np.array(
        work_times, dtype=np.float64, copy=True, order="C"
    )

    if rot_matrices is not None:
        rot_c = np.ascontiguousarray(rot_matrices, dtype=np.float64)
        sensor_in.rot_matrices_ptr = <const double *>rot_c.data
        sensor_in.num_rot_matrices = rot_c.shape[0]
    else:
        sensor_in.rot_matrices_ptr = NULL
        sensor_in.num_rot_matrices = 0

    cdef cnp.ndarray[double, ndim=3, mode="c"] work_rots
    if rot_matrices is not None:
        work_rots = np.array(rot_c, dtype=np.float64, copy=True, order="C")
    else:
        work_rots = np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :],
            pos_c.shape[0],
            axis=0,
        )
    cdef cnp.ndarray[double, ndim=3, mode="c"] scratch_rots = np.array(
        work_rots, dtype=np.float64, copy=True, order="C"
    )

    sensor_in.spatial_dims = <uint32_t>spatial_dims
    sensor_in.is_tensor = 1 if is_tensor else 0
    sensor_in.work_positions_ptr = <double *>work_pos.data
    sensor_in.work_times_ptr = <double *>work_times.data
    sensor_in.work_rot_matrices_ptr = <double *>work_rots.data
    sensor_in.scratch_positions_ptr = <double *>scratch_pos.data
    sensor_in.scratch_times_ptr = <double *>scratch_times.data
    sensor_in.scratch_rot_matrices_ptr = <double *>scratch_rots.data

    # Build ErrorSpec array
    cdef size_t num_errors = 0
    cdef cf.ErrorSpec *error_specs_ptr = NULL
    cdef cf.FieldPerturbationSpec *field_specs_ptr = NULL
    cdef list temp_tables = []
    cdef list temp_polys = []
    cdef list temp_field_arrays = []
    cdef cnp.ndarray[double, ndim=2, mode="c"] tbl_c
    cdef cnp.ndarray[double, ndim=1, mode="c"] poly_c
    cdef cnp.ndarray[double, ndim=2, mode="c"] field_arr_2d
    cdef cnp.ndarray[uint8_t, ndim=2, mode="c"] field_mask_2d
    cdef cnp.ndarray[double, ndim=1, mode="c"] field_arr_1d
    cdef object field_data
    cdef object dist_obj
    cdef dict dist_spec
    cdef object drift
    cdef object spatial_kind_obj

    if error_specs_list is not None and len(error_specs_list) > 0:
        num_errors = len(error_specs_list)
        error_specs_ptr = <cf.ErrorSpec *>malloc(num_errors * sizeof(cf.ErrorSpec))
        field_specs_ptr = <cf.FieldPerturbationSpec *>malloc(
            num_errors * sizeof(cf.FieldPerturbationSpec)
        )
        for ii, spec_dict in enumerate(error_specs_list):
            error_specs_ptr[ii].kind = <uint32_t>spec_dict.get("kind", 0)
            error_specs_ptr[ii].err_type = <uint32_t>spec_dict.get("err_type", 0)
            error_specs_ptr[ii].err_dep = <uint32_t>spec_dict.get("err_dep", 0)
            error_specs_ptr[ii].dist_type = <uint32_t>spec_dict.get("dist_type", 0)
            error_specs_ptr[ii].param0 = <double>spec_dict.get("param0", 0.0)
            error_specs_ptr[ii].param1 = <double>spec_dict.get("param1", 0.0)
            error_specs_ptr[ii].param2 = <double>spec_dict.get("param2", 0.0)
            error_specs_ptr[ii].seed = <uint64_t>(spec_dict.get("seed") or 0)
            error_specs_ptr[ii].has_seed = 1 if "seed" in spec_dict and spec_dict["seed"] is not None else 0
            error_specs_ptr[ii].field_spec_ptr = NULL

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

            if error_specs_ptr[ii].kind == 12:
                field_data = spec_dict["field_data"]
                field_specs_ptr[ii].pos_offset_ptr = NULL
                field_specs_ptr[ii].pos_lock_ptr = NULL
                field_specs_ptr[ii].angle_offset_ptr = NULL
                field_specs_ptr[ii].angle_lock_ptr = NULL
                field_specs_ptr[ii].time_offset_ptr = NULL
                field_specs_ptr[ii].drift_kind = 0
                field_specs_ptr[ii].drift_param0 = 0.0
                field_specs_ptr[ii].drift_param1 = 0.0
                field_specs_ptr[ii].drift_param2 = 0.0
                field_specs_ptr[ii].drift_poly_ptr = NULL
                field_specs_ptr[ii].drift_poly_len = 0
                field_specs_ptr[ii].spatial_kind = 0
                field_specs_ptr[ii].spatial_dim_x = 0.0
                field_specs_ptr[ii].spatial_dim_y = 0.0
                field_specs_ptr[ii].spatial_dim_z = 0.0

                if field_data.pos_offset_xyz is not None:
                    field_arr_2d = np.ascontiguousarray(
                        np.broadcast_to(
                            field_data.pos_offset_xyz,
                            (pos_c.shape[0], 3),
                        ),
                        dtype=np.float64,
                    )
                    temp_field_arrays.append(field_arr_2d)
                    field_specs_ptr[ii].pos_offset_ptr = <const double *>field_arr_2d.data
                if field_data.pos_lock_xyz is not None:
                    field_mask_2d = np.ascontiguousarray(
                        np.broadcast_to(
                            field_data.pos_lock_xyz,
                            (pos_c.shape[0], 3),
                        ),
                        dtype=np.uint8,
                    )
                    temp_field_arrays.append(field_mask_2d)
                    field_specs_ptr[ii].pos_lock_ptr = <const uint8_t *>field_mask_2d.data
                if field_data.ang_offset_zyx is not None:
                    field_arr_2d = np.ascontiguousarray(
                        np.broadcast_to(
                            field_data.ang_offset_zyx,
                            (pos_c.shape[0], 3),
                        ),
                        dtype=np.float64,
                    )
                    temp_field_arrays.append(field_arr_2d)
                    field_specs_ptr[ii].angle_offset_ptr = <const double *>field_arr_2d.data
                if field_data.ang_lock_zyx is not None:
                    field_mask_2d = np.ascontiguousarray(
                        np.broadcast_to(
                            field_data.ang_lock_zyx,
                            (pos_c.shape[0], 3),
                        ),
                        dtype=np.uint8,
                    )
                    temp_field_arrays.append(field_mask_2d)
                    field_specs_ptr[ii].angle_lock_ptr = <const uint8_t *>field_mask_2d.data
                if field_data.time_offset is not None:
                    field_arr_1d = np.ascontiguousarray(
                        np.broadcast_to(
                            field_data.time_offset,
                            (work_times.shape[0],),
                        ),
                        dtype=np.float64,
                    )
                    temp_field_arrays.append(field_arr_1d)
                    field_specs_ptr[ii].time_offset_ptr = <const double *>field_arr_1d.data

                for aa in range(3):
                    dist_obj = field_data.pos_rand_xyz[aa]
                    dist_spec = {} if dist_obj is None else dist_obj.to_spec_dict()
                    field_specs_ptr[ii].pos_rand[aa].dist_type = <uint32_t>dist_spec.get("dist_type", 0)
                    field_specs_ptr[ii].pos_rand[aa].param0 = <double>dist_spec.get("param0", 0.0)
                    field_specs_ptr[ii].pos_rand[aa].param1 = <double>dist_spec.get("param1", 0.0)
                    field_specs_ptr[ii].pos_rand[aa].param2 = <double>dist_spec.get("param2", 0.0)
                    field_specs_ptr[ii].pos_rand[aa].seed = <uint64_t>(dist_spec.get("seed") or 0)
                    field_specs_ptr[ii].pos_rand[aa].has_seed = 1 if dist_spec.get("seed") is not None else 0

                    dist_obj = field_data.ang_rand_zyx[aa]
                    dist_spec = {} if dist_obj is None else dist_obj.to_spec_dict()
                    field_specs_ptr[ii].angle_rand[aa].dist_type = <uint32_t>dist_spec.get("dist_type", 0)
                    field_specs_ptr[ii].angle_rand[aa].param0 = <double>dist_spec.get("param0", 0.0)
                    field_specs_ptr[ii].angle_rand[aa].param1 = <double>dist_spec.get("param1", 0.0)
                    field_specs_ptr[ii].angle_rand[aa].param2 = <double>dist_spec.get("param2", 0.0)
                    field_specs_ptr[ii].angle_rand[aa].seed = <uint64_t>(dist_spec.get("seed") or 0)
                    field_specs_ptr[ii].angle_rand[aa].has_seed = 1 if dist_spec.get("seed") is not None else 0

                dist_obj = field_data.time_rand
                dist_spec = {} if dist_obj is None else dist_obj.to_spec_dict()
                field_specs_ptr[ii].time_rand.dist_type = <uint32_t>dist_spec.get("dist_type", 0)
                field_specs_ptr[ii].time_rand.param0 = <double>dist_spec.get("param0", 0.0)
                field_specs_ptr[ii].time_rand.param1 = <double>dist_spec.get("param1", 0.0)
                field_specs_ptr[ii].time_rand.param2 = <double>dist_spec.get("param2", 0.0)
                field_specs_ptr[ii].time_rand.seed = <uint64_t>(dist_spec.get("seed") or 0)
                field_specs_ptr[ii].time_rand.has_seed = 1 if dist_spec.get("seed") is not None else 0

                drift = field_data.time_drift
                if drift is not None:
                    dist_spec = drift.to_drift_spec_dict()
                    field_specs_ptr[ii].drift_kind = <uint32_t>dist_spec.get("drift_kind", 0)
                    field_specs_ptr[ii].drift_param0 = <double>dist_spec.get("param0", 0.0)
                    field_specs_ptr[ii].drift_param1 = <double>dist_spec.get("param1", 0.0)
                    field_specs_ptr[ii].drift_param2 = <double>dist_spec.get("param2", 0.0)
                    if dist_spec.get("poly_coeffs") is not None:
                        field_arr_1d = np.ascontiguousarray(
                            dist_spec["poly_coeffs"], dtype=np.float64
                        )
                        temp_field_arrays.append(field_arr_1d)
                        field_specs_ptr[ii].drift_poly_ptr = <const double *>field_arr_1d.data
                        field_specs_ptr[ii].drift_poly_len = field_arr_1d.shape[0]

                if field_data.spatial_averager is not None:
                    spatial_kind_obj = field_data.spatial_averager
                    field_specs_ptr[ii].spatial_kind = <uint32_t>getattr(
                        spatial_kind_obj, "value", spatial_kind_obj
                    )
                if field_data.spatial_dims is not None:
                    field_specs_ptr[ii].spatial_dim_x = field_data.spatial_dims[0]
                    field_specs_ptr[ii].spatial_dim_y = field_data.spatial_dims[1]
                    field_specs_ptr[ii].spatial_dim_z = field_data.spatial_dims[2]

                error_specs_ptr[ii].field_spec_ptr = &field_specs_ptr[ii]

    cdef size_t n_sensors = sensor_in.num_sensors
    cdef size_t n_comps = mesh_in.num_components
    cdef size_t n_out_times = sensor_in.num_sample_times if sensor_in.num_sample_times > 0 else mesh_in.num_sim_times

    if num_experiments == 0:
        raise ValueError("num_experiments must be positive")
    result_shape = (n_sensors, n_comps, n_out_times)
    if num_experiments > 1:
        result_shape = (num_experiments,) + result_shape
    cdef cnp.ndarray truth = np.zeros(result_shape, dtype=np.float64)
    cdef cnp.ndarray meas = np.zeros(result_shape, dtype=np.float64)
    cdef cnp.ndarray errs_sys = np.zeros(result_shape, dtype=np.float64)
    cdef cnp.ndarray errs_rand = np.zeros(result_shape, dtype=np.float64)
    cdef cnp.ndarray errs_total = np.zeros(result_shape, dtype=np.float64)
    cdef cnp.ndarray pert_positions = np.empty(
        (num_experiments, n_sensors, 3), dtype=np.float64
    )
    cdef cnp.ndarray pert_times = np.empty(
        (num_experiments, n_out_times), dtype=np.float64
    )

    try:
        if num_experiments == 1:
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
        else:
            status = cf.felixSimulatePointSensorExperiments(
                &mesh_in,
                &sensor_in,
                error_specs_ptr,
                num_errors,
                num_experiments,
                experiment_seed,
                <double *>truth.data,
                <double *>meas.data,
                <double *>errs_sys.data,
                <double *>errs_rand.data,
                <double *>errs_total.data,
                <double *>pert_positions.data,
                <double *>pert_times.data,
            )
        if status != 0:
            err_msg = get_last_error()
            raise RuntimeError(f"Felix simulation failed: {err_msg}")
    finally:
        if error_specs_ptr != NULL:
            free(error_specs_ptr)
        if field_specs_ptr != NULL:
            free(field_specs_ptr)

    if num_experiments > 1:
        return (
            truth,
            meas,
            errs_sys,
            errs_rand,
            errs_total,
            pert_positions,
            pert_times,
        )
    return truth, meas, errs_sys, errs_rand, errs_total


def sample_field_config_graph(
    object field,
    cnp.ndarray positions,
    object graph_dict,
    object sample_times=None,
    object angles=None,
    size_t num_experiments=1,
    uint64_t experiment_seed=0,
):
    """Marshal a Felix field configuration with an Error Graph and execute in Zig."""
    sim_data = field.get_sim_data()
    connect = sim_data.connect
    if connect is None:
        raise ValueError("Felix field sampling requires mesh connectivity")
    if isinstance(connect, dict):
        connect_arr = np.vstack(tuple(connect.values()))
    else:
        connect_arr = connect

    spatial_dims = field._spatial_dims
    is_2d = getattr(spatial_dims, "value", spatial_dims) == 2
    num_nodes_per_elem = connect_arr.shape[1]
    if is_2d:
        if num_nodes_per_elem in (8, 9):
            connect_arr = connect_arr[:, :8]
            elem_type = 5
        elif num_nodes_per_elem == 6:
            connect_arr = connect_arr[:, :3]
            elem_type = 0
        elif num_nodes_per_elem == 4:
            elem_type = 1
        elif num_nodes_per_elem == 3:
            elem_type = 0
        else:
            raise ValueError("Unsupported 2D element node count")
    else:
        if num_nodes_per_elem in (20, 27):
            connect_arr = connect_arr[:, :20]
            elem_type = 6
        elif num_nodes_per_elem == 10:
            connect_arr = connect_arr[:, :4]
            elem_type = 2
        elif num_nodes_per_elem == 8:
            elem_type = 3
        elif num_nodes_per_elem == 4:
            elem_type = 2
        else:
            raise ValueError("Unsupported 3D element node count")

    comp_keys = field.get_all_components()
    sim_times = (
        sim_data.time
        if sim_data.time is not None
        else np.array([0.0], dtype=np.float64)
    )
    field_data = np.empty(
        (sim_data.coords.shape[0], len(comp_keys), sim_times.shape[0]),
        dtype=np.float64,
    )
    for cc, comp_key in enumerate(comp_keys):
        field_data[:, cc, :] = sim_data.node_vars[comp_key]

    rot_matrices = None
    if angles is not None:
        rot_matrices = np.array(
            [rotation.as_matrix().T for rotation in angles],
            dtype=np.float64,
        )

    is_tensor = hasattr(field, "_norm_comp_keys")

    # First get nominal truth
    truth_res = simulate_point_sensors(
        coords=sim_data.coords,
        connect=connect_arr,
        elem_type=elem_type,
        nodal_fields=field_data,
        sim_times=sim_times,
        positions=positions,
        sample_times=sample_times,
        rot_matrices=rot_matrices,
        spatial_dims=2 if is_2d else 3,
        is_tensor=is_tensor,
        error_specs_list=None,
        num_experiments=1,
    )
    truth = truth_res[0]

    return simulate_err_graph(
        coords=sim_data.coords,
        connect=connect_arr,
        elem_type=elem_type,
        nodal_fields=field_data,
        sim_times=sim_times,
        positions=positions,
        sample_times=sample_times,
        rot_matrices=rot_matrices,
        spatial_dims=2 if is_2d else 3,
        is_tensor=is_tensor,
        graph_dict=graph_dict,
        truth=truth,
        num_experiments=num_experiments,
        experiment_seed=experiment_seed,
    )


def simulate_err_graph(
    cnp.ndarray coords,
    cnp.ndarray connect,
    uint32_t elem_type,
    cnp.ndarray nodal_fields,
    cnp.ndarray sim_times,
    cnp.ndarray positions,
    object sample_times=None,
    object rot_matrices=None,
    uint32_t spatial_dims=3,
    bint is_tensor=False,
    object graph_dict=None,
    cnp.ndarray truth=None,
    size_t num_experiments=1,
    uint64_t experiment_seed=0,
):
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

    cdef cf.SimMeshInput mesh_in
    mesh_in.coords_ptr = <const double *>coords_c.data
    mesh_in.num_nodes = coords_c.shape[0]
    mesh_in.connect_ptr = <const size_t *>connect_c.data
    mesh_in.num_elements = connect_c.shape[0]
    mesh_in.elem_type = elem_type
    mesh_in.nodal_fields_ptr = <const double *>fields_c.data
    mesh_in.num_components = fields_c.shape[1]
    mesh_in.sim_times_ptr = <const double *>sim_times_c.data
    mesh_in.num_sim_times = sim_times_c.shape[0]

    cdef cnp.ndarray[double, ndim=2, mode="c"] pos_c = np.ascontiguousarray(
        positions, dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=1, mode="c"] st_c
    cdef cnp.ndarray[double, ndim=3, mode="c"] rot_c
    cdef cnp.ndarray[double, ndim=2, mode="c"] work_pos = np.array(
        pos_c, dtype=np.float64, copy=True, order="C"
    )
    cdef cnp.ndarray[double, ndim=2, mode="c"] scratch_pos = np.array(
        pos_c, dtype=np.float64, copy=True, order="C"
    )

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

    cdef cnp.ndarray[double, ndim=1, mode="c"] work_times
    if sample_times is not None:
        work_times = np.array(st_c, dtype=np.float64, copy=True, order="C")
    else:
        work_times = np.array(
            sim_times_c, dtype=np.float64, copy=True, order="C"
        )
    cdef cnp.ndarray[double, ndim=1, mode="c"] scratch_times = np.array(
        work_times, dtype=np.float64, copy=True, order="C"
    )

    if rot_matrices is not None:
        rot_c = np.ascontiguousarray(rot_matrices, dtype=np.float64)
        sensor_in.rot_matrices_ptr = <const double *>rot_c.data
        sensor_in.num_rot_matrices = rot_c.shape[0]
    else:
        sensor_in.rot_matrices_ptr = NULL
        sensor_in.num_rot_matrices = 0

    cdef cnp.ndarray[double, ndim=3, mode="c"] work_rots
    if rot_matrices is not None:
        work_rots = np.array(rot_c, dtype=np.float64, copy=True, order="C")
    else:
        work_rots = np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :],
            pos_c.shape[0],
            axis=0,
        )
    cdef cnp.ndarray[double, ndim=3, mode="c"] scratch_rots = np.array(
        work_rots, dtype=np.float64, copy=True, order="C"
    )

    sensor_in.spatial_dims = <uint32_t>spatial_dims
    sensor_in.is_tensor = 1 if is_tensor else 0
    sensor_in.work_positions_ptr = <double *>work_pos.data
    sensor_in.work_times_ptr = <double *>work_times.data
    sensor_in.work_rot_matrices_ptr = <double *>work_rots.data
    sensor_in.scratch_positions_ptr = <double *>scratch_pos.data
    sensor_in.scratch_times_ptr = <double *>scratch_times.data
    sensor_in.scratch_rot_matrices_ptr = <double *>scratch_rots.data

    cdef size_t n_sensors = pos_c.shape[0]
    cdef size_t n_comps = fields_c.shape[1]
    cdef size_t n_out_times = (
        sensor_in.num_sample_times
        if sensor_in.num_sample_times > 0
        else sim_times_c.shape[0]
    )

    cdef cnp.ndarray[double, ndim=3, mode="c"] truth_c = np.ascontiguousarray(
        truth, dtype=np.float64
    )

    # Marshal ErrGraphSpec
    cdef list node_list = graph_dict.get("nodes", [])
    cdef size_t num_nodes = len(node_list)
    cdef list exec_order = graph_dict.get("execution_order", list(range(num_nodes)))
    cdef list leaf_indices = graph_dict.get("leaf_indices", [])
    cdef bint store_node_outputs = graph_dict.get("store_node_outputs", False)

    cdef cf.ErrGraphSpec graph_spec
    graph_spec.num_nodes = num_nodes
    graph_spec.num_leaves = len(leaf_indices)
    graph_spec.store_node_outputs = 1 if store_node_outputs else 0

    cdef cf.ErrGraphNodeSpec *nodes_ptr = NULL
    cdef size_t *exec_order_ptr = NULL
    cdef size_t *leaf_indices_ptr = NULL
    cdef list temp_input_indices = []
    cdef cnp.ndarray[size_t, ndim=1, mode="c"] inp_c
    cdef cnp.ndarray[size_t, ndim=1, mode="c"] exec_c = np.array(exec_order, dtype=np.uint64)
    cdef cnp.ndarray[size_t, ndim=1, mode="c"] leaf_c = np.array(leaf_indices, dtype=np.uint64)

    exec_order_ptr = <size_t *>exec_c.data
    leaf_indices_ptr = <size_t *>leaf_c.data
    graph_spec.execution_order_ptr = exec_order_ptr
    graph_spec.leaf_indices_ptr = leaf_indices_ptr

    if num_nodes > 0:
        nodes_ptr = <cf.ErrGraphNodeSpec *>malloc(num_nodes * sizeof(cf.ErrGraphNodeSpec))
        for ii, n_dict in enumerate(node_list):
            nodes_ptr[ii].op = <uint32_t>n_dict.get("op", 0)
            inps = n_dict.get("inputs", [])
            nodes_ptr[ii].num_inputs = len(inps)
            if len(inps) > 0:
                inp_c = np.array(inps, dtype=np.uint64)
                temp_input_indices.append(inp_c)
                nodes_ptr[ii].input_indices_ptr = <const size_t *>inp_c.data
            else:
                nodes_ptr[ii].input_indices_ptr = NULL

            spec_dict = n_dict.get("spec", {})
            nodes_ptr[ii].error_spec.kind = <uint32_t>spec_dict.get("kind", 0)
            nodes_ptr[ii].error_spec.err_type = <uint32_t>spec_dict.get("err_type", 0)
            nodes_ptr[ii].error_spec.err_dep = <uint32_t>spec_dict.get("err_dep", 0)
            nodes_ptr[ii].error_spec.dist_type = <uint32_t>spec_dict.get("dist_type", 0)
            nodes_ptr[ii].error_spec.param0 = <double>spec_dict.get("param0", 0.0)
            nodes_ptr[ii].error_spec.param1 = <double>spec_dict.get("param1", 0.0)
            nodes_ptr[ii].error_spec.param2 = <double>spec_dict.get("param2", 0.0)
            nodes_ptr[ii].error_spec.seed = <uint64_t>(spec_dict.get("seed") or 0)
            nodes_ptr[ii].error_spec.has_seed = 1 if "seed" in spec_dict and spec_dict["seed"] is not None else 0
            nodes_ptr[ii].error_spec.table_ptr = NULL
            nodes_ptr[ii].error_spec.table_rows = 0
            nodes_ptr[ii].error_spec.poly_coeffs_ptr = NULL
            nodes_ptr[ii].error_spec.poly_coeffs_len = 0
            nodes_ptr[ii].error_spec.field_spec_ptr = NULL

    graph_spec.nodes_ptr = nodes_ptr

    cdef cnp.ndarray meas = np.empty((n_sensors, n_comps, n_out_times), dtype=np.float64)
    cdef cnp.ndarray errs_sys = np.empty((n_sensors, n_comps, n_out_times), dtype=np.float64)
    cdef cnp.ndarray errs_rand = np.empty((n_sensors, n_comps, n_out_times), dtype=np.float64)
    cdef cnp.ndarray errs_total = np.empty((n_sensors, n_comps, n_out_times), dtype=np.float64)
    cdef cnp.ndarray node_outputs = None
    if store_node_outputs and num_nodes > 0:
        node_outputs = np.empty((num_nodes, n_sensors, n_comps, n_out_times), dtype=np.float64)

    cdef cnp.ndarray pert_positions = np.empty(
        (num_experiments, n_sensors, 3), dtype=np.float64
    )
    cdef cnp.ndarray pert_times = np.empty(
        (num_experiments, n_out_times), dtype=np.float64
    )

    try:
        if num_experiments == 1:
            status = cf.felixSimulateErrGraph(
                &mesh_in,
                &sensor_in,
                &graph_spec,
                <const double *>truth_c.data,
                <double *>meas.data,
                <double *>errs_sys.data,
                <double *>errs_rand.data,
                <double *>errs_total.data,
                <double *>node_outputs.data if node_outputs is not None else NULL,
            )
        else:
            status = cf.felixSimulateErrGraphExperiments(
                &mesh_in,
                &sensor_in,
                &graph_spec,
                num_experiments,
                experiment_seed,
                <const double *>truth_c.data,
                <double *>meas.data,
                <double *>errs_sys.data,
                <double *>errs_rand.data,
                <double *>errs_total.data,
                <double *>node_outputs.data if node_outputs is not None else NULL,
                <double *>pert_positions.data,
                <double *>pert_times.data,
            )
        if status != 0:
            err_msg = get_last_error()
            raise RuntimeError(f"Felix graph simulation failed: {err_msg}")
    finally:
        if nodes_ptr != NULL:
            free(nodes_ptr)

    if num_experiments > 1:
        return (
            truth_c,
            meas,
            errs_sys,
            errs_rand,
            errs_total,
            pert_positions,
            pert_times,
            node_outputs,
        )
    return truth_c, meas, errs_sys, errs_rand, errs_total, node_outputs


def eval_kernel_weights(
    uint32_t kernel_type,
    cnp.ndarray coords,
    double param0=0.0,
    double param1=0.0,
    double param2=0.0,
):
    cdef cnp.ndarray[double, ndim=2, mode="c"] coords_c = np.ascontiguousarray(
        coords, dtype=np.float64
    )
    cdef size_t num_points = coords_c.shape[0]
    cdef size_t dims = coords_c.shape[1]
    cdef cnp.ndarray[double, ndim=1, mode="c"] out_weights = np.empty(
        num_points, dtype=np.float64
    )

    cdef cf.KernelSpec spec
    spec.kernel_type = kernel_type
    spec.param0 = param0
    spec.param1 = param1
    spec.param2 = param2

    cdef int status = cf.felixEvalWeightsBatch(
        &spec,
        <const double *>coords_c.data,
        num_points,
        dims,
        <double *>out_weights.data,
    )
    if status != 0:
        err_msg = get_last_error()
        raise RuntimeError(f"Felix kernel evaluation failed: {err_msg}")
    return out_weights


def generate_quadrature_rule(
    uint32_t rule,
    size_t order=2,
    size_t dims=1,
):
    cdef cf.QuadSpec spec
    spec.rule = rule
    spec.order = order
    spec.dims = dims

    # Upper bound max nodes for dims 1..3
    cdef size_t max_pts = (order + 2) ** dims
    cdef cnp.ndarray[double, ndim=2, mode="c"] nodes = np.empty(
        (max_pts, dims), dtype=np.float64
    )
    cdef cnp.ndarray[double, ndim=1, mode="c"] weights = np.empty(
        max_pts, dtype=np.float64
    )
    cdef size_t count = 0

    cdef int status = cf.felixGenerateQuadNodesAndWeights(
        &spec,
        <double *>nodes.data,
        <double *>weights.data,
        &count,
    )
    if status != 0:
        err_msg = get_last_error()
        raise RuntimeError(f"Felix quadrature generation failed: {err_msg}")
    return nodes[:count], weights[:count]


def transform_tensor_invariants(
    cnp.ndarray raw_tensor,
    uint32_t inv_type,
    int spatial_dims=2,
):
    cdef cnp.ndarray[double, ndim=3, mode="c"] tensor_c = np.ascontiguousarray(
        raw_tensor, dtype=np.float64
    )
    cdef size_t num_points = tensor_c.shape[0]
    cdef size_t num_comps = tensor_c.shape[1]
    cdef size_t num_times = tensor_c.shape[2]

    cdef cnp.ndarray[double, ndim=3, mode="c"] out_derived = np.empty(
        (num_points, 1, num_times), dtype=np.float64
    )

    cdef int status = 0
    if spatial_dims == 2 or num_comps == 3:
        status = cf.felixTransformTensorArray2D(
            <const double *>tensor_c.data,
            num_points,
            num_times,
            inv_type,
            <double *>out_derived.data,
        )
    else:
        status = cf.felixTransformTensorArray3D(
            <const double *>tensor_c.data,
            num_points,
            num_times,
            inv_type,
            <double *>out_derived.data,
        )

    if status != 0:
        err_msg = get_last_error()
        raise RuntimeError(f"Felix tensor transformation failed: {err_msg}")
    return out_derived

