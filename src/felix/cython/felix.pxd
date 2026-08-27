# --------------------------------------------------------------------------
# Felix: A High Performance Sensor Simulation Core
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from "felix.h":

    ctypedef struct SimMeshInput:
        const double *coords_ptr
        size_t        num_nodes
        const size_t *connect_ptr
        size_t        num_elements
        uint32_t      elem_type
        const double *nodal_fields_ptr
        size_t        num_components
        const double *sim_times_ptr
        size_t        num_sim_times

    ctypedef struct SensorArrayInput:
        const double *positions_ptr
        size_t        num_sensors
        const double *sample_times_ptr
        size_t        num_sample_times
        const double *rot_matrices_ptr
        size_t        num_rot_matrices
        uint32_t      spatial_dims
        uint32_t      is_tensor

    ctypedef struct ErrorSpec:
        uint32_t      kind
        uint32_t      err_type
        uint32_t      err_dep
        uint32_t      dist_type
        double        param0
        double        param1
        double        param2
        uint64_t      seed
        uint32_t      has_seed
        const double *table_ptr
        size_t        table_rows
        const double *poly_coeffs_ptr
        size_t        poly_coeffs_len

    size_t felixGetLastError(
        uint8_t *out_buf,
        size_t out_buf_len,
    )

    int felixSimulatePointSensors(
        const SimMeshInput      *mesh_in_ptr,
        const SensorArrayInput  *sensor_in_ptr,
        const ErrorSpec         *error_specs_ptr,
        size_t                   num_errors,
        double                  *out_truth_ptr,
        double                  *out_measurements_ptr,
        double                  *out_errs_sys_ptr,
        double                  *out_errs_rand_ptr,
        double                  *out_errs_total_ptr,
    )

    void felixTransformVectors2D(
        const double *rot_mat_22_ptr,
        const double *vx_in_ptr,
        const double *vy_in_ptr,
        size_t        num_vectors,
        double       *out_vx_ptr,
        double       *out_vy_ptr,
    )

    void felixTransformVectors3D(
        const double *rot_mat_33_ptr,
        const double *vx_in_ptr,
        const double *vy_in_ptr,
        const double *vz_in_ptr,
        size_t        num_vectors,
        double       *out_vx_ptr,
        double       *out_vy_ptr,
        double       *out_vz_ptr,
    )

    void felixPrintSensorData(
        const double *positions_ptr,
        size_t        positions_len,
        const double *sample_times_ptr,
        size_t        sample_times_len,
        const double *spatial_dims_ptr,
        size_t        spatial_dims_len,
    )
