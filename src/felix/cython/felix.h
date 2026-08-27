// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
#ifndef FELIX_H
#define FELIX_H

/*
 * Felix: A High Performance Sensor Simulation Core
 *
 * Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
 * Licensed under the MIT License (see LICENSE file for details)
 *
 * Public Felix C ABI
 * Fixed to f64 precision for ABI stability.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SimMeshInput {
    const double *coords_ptr;
    size_t        num_nodes;
    const size_t *connect_ptr;
    size_t        num_elements;
    uint32_t      elem_type;
    const double *nodal_fields_ptr;
    size_t        num_components;
    const double *sim_times_ptr;
    size_t        num_sim_times;
} SimMeshInput;

typedef struct SensorArrayInput {
    const double *positions_ptr;
    size_t        num_sensors;
    const double *sample_times_ptr;
    size_t        num_sample_times;
    const double *rot_matrices_ptr;
    size_t        num_rot_matrices;
    uint32_t      spatial_dims;
    uint32_t      is_tensor;
    double       *work_positions_ptr;
    double       *work_times_ptr;
    double       *work_rot_matrices_ptr;
    double       *scratch_positions_ptr;
    double       *scratch_times_ptr;
    double       *scratch_rot_matrices_ptr;
} SensorArrayInput;

typedef struct DistributionSpec {
    uint32_t dist_type;
    double   param0;
    double   param1;
    double   param2;
    uint64_t seed;
    uint32_t has_seed;
} DistributionSpec;

typedef struct FieldPerturbationSpec {
    const double    *pos_offset_ptr;
    const uint8_t   *pos_lock_ptr;
    const double    *angle_offset_ptr;
    const uint8_t   *angle_lock_ptr;
    const double    *time_offset_ptr;
    DistributionSpec pos_rand[3];
    DistributionSpec angle_rand[3];
    DistributionSpec time_rand;
    uint32_t          drift_kind;
    double            drift_param0;
    double            drift_param1;
    double            drift_param2;
    const double     *drift_poly_ptr;
    size_t            drift_poly_len;
    uint32_t          spatial_kind;
    double            spatial_dim_x;
    double            spatial_dim_y;
    double            spatial_dim_z;
} FieldPerturbationSpec;

typedef struct ErrorSpec {
    uint32_t      kind;
    uint32_t      err_type;
    uint32_t      err_dep;
    uint32_t      dist_type;
    double        param0;
    double        param1;
    double        param2;
    uint64_t      seed;
    uint32_t      has_seed;
    const double *table_ptr;
    size_t        table_rows;
    const double *poly_coeffs_ptr;
    size_t        poly_coeffs_len;
    const FieldPerturbationSpec *field_spec_ptr;
} ErrorSpec;

typedef struct ErrGraphNodeSpec {
    uint32_t      op;
    size_t        num_inputs;
    const size_t *input_indices_ptr;
    ErrorSpec     error_spec;
} ErrGraphNodeSpec;

typedef struct ErrGraphSpec {
    size_t                  num_nodes;
    const ErrGraphNodeSpec *nodes_ptr;
    const size_t           *execution_order_ptr;
    size_t                  num_leaves;
    const size_t           *leaf_indices_ptr;
    uint32_t                store_node_outputs;
} ErrGraphSpec;

size_t felixGetLastError(uint8_t *out_buf, size_t out_buf_len);

int felixSimulatePointSensors(
    const SimMeshInput      *mesh_in_ptr,
    const SensorArrayInput  *sensor_in_ptr,
    const ErrorSpec         *error_specs_ptr,
    size_t                   num_errors,
    double                  *out_truth_ptr,
    double                  *out_measurements_ptr,
    double                  *out_errs_sys_ptr,
    double                  *out_errs_rand_ptr,
    double                  *out_errs_total_ptr
);

int felixSimulatePointSensorExperiments(
    const SimMeshInput      *mesh_in_ptr,
    const SensorArrayInput  *sensor_in_ptr,
    const ErrorSpec         *error_specs_ptr,
    size_t                   num_errors,
    size_t                   num_experiments,
    uint64_t                 seed,
    double                  *out_truth_ptr,
    double                  *out_measurements_ptr,
    double                  *out_errs_sys_ptr,
    double                  *out_errs_rand_ptr,
    double                  *out_errs_total_ptr,
    double                  *out_pert_positions_ptr,
    double                  *out_pert_times_ptr
);

int felixSimulateErrGraph(
    const SimMeshInput      *mesh_in_ptr,
    const SensorArrayInput  *sensor_in_ptr,
    const ErrGraphSpec      *graph_spec_ptr,
    const double            *truth_values_ptr,
    double                  *out_measurements_ptr,
    double                  *out_errs_sys_ptr,
    double                  *out_errs_rand_ptr,
    double                  *out_errs_total_ptr,
    double                  *out_node_outputs_ptr
);

int felixSimulateErrGraphExperiments(
    const SimMeshInput      *mesh_in_ptr,
    const SensorArrayInput  *sensor_in_ptr,
    const ErrGraphSpec      *graph_spec_ptr,
    size_t                   num_experiments,
    uint64_t                 seed,
    const double            *truth_values_ptr,
    double                  *out_measurements_ptr,
    double                  *out_errs_sys_ptr,
    double                  *out_errs_rand_ptr,
    double                  *out_errs_total_ptr,
    double                  *out_node_outputs_ptr,
    double                  *out_pert_positions_ptr,
    double                  *out_pert_times_ptr
);

int felixCalcExperimentStats(
    const double *values_ptr,
    size_t        num_experiments,
    size_t        num_values,
    double       *out_mean_ptr,
    double       *out_std_ptr,
    double       *out_min_ptr,
    double       *out_max_ptr,
    double       *out_median_ptr,
    double       *out_var_ptr,
    double       *out_mad_ptr
);

void felixTransformVectors2D(
    const double *rot_mat_22_ptr,
    const double *vx_in_ptr,
    const double *vy_in_ptr,
    size_t        num_vectors,
    double       *out_vx_ptr,
    double       *out_vy_ptr
);

void felixTransformVectors3D(
    const double *rot_mat_33_ptr,
    const double *vx_in_ptr,
    const double *vy_in_ptr,
    const double *vz_in_ptr,
    size_t        num_vectors,
    double       *out_vx_ptr,
    double       *out_vy_ptr,
    double       *out_vz_ptr
);

void felixPrintSensorData(
    const double *positions_ptr,
    size_t        positions_len,
    const double *sample_times_ptr,
    size_t        sample_times_len,
    const double *spatial_dims_ptr,
    size_t        spatial_dims_len
);

typedef struct KernelSpec {
    uint32_t kernel_type;
    double   param0;
    double   param1;
    double   param2;
} KernelSpec;

typedef struct QuadSpec {
    uint32_t rule;
    size_t   order;
    size_t   dims;
} QuadSpec;

int felixEvalWeightsBatch(
    const KernelSpec *spec_ptr,
    const double     *coords_ptr,
    size_t            num_points,
    size_t            dims,
    double           *out_weights_ptr
);

int felixGenerateQuadNodesAndWeights(
    const QuadSpec *spec_ptr,
    double         *out_nodes_ptr,
    double         *out_weights_ptr,
    size_t         *out_count_ptr
);

int felixTransformTensorArray2D(
    const double *raw_tensor_ptr,
    size_t        num_points,
    size_t        num_times,
    uint32_t      inv_type,
    double       *out_derived_ptr
);

int felixTransformTensorArray3D(
    const double *raw_tensor_ptr,
    size_t        num_points,
    size_t        num_times,
    uint32_t      inv_type,
    double       *out_derived_ptr
);

#ifdef __cplusplus
}
#endif

#endif /* FELIX_H */
