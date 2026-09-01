// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const err_graph = @import("../felix/zig/err_graph.zig");
const sensor_sim = @import("../felix/zig/sensor_sim.zig");
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

const F = common.F;
const ErrGraphNodeSpec = err_graph.ErrGraphNodeSpec;
const ErrGraphSpec = err_graph.ErrGraphSpec;
const SimMeshInput = sensor_sim.SimMeshInput;
const SensorArrayInput = sensor_sim.SensorArrayInput;

// --------------------------------------------------------------------------------------
// Unit Tests: Error Graph DAG Execution
// --------------------------------------------------------------------------------------

test "Linear chain Error DAG execution" {
    // 1 sensor, 1 component, 1 time step
    const truth_val: [1]F = .{100.0};
    var meas_out: [1]F = undefined;
    var err_total_out: [1]F = undefined;

    const dummy_mesh = SimMeshInput{
        .coords_ptr = null,
        .num_nodes = 0,
        .connect_ptr = null,
        .num_elements = 0,
        .elem_type = 0,
        .nodal_fields_ptr = null,
        .num_components = 1,
        .sim_times_ptr = &[_]F{0.0},
        .num_sim_times = 1,
    };

    const dummy_sensor = SensorArrayInput{
        .positions_ptr = &[_]F{ 0.0, 0.0, 0.0 },
        .num_sensors = 1,
        .sample_times_ptr = null,
        .num_sample_times = 0,
        .rot_matrices_ptr = null,
        .num_rot_matrices = 0,
        .spatial_dims = 3,
        .is_tensor = 0,
        .work_positions_ptr = null,
        .work_times_ptr = null,
        .work_rot_matrices_ptr = null,
        .scratch_positions_ptr = null,
        .scratch_times_ptr = null,
        .scratch_rot_matrices_ptr = null,
    };

    // Node 0: Offset +10.0 (kind = 0, param0 = 10.0) -> Output = 110.0
    const node0 = ErrGraphNodeSpec{
        .op = 0, // add
        .num_inputs = 0,
        .input_indices_ptr = null,
        .error_spec = .{
            .kind = 0,
            .err_type = 0,
            .err_dep = 0,
            .dist_type = 0,
            .param0 = 10.0,
            .param1 = 0.0,
            .param2 = 0.0,
            .seed = 0,
            .has_seed = 0,
            .table_ptr = null,
            .table_rows = 0,
            .poly_coeffs_ptr = null,
            .poly_coeffs_len = 0,
            .field_spec_ptr = null,
        },
    };

    const nodes = [_]ErrGraphNodeSpec{node0};
    const execution_order = [_]usize{0};
    const leaf_indices = [_]usize{0};

    const graph_spec = ErrGraphSpec{
        .num_nodes = 1,
        .nodes_ptr = &nodes,
        .execution_order_ptr = &execution_order,
        .num_leaves = 1,
        .leaf_indices_ptr = &leaf_indices,
        .store_node_outputs = 0,
    };

    err_graph.runErrGraphSimulation(
        &dummy_mesh,
        &dummy_sensor,
        &graph_spec,
        &truth_val,
        &meas_out,
        null,
        null,
        &err_total_out,
        null,
        0,
    );

    try std.testing.expect(
        common.isApproxEqual(meas_out[0], 110.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(err_total_out[0], 10.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}
