// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("../felix/zig/sensor_sim_common.zig");
const exp_sim = @import("../felix/zig/experiment_sim.zig");

test "ExperimentSimulator parallel execution matches single-threaded" {
    const alloc = std.testing.allocator;

    var coords = [_]f64{
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
    };
    var connect = [_]usize{ 0, 1, 2, 3 };
    var times = [_]f64{ 0.0, 0.5, 1.0 };
    var nodal_vars = [_]f64{
        10.0, 12.0, 15.0,
        10.0, 12.0, 15.0,
        10.0, 12.0, 15.0,
        10.0, 12.0, 15.0,
    };

    const mesh_in = common.SimMeshInput{
        .coords_ptr = &coords,
        .num_nodes = 4,
        .connect_ptr = &connect,
        .num_elements = 1,
        .elem_type = 0,
        .nodal_fields_ptr = &nodal_vars,
        .num_components = 1,
        .sim_times_ptr = &times,
        .num_sim_times = 3,
    };

    var sensor_positions = [_]f64{
        0.25, 0.25, 0.0,
        0.75, 0.25, 0.0,
        0.75, 0.75, 0.0,
        0.25, 0.75, 0.0,
    };

    const sensor_in = common.SensorArrayInput{
        .positions_ptr = &sensor_positions,
        .num_sensors = 4,
        .sample_times_ptr = null,
        .num_sample_times = 0,
        .rot_matrices_ptr = null,
        .num_rot_matrices = 0,
        .spatial_dims = 2,
        .is_tensor = 0,
        .work_positions_ptr = null,
        .work_times_ptr = null,
        .work_rot_matrices_ptr = null,
        .scratch_positions_ptr = null,
        .scratch_times_ptr = null,
        .scratch_rot_matrices_ptr = null,
        .binding_ptr = null,
    };

    var err_specs = [_]common.ErrorSpec{
        .{
            .kind = 2,
            .err_type = 0,
            .err_dep = 0,
            .dist_type = 2,
            .param0 = 0.0,
            .param1 = 0.5,
            .param2 = 0.0,
            .seed = 42,
            .has_seed = 1,
            .table_ptr = null,
            .table_rows = 0,
            .poly_coeffs_ptr = null,
            .poly_coeffs_len = 0,
            .field_spec_ptr = null,
        },
    };

    const num_experiments: usize = 16;
    const exp_stride = 4 * 1 * 3;
    const total_values = num_experiments * exp_stride;

    const out_truth_seq = try alloc.alloc(f64, total_values);
    defer alloc.free(out_truth_seq);
    const out_meas_seq = try alloc.alloc(f64, total_values);
    defer alloc.free(out_meas_seq);

    const out_truth_par = try alloc.alloc(f64, total_values);
    defer alloc.free(out_truth_par);
    const out_meas_par = try alloc.alloc(f64, total_values);
    defer alloc.free(out_meas_par);

    try exp_sim.runExperimentSimulationParallel(
        alloc,
        &mesh_in,
        &sensor_in,
        &err_specs,
        err_specs.len,
        out_truth_seq.ptr,
        out_meas_seq.ptr,
        null,
        num_experiments,
        0,
        1000,
        1,
        1,
    );

    try exp_sim.runExperimentSimulationParallel(
        alloc,
        &mesh_in,
        &sensor_in,
        &err_specs,
        err_specs.len,
        out_truth_par.ptr,
        out_meas_par.ptr,
        null,
        num_experiments,
        0,
        1000,
        4,
        2,
    );

    for (0..total_values) |ii| {
        try std.testing.expectEqual(out_truth_seq[ii], out_truth_par[ii]);
        try std.testing.expectEqual(out_meas_seq[ii], out_meas_par[ii]);
    }
}
