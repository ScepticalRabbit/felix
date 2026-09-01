// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const sensor_sim = @import("../felix/zig/sensor_sim.zig");
const orch = @import("../dev_support/orchestration.zig");
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

const F = common.F;
const SensorArrayInput = sensor_sim.SensorArrayInput;

// --------------------------------------------------------------------------------------
// End-to-End Tests: Fusion Divertor Monoblock Sensor Simulation
// --------------------------------------------------------------------------------------

test "End-to-End Monoblock 3D HEX20 virtual thermocouple simulation" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        std.testing.expect(deinit_status == .ok) catch @panic("Leak in monoblock E2E test!");
    }

    const io = std.testing.io;

    const field_files = [_][]const u8{"field_temperature.csv"};
    var sim_data = try orch.loadCaseSimData(
        allocator,
        io,
        .monoblock_3d,
        &field_files,
        null,
    );
    defer sim_data.deinit(allocator);

    const num_sim_times: usize = if (sim_data.field) |f| f.getTimeN() else 1;
    const sim_times = try allocator.alloc(F, num_sim_times);
    defer allocator.free(sim_times);
    for (0..num_sim_times) |tt| {
        sim_times[tt] = @as(F, @floatFromInt(tt)) * 1.0;
    }

    const mesh_in = orch.buildSimMeshInput(&sim_data, .hex20, sim_times);

    // Place sensor inside the monoblock body
    const sens_pos = [_]F{ 0.0, 0.015, 0.006 };
    const num_sensors: usize = 1;

    var work_pos = sens_pos;
    var scratch_pos = sens_pos;

    const sensor_in = SensorArrayInput{
        .positions_ptr = &sens_pos,
        .num_sensors = num_sensors,
        .sample_times_ptr = null,
        .num_sample_times = 0,
        .rot_matrices_ptr = null,
        .num_rot_matrices = 0,
        .spatial_dims = 3,
        .is_tensor = 0,
        .work_positions_ptr = &work_pos,
        .work_times_ptr = null,
        .work_rot_matrices_ptr = null,
        .scratch_positions_ptr = &scratch_pos,
        .scratch_times_ptr = null,
        .scratch_rot_matrices_ptr = null,
    };

    const out_truth = try allocator.alloc(F, num_sensors * 1 * num_sim_times);
    defer allocator.free(out_truth);

    const out_meas = try allocator.alloc(F, num_sensors * 1 * num_sim_times);
    defer allocator.free(out_meas);

    sensor_sim.runSensorSimulation(
        &mesh_in,
        &sensor_in,
        null,
        0,
        out_truth.ptr,
        out_meas.ptr,
        null,
        null,
        null,
        0,
    );

    // Assert temperature results are physically positive and non-zero
    for (0..out_truth.len) |ii| {
        try std.testing.expect(out_truth[ii] > 0.0);
        try std.testing.expect(
            common.isApproxEqual(out_truth[ii], out_meas[ii], tcfg.REL_TOL, tcfg.ABS_TOL),
        );
    }
}
