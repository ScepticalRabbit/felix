// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const stats = @import("../felix/zig/stats.zig");
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

const F = common.F;

// --------------------------------------------------------------------------------------
// Unit Tests: Experiment Statistics
// --------------------------------------------------------------------------------------

test "Experiment statistical reduction" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        std.testing.expect(deinit_status == .ok) catch @panic("Leak in stats test!");
    }

    const num_experiments: usize = 4;
    const num_values: usize = 2;

    // 4 experiments, 2 values per experiment
    // Value 0: 10, 20, 30, 40 -> mean = 25, min = 10, max = 40, median = 25
    // Value 1: 5, 5, 5, 5 -> mean = 5, min = 5, max = 5, var = 0
    const values = [_]F{
        10.0, 5.0,
        20.0, 5.0,
        30.0, 5.0,
        40.0, 5.0,
    };

    var out_mean: [2]F = undefined;
    var out_std: [2]F = undefined;
    var out_min: [2]F = undefined;
    var out_max: [2]F = undefined;
    var out_median: [2]F = undefined;
    var out_var: [2]F = undefined;
    var out_mad: [2]F = undefined;

    try stats.calcExperimentStats(
        allocator,
        &values,
        num_experiments,
        num_values,
        &out_mean,
        &out_std,
        &out_min,
        &out_max,
        &out_median,
        &out_var,
        &out_mad,
    );

    try std.testing.expect(
        common.isApproxEqual(out_mean[0], 25.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_min[0], 10.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_max[0], 40.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_median[0], 25.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    try std.testing.expect(
        common.isApproxEqual(out_mean[1], 5.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_var[1], 0.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}
