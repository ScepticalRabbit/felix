// --------------------------------------------------------------------------------------
// Felix: A High Performance Sensor Simulation Core
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const F: type = f64;

// --------------------------------------------------------------------------------------
// Public Entry Points
// --------------------------------------------------------------------------------------

pub fn calcExperimentStats(
    alloc: std.mem.Allocator,
    values: []const F,
    num_experiments: usize,
    num_values: usize,
    out_mean: []F,
    out_std: []F,
    out_min: []F,
    out_max: []F,
    out_median: []F,
    out_var: []F,
    out_mad: []F,
) !void {
    if (num_experiments == 0) return error.NoExperiments;
    if (values.len != num_experiments * num_values) return error.InvalidShape;

    const work = try alloc.alloc(F, num_experiments);
    defer alloc.free(work);

    for (0..num_values) |vv| {
        var sum: F = 0.0;
        var min_val = values[vv];
        var max_val = values[vv];

        for (0..num_experiments) |ee| {
            const val = values[ee * num_values + vv];
            work[ee] = val;
            sum += val;
            min_val = @min(min_val, val);
            max_val = @max(max_val, val);
        }

        const mean_val = sum / @as(F, @floatFromInt(num_experiments));
        var variance_sum: F = 0.0;
        for (work) |val| {
            const delta = val - mean_val;
            variance_sum += delta * delta;
        }
        const variance = variance_sum / @as(F, @floatFromInt(num_experiments));

        std.sort.pdq(F, work, {}, lessThan);
        const median_val = calcMedian(work);
        for (work) |*val| val.* = @abs(val.* - median_val);
        std.sort.pdq(F, work, {}, lessThan);

        out_mean[vv] = mean_val;
        out_std[vv] = @sqrt(variance);
        out_min[vv] = min_val;
        out_max[vv] = max_val;
        out_median[vv] = median_val;
        out_var[vv] = variance;
        out_mad[vv] = calcMedian(work);
    }
}

// --------------------------------------------------------------------------------------
// Private Helpers
// --------------------------------------------------------------------------------------

fn calcMedian(sorted: []const F) F {
    const mid = sorted.len / 2;
    if (sorted.len % 2 == 1) return sorted[mid];
    return (sorted[mid - 1] + sorted[mid]) / 2.0;
}

fn lessThan(_: void, lhs: F, rhs: F) bool {
    return lhs < rhs;
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "experiment statistics match population definitions" {
    const values = [_]F{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    };
    var mean: [2]F = undefined;
    var std_dev: [2]F = undefined;
    var min_val: [2]F = undefined;
    var max_val: [2]F = undefined;
    var median: [2]F = undefined;
    var variance: [2]F = undefined;
    var mad: [2]F = undefined;

    try calcExperimentStats(
        std.testing.allocator,
        &values,
        3,
        2,
        &mean,
        &std_dev,
        &min_val,
        &max_val,
        &median,
        &variance,
        &mad,
    );

    try std.testing.expectApproxEqAbs(2.0, mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(2.0 / 3.0, variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@sqrt(2.0 / 3.0), std_dev[0], 1e-12);
    try std.testing.expectEqual(1.0, min_val[0]);
    try std.testing.expectEqual(3.0, max_val[0]);
    try std.testing.expectEqual(2.0, median[0]);
    try std.testing.expectEqual(1.0, mad[0]);
}
