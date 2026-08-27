// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("stats_common.zig");

const F = common.F;

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

        std.sort.pdq(F, work, {}, common.lessThan);
        const median_val = common.calcMedian(work);
        for (work) |*val| val.* = @abs(val.* - median_val);
        std.sort.pdq(F, work, {}, common.lessThan);

        out_mean[vv] = mean_val;
        out_std[vv] = @sqrt(variance);
        out_min[vv] = min_val;
        out_max[vv] = max_val;
        out_median[vv] = median_val;
        out_var[vv] = variance;
        out_mad[vv] = common.calcMedian(work);
    }
}
