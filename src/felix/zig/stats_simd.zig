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
const scalar = @import("stats_scalar.zig");

const F = common.F;
const VecSF = common.VecSF;
const SimdWidth = common.SimdWidth;

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

    const n_exp_f = @as(F, @floatFromInt(num_experiments));
    const n_exp_vec: VecSF = @splat(n_exp_f);

    var vv_offset: usize = 0;

    while (vv_offset + SimdWidth <= num_values) : (vv_offset += SimdWidth) {
        var sum_vec: VecSF = @splat(0.0);
        const init_vals: VecSF = values[vv_offset..][0..SimdWidth].*;
        var min_vec: VecSF = init_vals;
        var max_vec: VecSF = init_vals;

        for (0..num_experiments) |ee| {
            const row_offset = ee * num_values + vv_offset;
            const val_vec: VecSF = values[row_offset..][0..SimdWidth].*;
            sum_vec += val_vec;
            min_vec = @min(min_vec, val_vec);
            max_vec = @max(max_vec, val_vec);
        }

        const mean_vec = sum_vec / n_exp_vec;
        var var_sum_vec: VecSF = @splat(0.0);

        for (0..num_experiments) |ee| {
            const row_offset = ee * num_values + vv_offset;
            const val_vec: VecSF = values[row_offset..][0..SimdWidth].*;
            const delta_vec = val_vec - mean_vec;
            var_sum_vec += delta_vec * delta_vec;
        }

        const variance_vec = var_sum_vec / n_exp_vec;
        const std_vec = @sqrt(variance_vec);

        out_mean[vv_offset..][0..SimdWidth].* = mean_vec;
        out_std[vv_offset..][0..SimdWidth].* = std_vec;
        out_min[vv_offset..][0..SimdWidth].* = min_vec;
        out_max[vv_offset..][0..SimdWidth].* = max_vec;
        out_var[vv_offset..][0..SimdWidth].* = variance_vec;

        for (0..SimdWidth) |lane_idx| {
            const curr_v = vv_offset + lane_idx;
            for (0..num_experiments) |ee| {
                work[ee] = values[ee * num_values + curr_v];
            }
            std.sort.pdq(F, work, {}, common.lessThan);
            const median_val = common.calcMedian(work);
            for (work) |*val| val.* = @abs(val.* - median_val);
            std.sort.pdq(F, work, {}, common.lessThan);

            out_median[curr_v] = median_val;
            out_mad[curr_v] = common.calcMedian(work);
        }
    }

    while (vv_offset < num_values) : (vv_offset += 1) {
        var sum: F = 0.0;
        var min_val = values[vv_offset];
        var max_val = values[vv_offset];

        for (0..num_experiments) |ee| {
            const val = values[ee * num_values + vv_offset];
            work[ee] = val;
            sum += val;
            min_val = @min(min_val, val);
            max_val = @max(max_val, val);
        }

        const mean_val = sum / n_exp_f;
        var variance_sum: F = 0.0;
        for (work) |val| {
            const delta = val - mean_val;
            variance_sum += delta * delta;
        }
        const variance = variance_sum / n_exp_f;

        std.sort.pdq(F, work, {}, common.lessThan);
        const median_val = common.calcMedian(work);
        for (work) |*val| val.* = @abs(val.* - median_val);
        std.sort.pdq(F, work, {}, common.lessThan);

        out_mean[vv_offset] = mean_val;
        out_std[vv_offset] = @sqrt(variance);
        out_min[vv_offset] = min_val;
        out_max[vv_offset] = max_val;
        out_median[vv_offset] = median_val;
        out_var[vv_offset] = variance;
        out_mad[vv_offset] = common.calcMedian(work);
    }
}

// --------------------------------------------------------------------------------------
// Tests: Parity Verification
// --------------------------------------------------------------------------------------

test "stats SIMD vs scalar parity" {
    const num_experiments = 4;
    const num_values = 16;
    var values: [num_experiments * num_values]F = undefined;

    for (0..num_experiments) |ee| {
        for (0..num_values) |vv| {
            values[ee * num_values + vv] =
                @as(F, @floatFromInt(ee * 10 + vv)) * 0.25;
        }
    }

    var scal_mean: [num_values]F = undefined;
    var scal_std: [num_values]F = undefined;
    var scal_min: [num_values]F = undefined;
    var scal_max: [num_values]F = undefined;
    var scal_med: [num_values]F = undefined;
    var scal_var: [num_values]F = undefined;
    var scal_mad: [num_values]F = undefined;

    try scalar.calcExperimentStats(
        std.testing.allocator,
        &values,
        num_experiments,
        num_values,
        &scal_mean,
        &scal_std,
        &scal_min,
        &scal_max,
        &scal_med,
        &scal_var,
        &scal_mad,
    );

    var simd_mean: [num_values]F = undefined;
    var simd_std: [num_values]F = undefined;
    var simd_min: [num_values]F = undefined;
    var simd_max: [num_values]F = undefined;
    var simd_med: [num_values]F = undefined;
    var simd_var: [num_values]F = undefined;
    var simd_mad: [num_values]F = undefined;

    try calcExperimentStats(
        std.testing.allocator,
        &values,
        num_experiments,
        num_values,
        &simd_mean,
        &simd_std,
        &simd_min,
        &simd_max,
        &simd_med,
        &simd_var,
        &simd_mad,
    );

    for (0..num_values) |vv| {
        try std.testing.expectApproxEqAbs(scal_mean[vv], simd_mean[vv], 1e-12);
        try std.testing.expectApproxEqAbs(scal_std[vv], simd_std[vv], 1e-12);
        try std.testing.expectEqual(scal_min[vv], simd_min[vv]);
        try std.testing.expectEqual(scal_max[vv], simd_max[vv]);
        try std.testing.expectEqual(scal_med[vv], simd_med[vv]);
        try std.testing.expectApproxEqAbs(scal_var[vv], simd_var[vv], 1e-12);
        try std.testing.expectEqual(scal_mad[vv], simd_mad[vv]);
    }
}
