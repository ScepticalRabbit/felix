// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../felix/zig/buildconfig.zig");
const csvio = @import("../felix/zig/csvio.zig");
const NDArray = @import("../felix/zig/ndarray.zig").NDArray;
const tcfg = @import("testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const F = buildconfig.F;
pub const default_rel_tol: F = tcfg.REL_TOL;
pub const default_abs_tol: F = tcfg.ABS_TOL;

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn isApproxEqual(v1: F, v2: F, rel_tol: F, abs_tol: F) bool {
    if (v1 == v2) return true;
    if (std.math.isNan(v1) and std.math.isNan(v2)) return true;
    if (std.math.isNan(v1) or std.math.isNan(v2)) return false;

    const diff: F = @abs(v1 - v2);
    if (diff <= abs_tol) return true;

    const abs_v1: F = @abs(v1);
    const abs_v2: F = @abs(v2);
    const largest: F = if (abs_v1 > abs_v2) abs_v1 else abs_v2;

    return (diff / largest) <= rel_tol;
}

pub fn assertSlicesClose(
    expected: []const F,
    actual: []const F,
    rel_tol: F,
    abs_tol: F,
) !void {
    if (expected.len != actual.len) {
        std.debug.print(
            "Slice length mismatch: expected {}, got {}\n",
            .{ expected.len, actual.len },
        );
        return error.SliceLengthMismatch;
    }

    var max_diff: F = 0.0;
    var max_idx: usize = 0;
    var mismatch_found: bool = false;

    for (expected, actual, 0..) |exp_val, act_val, ii| {
        if (!isApproxEqual(exp_val, act_val, rel_tol, abs_tol)) {
            const diff: F = @abs(exp_val - act_val);
            if (diff > max_diff) {
                max_diff = diff;
                max_idx = ii;
            }
            mismatch_found = true;
        }
    }

    if (mismatch_found) {
        std.debug.print(
            "Slice mismatch at index {}: expected {d:.8e}, got {d:.8e}, max_diff={d:.8e}\n",
            .{ max_idx, expected[max_idx], actual[max_idx], max_diff },
        );
        return error.SliceValueMismatch;
    }
}

pub fn compareNDArrayToGold(
    allocator: std.mem.Allocator,
    io: std.Io,
    array: *const NDArray(F),
    gold_path: []const u8,
    rel_tol: F,
    abs_tol: F,
) !void {
    var gold_arr = try csvio.loadScalarCsv2D(allocator, io, gold_path);
    defer {
        allocator.free(gold_arr.slice);
        gold_arr.deinit(allocator);
    }

    if (gold_arr.slice.len != array.slice.len) {
        std.debug.print(
            "Array size mismatch vs gold {s}: expected {}, got {}\n",
            .{ gold_path, gold_arr.slice.len, array.slice.len },
        );
        return error.GoldArraySizeMismatch;
    }

    try assertSlicesClose(gold_arr.slice, array.slice, rel_tol, abs_tol);
}
