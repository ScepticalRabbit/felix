// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const pce = @import("../felix/zig/parachunkexec.zig");

const ContextSum = struct {
    data: []const f64,
    partial_sums: []f64,
};

fn jobStaticSum(
    ctx_ptr: *anyopaque,
    worker_idx: usize,
    range_start: usize,
    range_end: usize,
) void {
    const ctx: *ContextSum = @ptrCast(@alignCast(ctx_ptr));
    var sum: f64 = 0.0;
    for (range_start..range_end) |ii| {
        sum += ctx.data[ii];
    }
    ctx.partial_sums[worker_idx] = sum;
}

const ContextFill = struct {
    out_data: []f64,
};

fn jobDynFill(
    ctx_ptr: *anyopaque,
    worker_idx: usize,
    range_start: usize,
    range_end: usize,
) void {
    _ = worker_idx;
    const ctx: *ContextFill = @ptrCast(@alignCast(ctx_ptr));
    for (range_start..range_end) |ii| {
        ctx.out_data[ii] = @as(f64, @floatFromInt(ii)) * 2.0;
    }
}

test "ParaChunkExecutor static range execution" {
    const total_threads: u16 = 4;
    var threaded_io = pce.initThreadedIo(std.testing.allocator, total_threads);
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const n_elements: usize = 1000;
    const data = try std.testing.allocator.alloc(f64, n_elements);
    defer std.testing.allocator.free(data);

    for (0..n_elements) |ii| {
        data[ii] = 1.0;
    }

    const partial_sums = try std.testing.allocator.alloc(f64, total_threads);
    defer std.testing.allocator.free(partial_sums);
    @memset(partial_sums, 0.0);

    var ctx = ContextSum{
        .data = data,
        .partial_sums = partial_sums,
    };

    var chunk_exec = pce.ParaChunkExecutor.init(io, total_threads);
    try chunk_exec.runStaticRange(
        &ctx,
        jobStaticSum,
        n_elements,
        n_elements / total_threads,
    );

    var total_sum: f64 = 0.0;
    for (partial_sums) |ps| {
        total_sum += ps;
    }
    try std.testing.expectEqual(@as(f64, 1000.0), total_sum);
}

test "ParaChunkExecutor dynamic range execution" {
    const total_threads: u16 = 4;
    var threaded_io = pce.initThreadedIo(std.testing.allocator, total_threads);
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const n_elements: usize = 2048;
    const out_data = try std.testing.allocator.alloc(f64, n_elements);
    defer std.testing.allocator.free(out_data);
    @memset(out_data, 0.0);

    var ctx = ContextFill{
        .out_data = out_data,
    };

    var chunk_exec = pce.ParaChunkExecutor.init(io, total_threads);
    try chunk_exec.runDynRange(
        &ctx,
        jobDynFill,
        n_elements,
        32,
    );

    for (0..n_elements) |ii| {
        const expected = @as(f64, @floatFromInt(ii)) * 2.0;
        try std.testing.expectEqual(expected, out_data[ii]);
    }
}
