// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("sensor_sim_common.zig");
const pce = @import("parachunkexec.zig");
const sensor_sim = @import("sensor_sim.zig");

const F = common.F;
const SimMeshInput = common.SimMeshInput;
const SensorArrayInput = common.SensorArrayInput;
const ErrorSpec = common.ErrorSpec;

pub const ExpSimParallelContext = struct {
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    error_specs_ptr: [*c]const ErrorSpec,
    num_errors: usize,
    out_truth_all: [*c]F,
    out_meas_all: [*c]F,
    out_errs_total_all: ?[*c]F,
    exp_stride: usize,
    base_seed: u64,
    seed_stride: u64,
};

fn jobExperimentChunk(
    ctx_ptr: *anyopaque,
    worker_idx: usize,
    range_start: usize,
    range_end: usize,
) void {
    _ = worker_idx;
    const ctx: *const ExpSimParallelContext = @ptrCast(@alignCast(ctx_ptr));
    for (range_start..range_end) |exp_idx| {
        const seed_offset = ctx.base_seed +% (@as(u64, @intCast(exp_idx)) *% ctx.seed_stride);
        const exp_truth = ctx.out_truth_all + exp_idx * ctx.exp_stride;
        const exp_meas = ctx.out_meas_all + exp_idx * ctx.exp_stride;
        const exp_total_err = if (ctx.out_errs_total_all) |ptr|
            ptr + exp_idx * ctx.exp_stride
        else
            null;

        sensor_sim.runSensorSimulation(
            ctx.mesh_in,
            ctx.sensor_in,
            ctx.error_specs_ptr,
            ctx.num_errors,
            exp_truth,
            exp_meas,
            null,
            null,
            exp_total_err,
            seed_offset,
        );
    }
}

pub fn runExperimentSimulationParallel(
    outer_alloc: std.mem.Allocator,
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    error_specs_ptr: [*c]const ErrorSpec,
    num_errors: usize,
    out_truth_all: [*c]F,
    out_meas_all: [*c]F,
    out_errs_total_all: ?[*c]F,
    num_experiments: usize,
    base_seed: u64,
    seed_stride: u64,
    workers_num: u16,
    grain_size: usize,
) !void {
    if (num_experiments == 0) {
        return;
    }

    const num_sensors = sensor_in.num_sensors;
    const num_comps = mesh_in.num_components;
    const num_out_times = if (sensor_in.num_sample_times > 0)
        sensor_in.num_sample_times
    else
        mesh_in.num_sim_times;
    const exp_stride = num_sensors * num_comps * num_out_times;

    const ctx = ExpSimParallelContext{
        .mesh_in = mesh_in,
        .sensor_in = sensor_in,
        .error_specs_ptr = error_specs_ptr,
        .num_errors = num_errors,
        .out_truth_all = out_truth_all,
        .out_meas_all = out_meas_all,
        .out_errs_total_all = out_errs_total_all,
        .exp_stride = exp_stride,
        .base_seed = base_seed,
        .seed_stride = seed_stride,
    };

    const effective_workers = if (workers_num == 0)
        @as(u16, @intCast(@min(@as(usize, 64), std.Thread.getCpuCount() catch 1)))
    else
        workers_num;

    if (effective_workers <= 1 or num_experiments <= 1) {
        jobExperimentChunk(@constCast(&ctx), 0, 0, num_experiments);
        return;
    }

    var threaded_io = pce.initThreadedIo(outer_alloc, effective_workers);
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const effective_grain = if (grain_size == 0) 1 else grain_size;
    var chunk_exec = pce.ParaChunkExecutor.init(io, effective_workers);
    try chunk_exec.runDynRange(
        @constCast(&ctx),
        jobExperimentChunk,
        num_experiments,
        effective_grain,
    );
}
