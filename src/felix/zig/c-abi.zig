// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const elements = @import("elements.zig");
const mesh_interp = @import("mesh_interp.zig");
const transforms = @import("transforms.zig");
const errors = @import("errors.zig");
const random = @import("random.zig");
const sensor_sim = @import("sensor_sim.zig");
const stats = @import("stats.zig");

const F: type = f64;

// --------------------------------------------------------------------------------------
// Public C ABI
//
// The public Felix C ABI is fixed to f64 precision.
// This keeps the exported ABI stable for C, Cython, and Python callers.
// --------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------
// Error Handling
// --------------------------------------------------------------------------------------

const error_buf_len: usize = 512;
var last_error_buf: [error_buf_len]u8 = [_]u8{0} ** error_buf_len;

fn clearLastError() void {
    @memset(last_error_buf[0..], 0);
}

fn setLastErrorSlice(msg: []const u8) void {
    clearLastError();
    const copy_len = @min(msg.len, error_buf_len - 1);
    @memcpy(last_error_buf[0..copy_len], msg[0..copy_len]);
}

fn setLastError(err: anyerror) void {
    var msg_buf: [error_buf_len]u8 = undefined;
    const msg = std.fmt.bufPrint(
        msg_buf[0..],
        "{s}",
        .{@errorName(err)},
    ) catch @errorName(err);
    setLastErrorSlice(msg);
}

pub export fn felixGetLastError(
    out_buf: [*c]u8,
    out_buf_len: usize,
) usize {
    if (out_buf == null or out_buf_len == 0) {
        return 0;
    }
    const msg_len = std.mem.indexOfScalar(
        u8,
        last_error_buf[0..],
        0,
    ) orelse last_error_buf.len;
    const copy_len = @min(msg_len, out_buf_len - 1);
    @memcpy(out_buf[0..copy_len], last_error_buf[0..copy_len]);
    out_buf[copy_len] = 0;
    return copy_len;
}

// --------------------------------------------------------------------------------------
// Public Entry Points
// --------------------------------------------------------------------------------------

pub export fn felixSimulatePointSensors(
    mesh_in_ptr: [*c]const sensor_sim.SimMeshInput,
    sensor_in_ptr: [*c]const sensor_sim.SensorArrayInput,
    error_specs_ptr: [*c]const errors.ErrorSpec,
    num_errors: usize,
    out_truth_ptr: [*c]F,
    out_measurements_ptr: [*c]F,
    out_errs_sys_ptr: [*c]F,
    out_errs_rand_ptr: [*c]F,
    out_errs_total_ptr: [*c]F,
) i32 {
    if (mesh_in_ptr == null or sensor_in_ptr == null or
        out_truth_ptr == null or out_measurements_ptr == null)
    {
        setLastErrorSlice("Null pointer passed to felixSimulatePointSensors");
        return -1;
    }

    sensor_sim.runSensorSimulation(
        mesh_in_ptr,
        sensor_in_ptr,
        error_specs_ptr,
        num_errors,
        out_truth_ptr,
        out_measurements_ptr,
        out_errs_sys_ptr,
        out_errs_rand_ptr,
        out_errs_total_ptr,
        0,
    );

    return 0;
}

pub export fn felixSimulatePointSensorExperiments(
    mesh_in_ptr: [*c]const sensor_sim.SimMeshInput,
    sensor_in_ptr: [*c]const sensor_sim.SensorArrayInput,
    error_specs_ptr: [*c]const errors.ErrorSpec,
    num_errors: usize,
    num_experiments: usize,
    seed: u64,
    out_truth_ptr: [*c]F,
    out_measurements_ptr: [*c]F,
    out_errs_sys_ptr: [*c]F,
    out_errs_rand_ptr: [*c]F,
    out_errs_total_ptr: [*c]F,
    out_pert_positions_ptr: [*c]F,
    out_pert_times_ptr: [*c]F,
) i32 {
    if (mesh_in_ptr == null or sensor_in_ptr == null or
        out_truth_ptr == null or out_measurements_ptr == null)
    {
        setLastErrorSlice("Null pointer passed to experiment simulation");
        return -1;
    }

    const num_times = if (sensor_in_ptr[0].num_sample_times > 0)
        sensor_in_ptr[0].num_sample_times
    else
        mesh_in_ptr[0].num_sim_times;
    const result_len = sensor_in_ptr[0].num_sensors *
        mesh_in_ptr[0].num_components * num_times;
    for (0..num_experiments) |ee| {
        const offset = ee * result_len;
        sensor_sim.runSensorSimulation(
            mesh_in_ptr,
            sensor_in_ptr,
            error_specs_ptr,
            num_errors,
            out_truth_ptr + offset,
            out_measurements_ptr + offset,
            if (out_errs_sys_ptr != null) out_errs_sys_ptr + offset else null,
            if (out_errs_rand_ptr != null) out_errs_rand_ptr + offset else null,
            if (out_errs_total_ptr != null) out_errs_total_ptr + offset else null,
            seed +% @as(u64, @intCast(ee)),
        );
        if (out_pert_positions_ptr != null and sensor_in_ptr[0].work_positions_ptr != null) {
            @memcpy(
                (out_pert_positions_ptr + ee * sensor_in_ptr[0].num_sensors * 3)[0 .. sensor_in_ptr[0].num_sensors * 3],
                sensor_in_ptr[0].work_positions_ptr[0 .. sensor_in_ptr[0].num_sensors * 3],
            );
        }
        if (out_pert_times_ptr != null and sensor_in_ptr[0].work_times_ptr != null) {
            @memcpy(
                (out_pert_times_ptr + ee * num_times)[0..num_times],
                sensor_in_ptr[0].work_times_ptr[0..num_times],
            );
        }
    }
    return 0;
}

pub export fn felixCalcExperimentStats(
    values_ptr: [*c]const F,
    num_experiments: usize,
    num_values: usize,
    out_mean_ptr: [*c]F,
    out_std_ptr: [*c]F,
    out_min_ptr: [*c]F,
    out_max_ptr: [*c]F,
    out_median_ptr: [*c]F,
    out_var_ptr: [*c]F,
    out_mad_ptr: [*c]F,
) i32 {
    if (values_ptr == null or out_mean_ptr == null or out_std_ptr == null or
        out_min_ptr == null or out_max_ptr == null or out_median_ptr == null or
        out_var_ptr == null or out_mad_ptr == null)
    {
        setLastErrorSlice("Null pointer passed to felixCalcExperimentStats");
        return -1;
    }

    stats.calcExperimentStats(
        std.heap.c_allocator,
        values_ptr[0 .. num_experiments * num_values],
        num_experiments,
        num_values,
        out_mean_ptr[0..num_values],
        out_std_ptr[0..num_values],
        out_min_ptr[0..num_values],
        out_max_ptr[0..num_values],
        out_median_ptr[0..num_values],
        out_var_ptr[0..num_values],
        out_mad_ptr[0..num_values],
    ) catch |err| {
        setLastError(err);
        return -1;
    };
    return 0;
}

pub export fn felixTransformVectors2D(
    rot_mat_22_ptr: [*c]const F,
    vx_in_ptr: [*c]const F,
    vy_in_ptr: [*c]const F,
    num_vectors: usize,
    out_vx_ptr: [*c]F,
    out_vy_ptr: [*c]F,
) void {
    var r22: [4]F = undefined;
    for (0..4) |ii| r22[ii] = rot_mat_22_ptr[ii];

    for (0..num_vectors) |ii| {
        var tx: F = undefined;
        var ty: F = undefined;
        transforms.transformVector2D(
            &r22,
            vx_in_ptr[ii],
            vy_in_ptr[ii],
            &tx,
            &ty,
        );
        out_vx_ptr[ii] = tx;
        out_vy_ptr[ii] = ty;
    }
}

pub export fn felixTransformVectors3D(
    rot_mat_33_ptr: [*c]const F,
    vx_in_ptr: [*c]const F,
    vy_in_ptr: [*c]const F,
    vz_in_ptr: [*c]const F,
    num_vectors: usize,
    out_vx_ptr: [*c]F,
    out_vy_ptr: [*c]F,
    out_vz_ptr: [*c]F,
) void {
    var r33: [9]F = undefined;
    for (0..9) |ii| r33[ii] = rot_mat_33_ptr[ii];

    for (0..num_vectors) |ii| {
        var tx: F = undefined;
        var ty: F = undefined;
        var tz: F = undefined;
        transforms.transformVector3D(
            &r33,
            vx_in_ptr[ii],
            vy_in_ptr[ii],
            vz_in_ptr[ii],
            &tx,
            &ty,
            &tz,
        );
        out_vx_ptr[ii] = tx;
        out_vy_ptr[ii] = ty;
        out_vz_ptr[ii] = tz;
    }
}

pub export fn felixPrintSensorData(
    positions_ptr: [*c]const f64,
    positions_len: usize,
    sample_times_ptr: [*c]const f64,
    sample_times_len: usize,
    spatial_dims_ptr: [*c]const f64,
    spatial_dims_len: usize,
) void {
    if (positions_ptr == null and positions_len > 0) return;
    const positions = positions_ptr[0..positions_len];

    const sample_times = if (sample_times_ptr != null and sample_times_len > 0)
        sample_times_ptr[0..sample_times_len]
    else
        &[_]f64{};

    const spatial_dims = if (spatial_dims_ptr != null and spatial_dims_len > 0)
        spatial_dims_ptr[0..spatial_dims_len]
    else
        &[_]f64{};

    if (positions.len % 3 != 0) {
        std.debug.print(
            "[Felix] ERROR: positions length {d} is not divisible by 3.\n",
            .{positions.len},
        );
        return;
    }

    const n_sensors: usize = positions.len / 3;

    std.debug.print("\n[Felix] ---- SensorData ----\n", .{});
    std.debug.print("[Felix] n_sensors       : {d}\n", .{n_sensors});
    std.debug.print("[Felix] n_sample_times  : {d}\n", .{sample_times.len});

    var ii: usize = 0;
    while (ii < n_sensors) : (ii += 1) {
        const px = positions[ii * 3 + 0];
        const py = positions[ii * 3 + 1];
        const pz = positions[ii * 3 + 2];
        std.debug.print(
            "[Felix]   sensor[{d}] pos = ({d:.4}, {d:.4}, {d:.4})\n",
            .{ ii, px, py, pz },
        );
    }

    if (sample_times.len > 0) {
        std.debug.print("[Felix] sample_times    :", .{});
        var jj: usize = 0;
        while (jj < sample_times.len) : (jj += 1) {
            std.debug.print(" {d:.4}", .{sample_times[jj]});
        }
        std.debug.print("\n", .{});
    }

    if (spatial_dims.len == 3) {
        std.debug.print(
            "[Felix] spatial_dims    : ({d:.4}, {d:.4}, {d:.4})\n",
            .{ spatial_dims[0], spatial_dims[1], spatial_dims[2] },
        );
    }

    std.debug.print("[Felix] ----------------------\n\n", .{});
}
