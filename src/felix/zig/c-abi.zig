// --------------------------------------------------------------------------------------
// Felix: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const F = f64;

const ndarray = @import("ndarray.zig");
const vec = @import("vecstack.zig");

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
// Helpers: safe pointer-to-slice conversion
// --------------------------------------------------------------------------------------

fn cConstSliceF64(
    ptr: [*c]const f64,
    len: usize,
) ![]const f64 {
    if (ptr == null and len > 0) {
        return error.NullPointer;
    }
    return ptr[0..len];
}

fn cConstSliceUsize(
    ptr: [*c]const usize,
    len: usize,
) ![]const usize {
    if (ptr == null and len > 0) {
        return error.NullPointer;
    }
    return ptr[0..len];
}

// --------------------------------------------------------------------------------------
// Public Entry Points
// --------------------------------------------------------------------------------------

/// Receive a pyvale SensorData (positions, sample_times, spatial_dims) from
/// Python and print a summary to stderr, proving the call path works.
///
/// positions      : flat row-major f64 array, shape (n_sensors, 3)
/// positions_len  : total number of f64 elements  (= n_sensors * 3)
/// sample_times   : flat f64 array, shape (n_times,); may be null/0-len
/// sample_times_len: number of sample-time elements
/// spatial_dims   : flat f64 array, shape (3,); may be null/0-len
/// spatial_dims_len: number of spatial-dim elements (0 or 3)
pub export fn felixPrintSensorData(
    positions_ptr: [*c]const f64,
    positions_len: usize,
    sample_times_ptr: [*c]const f64,
    sample_times_len: usize,
    spatial_dims_ptr: [*c]const f64,
    spatial_dims_len: usize,
) void {
    const positions = cConstSliceF64(
        positions_ptr,
        positions_len,
    ) catch {
        std.debug.print(
            "[Felix] ERROR: null positions pointer with non-zero length.\n",
            .{},
        );
        return;
    };

    const sample_times = cConstSliceF64(
        sample_times_ptr,
        sample_times_len,
    ) catch {
        std.debug.print(
            "[Felix] ERROR: null sample_times pointer with non-zero length.\n",
            .{},
        );
        return;
    };

    const spatial_dims = cConstSliceF64(
        spatial_dims_ptr,
        spatial_dims_len,
    ) catch {
        std.debug.print(
            "[Felix] ERROR: null spatial_dims pointer with non-zero length.\n",
            .{},
        );
        return;
    };

    // positions must be a multiple of 3 (one [x,y,z] triple per sensor)
    if (positions.len % 3 != 0) {
        std.debug.print(
            "[Felix] ERROR: positions length {d} is not divisible by 3.\n",
            .{positions.len},
        );
        return;
    }

    const n_sensors: usize = positions.len / 3;

    std.debug.print(
        "\n[Felix] ---- SensorData ----\n",
        .{},
    );
    std.debug.print(
        "[Felix] n_sensors       : {d}\n",
        .{n_sensors},
    );
    std.debug.print(
        "[Felix] n_sample_times  : {d}\n",
        .{sample_times.len},
    );

    // Print each sensor position
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

    // Print sample times if provided
    if (sample_times.len > 0) {
        std.debug.print("[Felix] sample_times    :", .{});
        var jj: usize = 0;
        while (jj < sample_times.len) : (jj += 1) {
            std.debug.print(" {d:.4}", .{sample_times[jj]});
        }
        std.debug.print("\n", .{});
    } else {
        std.debug.print(
            "[Felix] sample_times    : (none – using sim time steps)\n",
            .{},
        );
    }

    // Print spatial dims if provided
    if (spatial_dims.len == 3) {
        std.debug.print(
            "[Felix] spatial_dims    : ({d:.4}, {d:.4}, {d:.4})\n",
            .{ spatial_dims[0], spatial_dims[1], spatial_dims[2] },
        );
    } else {
        std.debug.print(
            "[Felix] spatial_dims    : (none)\n",
            .{},
        );
    }

    std.debug.print("[Felix] ----------------------\n\n", .{});
}
