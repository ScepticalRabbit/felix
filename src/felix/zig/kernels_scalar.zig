// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("kernels_common.zig");

const F = common.F;
const KernelType = common.KernelType;
const KernelSpec = common.KernelSpec;

pub fn evalWeight(
    spec: *const KernelSpec,
    x: F,
    y: F,
    z: F,
    dims: usize,
) F {
    const k_type: KernelType = @enumFromInt(spec.kernel_type);
    switch (k_type) {
        .uniform => return 1.0,
        .gaussian => {
            const sig_x = if (spec.param0 != 0.0) spec.param0 else 1.0;
            const sig_y = if (spec.param1 != 0.0) spec.param1 else sig_x;
            const sig_z = if (spec.param2 != 0.0) spec.param2 else sig_x;

            var r_sq: F = (x / sig_x) * (x / sig_x);
            if (dims >= 2) r_sq += (y / sig_y) * (y / sig_y);
            if (dims >= 3) r_sq += (z / sig_z) * (z / sig_z);
            return @exp(-0.5 * r_sq);
        },
        .triangular => {
            const rad_x = if (spec.param0 != 0.0) spec.param0 else 1.0;
            const rad_y = if (spec.param1 != 0.0) spec.param1 else rad_x;
            const rad_z = if (spec.param2 != 0.0) spec.param2 else rad_x;

            var r_sq: F = (x / rad_x) * (x / rad_x);
            if (dims >= 2) r_sq += (y / rad_y) * (y / rad_y);
            if (dims >= 3) r_sq += (z / rad_z) * (z / rad_z);
            const r_norm = @sqrt(r_sq);
            return if (r_norm < 1.0) 1.0 - r_norm else 0.0;
        },
        .cosine => {
            const rad = if (spec.param0 != 0.0) spec.param0 else 1.0;
            var r_sq: F = (x / rad) * (x / rad);
            if (dims >= 2) r_sq += (y / rad) * (y / rad);
            if (dims >= 3) r_sq += (z / rad) * (z / rad);
            const r_norm = @sqrt(r_sq);
            if (r_norm >= 1.0) return 0.0;
            return @cos(0.5 * std.math.pi * r_norm);
        },
        .epanechnikov => {
            const rad = if (spec.param0 != 0.0) spec.param0 else 1.0;
            var r_sq: F = (x / rad) * (x / rad);
            if (dims >= 2) r_sq += (y / rad) * (y / rad);
            if (dims >= 3) r_sq += (z / rad) * (z / rad);
            return if (r_sq < 1.0) 1.0 - r_sq else 0.0;
        },
        .sinc => {
            const scale = if (spec.param0 != 0.0) spec.param0 else 1.0;
            var r_sq: F = (x / scale) * (x / scale);
            if (dims >= 2) r_sq += (y / scale) * (y / scale);
            if (dims >= 3) r_sq += (z / scale) * (z / scale);
            const r = @sqrt(r_sq);
            if (r < 1e-12) return 1.0;
            const pi_r = std.math.pi * r;
            return @sin(pi_r) / pi_r;
        },
        .lanczos => {
            const a = if (spec.param0 != 0.0) spec.param0 else 2.0;
            var r_sq: F = x * x;
            if (dims >= 2) r_sq += y * y;
            if (dims >= 3) r_sq += z * z;
            const r = @sqrt(r_sq);
            if (r < 1e-12) return 1.0;
            if (r >= a) return 0.0;
            const pi_r = std.math.pi * r;
            const pi_r_a = pi_r / a;
            return (@sin(pi_r) / pi_r) * (@sin(pi_r_a) / pi_r_a);
        },
    }
}

pub fn evalWeightsBatch(
    spec: *const KernelSpec,
    coords_ptr: [*c]const F,
    num_points: usize,
    dims: usize,
    out_weights_ptr: [*c]F,
) void {
    for (0..num_points) |ii| {
        const offset = ii * dims;
        const x = coords_ptr[offset];
        const y = if (dims >= 2) coords_ptr[offset + 1] else 0.0;
        const z = if (dims >= 3) coords_ptr[offset + 2] else 0.0;
        out_weights_ptr[ii] = evalWeight(spec, x, y, z, dims);
    }
}
