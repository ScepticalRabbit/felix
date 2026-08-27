// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("errors_common.zig");
const scalar = @import("errors_scalar.zig");

const F = common.F;
const VecSF = common.VecSF;
const SimdWidth = common.SimdWidth;
const RoundMethod = common.RoundMethod;

// --------------------------------------------------------------------------------------
// SIMD Error Evaluation Functions
// --------------------------------------------------------------------------------------

pub fn evalRoundSIMD(val_packet: VecSF, base_packet: VecSF, method: RoundMethod) VecSF {
    const scaled = val_packet / base_packet;
    const rounded = switch (method) {
        .floor => @floor(scaled),
        .ceil => @ceil(scaled),
        .round => @round(scaled),
    };
    return base_packet * rounded;
}

pub fn evalPolySIMD(coeffs: []const F, x_packet: VecSF) VecSF {
    if (coeffs.len == 0) return @splat(0.0);
    var res: VecSF = @splat(0.0);
    var x_pow: VecSF = @splat(1.0);
    for (coeffs) |coeff| {
        const c_vec: VecSF = @splat(coeff);
        res += c_vec * x_pow;
        x_pow *= x_packet;
    }
    return res;
}

pub fn applyOffsetSIMD(buffer: []F, offset_val: F) void {
    const total_len = buffer.len;
    const offset_vec: VecSF = @splat(offset_val);
    var offset: usize = 0;

    while (offset + SimdWidth <= total_len) : (offset += SimdWidth) {
        const val_vec: VecSF = buffer[offset..][0..SimdWidth].*;
        buffer[offset..][0..SimdWidth].* = val_vec + offset_vec;
    }

    while (offset < total_len) : (offset += 1) {
        buffer[offset] += offset_val;
    }
}

pub fn applyPercentageSIMD(buffer: []F, percent_factor: F) void {
    const total_len = buffer.len;
    const factor_vec: VecSF = @splat(1.0 + percent_factor);
    var offset: usize = 0;

    while (offset + SimdWidth <= total_len) : (offset += SimdWidth) {
        const val_vec: VecSF = buffer[offset..][0..SimdWidth].*;
        buffer[offset..][0..SimdWidth].* = val_vec * factor_vec;
    }

    while (offset < total_len) : (offset += 1) {
        buffer[offset] *= (1.0 + percent_factor);
    }
}

pub fn applySaturationSIMD(buffer: []F, min_val: F, max_val: F) void {
    const total_len = buffer.len;
    const min_vec: VecSF = @splat(min_val);
    const max_vec: VecSF = @splat(max_val);
    var offset: usize = 0;

    while (offset + SimdWidth <= total_len) : (offset += SimdWidth) {
        const val_vec: VecSF = buffer[offset..][0..SimdWidth].*;
        const clamped_vec = @min(max_vec, @max(min_vec, val_vec));
        buffer[offset..][0..SimdWidth].* = clamped_vec;
    }

    while (offset < total_len) : (offset += 1) {
        buffer[offset] = std.math.clamp(buffer[offset], min_val, max_val);
    }
}

// --------------------------------------------------------------------------------------
// Re-export Scalar API
// --------------------------------------------------------------------------------------

pub const evalRound = scalar.evalRound;
pub const evalPoly = scalar.evalPoly;
pub const evalTableLookup1D = scalar.evalTableLookup1D;

// --------------------------------------------------------------------------------------
// Tests: Parity Verification
// --------------------------------------------------------------------------------------

test "errors SIMD vs scalar parity" {
    const coeffs = [_]F{ 1.0, 2.0, 0.5 };
    var x_arr: [SimdWidth]F = undefined;
    for (0..SimdWidth) |ii| {
        x_arr[ii] = @as(F, @floatFromInt(ii)) * 0.5;
    }

    const x_vec: VecSF = x_arr;
    const poly_vec = evalPolySIMD(&coeffs, x_vec);
    const poly_arr: [SimdWidth]F = poly_vec;

    for (0..SimdWidth) |ii| {
        const scal_val = scalar.evalPoly(&coeffs, x_arr[ii]);
        try std.testing.expectApproxEqAbs(scal_val, poly_arr[ii], 1e-12);
    }
}
