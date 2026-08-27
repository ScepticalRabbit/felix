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

const F = common.F;
const RoundMethod = common.RoundMethod;

// --------------------------------------------------------------------------------------
// Scalar Error Functions
// --------------------------------------------------------------------------------------

pub fn evalRound(val: F, base: F, method: RoundMethod) F {
    const scaled = val / base;
    const rounded = switch (method) {
        .floor => @floor(scaled),
        .ceil => @ceil(scaled),
        .round => @round(scaled),
    };
    return base * rounded;
}

pub fn evalPoly(coeffs: []const F, x_val: F) F {
    if (coeffs.len == 0) return 0.0;
    var res: F = 0.0;
    var x_pow: F = 1.0;
    for (coeffs) |coeff| {
        res += coeff * x_pow;
        x_pow *= x_val;
    }
    return res;
}

pub fn evalTableLookup1D(table: []const F, n_rows: usize, in_val: F) F {
    if (n_rows == 0) return in_val;
    if (n_rows == 1) return table[1];

    const x0 = table[0];
    const x_end = table[(n_rows - 1) * 2 + 0];

    if (in_val <= x0) return table[1];
    if (in_val >= x_end) return table[(n_rows - 1) * 2 + 1];

    for (0..n_rows - 1) |rr| {
        const row_x0 = table[rr * 2 + 0];
        const row_y0 = table[rr * 2 + 1];
        const row_x1 = table[(rr + 1) * 2 + 0];
        const row_y1 = table[(rr + 1) * 2 + 1];

        if (in_val >= row_x0 and in_val <= row_x1) {
            const dx = row_x1 - row_x0;
            if (dx < 1e-15) return row_y0;
            const alpha_val = (in_val - row_x0) / dx;
            return (1.0 - alpha_val) * row_y0 + alpha_val * row_y1;
        }
    }

    return table[(n_rows - 1) * 2 + 1];
}
