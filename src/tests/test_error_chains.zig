// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const errors = @import("../felix/zig/errors.zig");
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

const F = common.F;
const RoundMethod = errors.RoundMethod;

// --------------------------------------------------------------------------------------
// Unit Tests: Error Models & Mathematical Primitives
// --------------------------------------------------------------------------------------

test "Rounding and digitisation error evaluation" {
    const val: F = 12.345;
    const base: F = 0.5;

    const val_round = errors.evalRound(val, base, .round);
    try std.testing.expect(
        common.isApproxEqual(val_round, 12.5, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const val_floor = errors.evalRound(val, base, .floor);
    try std.testing.expect(
        common.isApproxEqual(val_floor, 12.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const val_ceil = errors.evalRound(val, base, .ceil);
    try std.testing.expect(
        common.isApproxEqual(val_ceil, 12.5, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "Polynomial calibration evaluation" {
    // P(x) = 2.0 + 3.0*x + 4.0*x^2
    const coeffs = [_]F{ 2.0, 3.0, 4.0 };
    const x_val: F = 2.0;

    // Expected = 2.0 + 3.0*2.0 + 4.0*4.0 = 2 + 6 + 16 = 24.0
    const res = errors.evalPoly(&coeffs, x_val);
    try std.testing.expect(
        common.isApproxEqual(res, 24.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "1D Table lookup interpolation" {
    // Lookup table: (x, y) pairs: (0, 0), (10, 100), (20, 400)
    const table = [_]F{
        0.0,  0.0,
        10.0, 100.0,
        20.0, 400.0,
    };

    const y_interp = errors.evalTableLookup1D(&table, 3, 5.0);
    try std.testing.expect(
        common.isApproxEqual(y_interp, 50.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const y_clamped_low = errors.evalTableLookup1D(&table, 3, -5.0);
    try std.testing.expect(
        common.isApproxEqual(y_clamped_low, 0.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const y_clamped_high = errors.evalTableLookup1D(&table, 3, 25.0);
    try std.testing.expect(
        common.isApproxEqual(y_clamped_high, 400.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}
