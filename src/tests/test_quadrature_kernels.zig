// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const quadrature = @import("../felix/zig/quadrature.zig");
const kernels = @import("../felix/zig/kernels.zig");
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

const F = common.F;
const KernelType = kernels.KernelType;
const KernelSpec = kernels.KernelSpec;

// --------------------------------------------------------------------------------------
// Unit Tests: Quadrature & Spatial Weighting Kernels
// --------------------------------------------------------------------------------------

test "Gauss-Legendre 1D quadrature weights sum to 2.0" {
    var nodes: [3]F = undefined;
    var weights: [3]F = undefined;

    _ = quadrature.getGaussLegendre1D(3, &nodes, &weights);

    var sum_weights: F = 0.0;
    for (weights) |weight_val| {
        sum_weights += weight_val;
    }

    try std.testing.expect(
        common.isApproxEqual(sum_weights, 2.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(nodes[1], 0.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(nodes[0], -nodes[2], tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "Trapezoidal 1D quadrature partition of unity" {
    var nodes: [5]F = undefined;
    var weights: [5]F = undefined;

    _ = quadrature.getTrapezoidal1D(4, &nodes, &weights);

    var sum_weights: F = 0.0;
    for (weights) |weight_val| {
        sum_weights += weight_val;
    }

    try std.testing.expect(
        common.isApproxEqual(sum_weights, 2.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(nodes[0], -1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(nodes[4], 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "Spatial weighting kernels evaluation" {
    const radius: F = 2.0;

    const uniform_spec = KernelSpec{
        .kernel_type = @intFromEnum(KernelType.uniform),
        .param0 = 0.0,
        .param1 = 0.0,
        .param2 = 0.0,
    };

    const w_uniform_inside = kernels.evalWeight(&uniform_spec, 1.0, 0.0, 0.0, 1);
    try std.testing.expect(
        common.isApproxEqual(w_uniform_inside, 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const tri_spec = KernelSpec{
        .kernel_type = @intFromEnum(KernelType.triangular),
        .param0 = radius,
        .param1 = radius,
        .param2 = radius,
    };

    const w_tri_center = kernels.evalWeight(&tri_spec, 0.0, 0.0, 0.0, 1);
    const w_tri_mid = kernels.evalWeight(&tri_spec, 1.0, 0.0, 0.0, 1);

    try std.testing.expect(
        common.isApproxEqual(w_tri_center, 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(w_tri_mid, 0.5, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}
