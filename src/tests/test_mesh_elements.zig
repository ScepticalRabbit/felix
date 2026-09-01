// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const elements = @import("../felix/zig/elements.zig");
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

const F = common.F;
const ElementType = elements.ElementType;

// --------------------------------------------------------------------------------------
// Unit Tests: Shape Functions & Inversion
// --------------------------------------------------------------------------------------

test "TRI3 shape function and inversion" {
    var weights: [3]F = undefined;
    elements.shapeTri3(0.2, 0.3, &weights);

    const sum_weights: F = weights[0] + weights[1] + weights[2];
    try std.testing.expect(
        common.isApproxEqual(sum_weights, 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const node_x = [3]F{ 0.0, 1.0, 0.0 };
    const node_y = [3]F{ 0.0, 0.0, 1.0 };
    var inv_res: elements.InverseResult = undefined;

    elements.invertTri3(&node_x, &node_y, 0.2, 0.3, 1e-6, &inv_res);

    try std.testing.expect(inv_res.inside);
    try std.testing.expectEqual(@as(usize, 3), inv_res.node_count);
    try std.testing.expect(
        common.isApproxEqual(inv_res.weights[1], 0.2, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(inv_res.weights[2], 0.3, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "QUAD4 shape function partition of unity" {
    var weights: [4]F = undefined;
    elements.shapeQuad4(0.25, -0.5, &weights, null, null);

    var sum_weights: F = 0.0;
    for (weights) |weight_val| {
        sum_weights += weight_val;
    }
    try std.testing.expect(
        common.isApproxEqual(sum_weights, 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const node_x = [4]F{ 0.0, 2.0, 2.0, 0.0 };
    const node_y = [4]F{ 0.0, 0.0, 2.0, 2.0 };
    var inv_res: elements.InverseResult = undefined;

    elements.invertQuad4(&node_x, &node_y, 1.0, 1.0, 1e-6, &inv_res);

    try std.testing.expect(inv_res.inside);
    try std.testing.expectEqual(@as(usize, 4), inv_res.node_count);
    for (0..4) |ii| {
        try std.testing.expect(
            common.isApproxEqual(inv_res.weights[ii], 0.25, tcfg.REL_TOL, tcfg.ABS_TOL),
        );
    }
}

test "TET4 shape function and inversion" {
    var weights: [4]F = undefined;
    elements.shapeTet4(0.1, 0.2, 0.3, &weights);

    var sum_weights: F = 0.0;
    for (weights) |weight_val| {
        sum_weights += weight_val;
    }
    try std.testing.expect(
        common.isApproxEqual(sum_weights, 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const node_x = [4]F{ 0.0, 1.0, 0.0, 0.0 };
    const node_y = [4]F{ 0.0, 0.0, 1.0, 0.0 };
    const node_z = [4]F{ 0.0, 0.0, 0.0, 1.0 };
    var inv_res: elements.InverseResult = undefined;

    elements.invertTet4(&node_x, &node_y, &node_z, 0.1, 0.2, 0.3, 1e-6, &inv_res);

    try std.testing.expect(inv_res.inside);
    try std.testing.expectEqual(@as(usize, 4), inv_res.node_count);
    try std.testing.expect(
        common.isApproxEqual(inv_res.weights[1], 0.1, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(inv_res.weights[2], 0.2, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(inv_res.weights[3], 0.3, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "HEX8 shape function and center point inversion" {
    var weights: [8]F = undefined;
    elements.shapeHex8(0.0, 0.0, 0.0, &weights, null, null, null);

    var sum_weights: F = 0.0;
    for (weights) |weight_val| {
        sum_weights += weight_val;
    }
    try std.testing.expect(
        common.isApproxEqual(sum_weights, 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const node_x = [8]F{ 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 };
    const node_y = [8]F{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0 };
    const node_z = [8]F{ 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };
    var inv_res: elements.InverseResult = undefined;

    elements.invertHex8(&node_x, &node_y, &node_z, 0.5, 0.5, 0.5, 1e-6, &inv_res);

    try std.testing.expect(inv_res.inside);
    try std.testing.expectEqual(@as(usize, 8), inv_res.node_count);
    for (0..8) |ii| {
        try std.testing.expect(
            common.isApproxEqual(inv_res.weights[ii], 0.125, tcfg.REL_TOL, tcfg.ABS_TOL),
        );
    }
}
