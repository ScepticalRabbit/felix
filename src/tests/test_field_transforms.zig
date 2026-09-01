// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const transforms = @import("../felix/zig/transforms.zig");
const invariants = @import("../felix/zig/tensor_invariants.zig");
const common = @import("../dev_support/tests.zig");
const tcfg = @import("../dev_support/testconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

const F = common.F;

// --------------------------------------------------------------------------------------
// Unit Tests: Vector & Tensor Transformations
// --------------------------------------------------------------------------------------

test "2D vector rotation transformation" {
    const angle_rad: F = std.math.pi / 2.0;
    const cos_theta = @cos(angle_rad);
    const sin_theta = @sin(angle_rad);

    const rot_mat_22 = [4]F{
        cos_theta, -sin_theta,
        sin_theta, cos_theta,
    };

    var out_vx: F = 0.0;
    var out_vy: F = 0.0;
    transforms.transformVector2D(&rot_mat_22, 1.0, 0.0, &out_vx, &out_vy);

    try std.testing.expect(
        common.isApproxEqual(out_vx, 0.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_vy, 1.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "3D vector rotation transformation" {
    // Identity rotation matrix
    const rot_mat = [9]F{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };

    var out_vx: F = 0.0;
    var out_vy: F = 0.0;
    var out_vz: F = 0.0;
    transforms.transformVector3D(&rot_mat, 1.5, 2.5, 3.5, &out_vx, &out_vy, &out_vz);

    try std.testing.expect(
        common.isApproxEqual(out_vx, 1.5, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_vy, 2.5, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_vz, 3.5, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "2D Von Mises and Hydrostatic stress invariants" {
    const sxx: F = 100.0;
    const syy: F = 50.0;
    const sxy: F = 25.0;

    const vm = invariants.evalVonMises2D(sxx, syy, sxy);
    const expected_vm: F = @sqrt(
        100.0 * 100.0 - 100.0 * 50.0 + 50.0 * 50.0 + 3.0 * 25.0 * 25.0,
    );

    try std.testing.expect(
        common.isApproxEqual(vm, expected_vm, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const hydro = invariants.evalHydrostatic2D(sxx, syy);
    try std.testing.expect(
        common.isApproxEqual(hydro, 75.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "3D Von Mises and Hydrostatic stress invariants" {
    const sxx: F = 100.0;
    const syy: F = 50.0;
    const szz: F = 25.0;
    const sxy: F = 10.0;
    const syz: F = 15.0;
    const sxz: F = 5.0;

    const hydro = invariants.evalHydrostatic3D(sxx, syy, szz);
    try std.testing.expect(
        common.isApproxEqual(hydro, (100.0 + 50.0 + 25.0) / 3.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );

    const vm = invariants.evalVonMises3D(sxx, syy, szz, sxy, syz, sxz);
    const d1 = sxx - syy;
    const d2 = syy - szz;
    const d3 = szz - sxx;
    const expected_vm = @sqrt(
        0.5 * (d1 * d1 + d2 * d2 + d3 * d3) +
            3.0 * (sxy * sxy + syz * syz + sxz * sxz),
    );

    try std.testing.expect(
        common.isApproxEqual(vm, expected_vm, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "2D Principal stresses" {
    const sxx: F = 100.0;
    const syy: F = 50.0;
    const sxy: F = 0.0;

    var p1: F = 0.0;
    var p2: F = 0.0;
    var max_shear: F = 0.0;
    invariants.evalPrincipal2D(sxx, syy, sxy, &p1, &p2, &max_shear);

    try std.testing.expect(
        common.isApproxEqual(p1, 100.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(p2, 50.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(max_shear, 25.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "2D tensor rotation transformation" {
    const angle_rad: F = std.math.pi / 4.0;
    const cos_theta = @cos(angle_rad);
    const sin_theta = @sin(angle_rad);

    const rot_mat_22 = [4]F{
        cos_theta, -sin_theta,
        sin_theta, cos_theta,
    };

    var out_s_xx: F = 0.0;
    var out_s_yy: F = 0.0;
    var out_s_xy: F = 0.0;
    transforms.transformTensor2D(
        &rot_mat_22,
        100.0,
        50.0,
        0.0,
        &out_s_xx,
        &out_s_yy,
        &out_s_xy,
    );

    try std.testing.expect(
        common.isApproxEqual(out_s_xx, 75.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_s_yy, 75.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_s_xy, 25.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}

test "3D tensor rotation transformation" {
    const angle_rad: F = std.math.pi / 4.0;
    const cos_theta = @cos(angle_rad);
    const sin_theta = @sin(angle_rad);

    const rot_mat_33 = [9]F{
        cos_theta, -sin_theta, 0.0,
        sin_theta, cos_theta,  0.0,
        0.0,       0.0,        1.0,
    };

    const in_tensor_6 = [6]F{ 100.0, 50.0, 10.0, 0.0, 0.0, 0.0 };
    var out_tensor_6: [6]F = undefined;
    transforms.transformTensor3D(&rot_mat_33, &in_tensor_6, &out_tensor_6);

    try std.testing.expect(
        common.isApproxEqual(out_tensor_6[0], 75.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_tensor_6[1], 75.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_tensor_6[2], 10.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_tensor_6[3], 25.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_tensor_6[4], 0.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
    try std.testing.expect(
        common.isApproxEqual(out_tensor_6[5], 0.0, tcfg.REL_TOL, tcfg.ABS_TOL),
    );
}
