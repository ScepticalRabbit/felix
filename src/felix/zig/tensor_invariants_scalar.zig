// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("tensor_invariants_common.zig");

const F = common.F;
const TensorInvariantType = common.TensorInvariantType;

pub fn evalPrincipal2D(
    sxx: F,
    syy: F,
    sxy: F,
    out_p1: *F,
    out_p2: *F,
    out_max_shear: *F,
) void {
    const center = 0.5 * (sxx + syy);
    const diff_half = 0.5 * (sxx - syy);
    const radius = @sqrt(diff_half * diff_half + sxy * sxy);
    out_p1.* = center + radius;
    out_p2.* = center - radius;
    out_max_shear.* = radius;
}

pub fn evalPrincipal3D(
    sxx: F,
    syy: F,
    szz: F,
    syz: F,
    sxz: F,
    sxy: F,
    out_p1: *F,
    out_p2: *F,
    out_p3: *F,
) void {
    const p1_val = sxy * sxy + syz * syz + sxz * sxz;
    if (p1_val < 1e-24) {
        var vals = [3]F{ sxx, syy, szz };
        if (vals[0] < vals[1]) std.mem.swap(F, &vals[0], &vals[1]);
        if (vals[1] < vals[2]) std.mem.swap(F, &vals[1], &vals[2]);
        if (vals[0] < vals[1]) std.mem.swap(F, &vals[0], &vals[1]);
        out_p1.* = vals[0];
        out_p2.* = vals[1];
        out_p3.* = vals[2];
        return;
    }

    const q = (sxx + syy + szz) / 3.0;
    const dx = sxx - q;
    const dy = syy - q;
    const dz = szz - q;
    const p2_val = dx * dx + dy * dy + dz * dz + 2.0 * p1_val;
    const p = @sqrt(p2_val / 6.0);

    const b_det = dx * (dy * dz - syz * syz) -
        sxy * (sxy * dz - syz * sxz) +
        sxz * (sxy * syz - dy * sxz);

    var r = b_det / (2.0 * p * p * p);
    if (r <= -1.0) {
        r = -1.0;
    } else if (r >= 1.0) {
        r = 1.0;
    }

    const phi = std.math.acos(r) / 3.0;
    const pi_23 = (2.0 * std.math.pi) / 3.0;

    const eig1 = q + 2.0 * p * @cos(phi);
    const eig3 = q + 2.0 * p * @cos(phi + pi_23);
    const eig2 = 3.0 * q - eig1 - eig3;

    var vals = [3]F{ eig1, eig2, eig3 };
    if (vals[0] < vals[1]) std.mem.swap(F, &vals[0], &vals[1]);
    if (vals[1] < vals[2]) std.mem.swap(F, &vals[1], &vals[2]);
    if (vals[0] < vals[1]) std.mem.swap(F, &vals[0], &vals[1]);

    out_p1.* = vals[0];
    out_p2.* = vals[1];
    out_p3.* = vals[2];
}

pub fn evalVonMises2D(sxx: F, syy: F, sxy: F) F {
    const val = sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy;
    return @sqrt(@max(0.0, val));
}

pub fn evalVonMises3D(
    sxx: F,
    syy: F,
    szz: F,
    syz: F,
    sxz: F,
    sxy: F,
) F {
    const dxy = sxx - syy;
    const dyz = syy - szz;
    const dzx = szz - sxx;
    const j2 = (dxy * dxy + dyz * dyz + dzx * dzx) / 6.0 +
        sxy * sxy + syz * syz + sxz * sxz;
    return @sqrt(@max(0.0, 3.0 * j2));
}

pub fn evalHydrostatic2D(sxx: F, syy: F) F {
    return 0.5 * (sxx + syy);
}

pub fn evalHydrostatic3D(sxx: F, syy: F, szz: F) F {
    return (sxx + syy + szz) / 3.0;
}

pub fn evalTresca2D(sxx: F, syy: F, sxy: F) F {
    var p1: F = 0.0;
    var p2: F = 0.0;
    var max_s: F = 0.0;
    evalPrincipal2D(sxx, syy, sxy, &p1, &p2, &max_s);
    return max_s;
}

pub fn evalTresca3D(
    sxx: F,
    syy: F,
    szz: F,
    syz: F,
    sxz: F,
    sxy: F,
) F {
    var p1: F = 0.0;
    var p2: F = 0.0;
    var p3: F = 0.0;
    evalPrincipal3D(sxx, syy, szz, syz, sxz, sxy, &p1, &p2, &p3);
    return 0.5 * (p1 - p3);
}

pub fn transformTensorArray2D(
    raw_tensor_ptr: [*c]const F,
    num_points: usize,
    num_times: usize,
    inv_type: u32,
    out_derived_ptr: [*c]F,
) void {
    const itype: TensorInvariantType = @enumFromInt(inv_type);
    for (0..num_points) |ii| {
        for (0..num_times) |tt| {
            const idx_xx = (ii * 3 + 0) * num_times + tt;
            const idx_yy = (ii * 3 + 1) * num_times + tt;
            const idx_xy = (ii * 3 + 2) * num_times + tt;

            const sxx = raw_tensor_ptr[idx_xx];
            const syy = raw_tensor_ptr[idx_yy];
            const sxy = raw_tensor_ptr[idx_xy];

            const out_idx = ii * num_times + tt;

            switch (itype) {
                .von_mises => {
                    out_derived_ptr[out_idx] = evalVonMises2D(sxx, syy, sxy);
                },
                .principal_1 => {
                    var p1: F = 0.0;
                    var p2: F = 0.0;
                    var ms: F = 0.0;
                    evalPrincipal2D(sxx, syy, sxy, &p1, &p2, &ms);
                    out_derived_ptr[out_idx] = p1;
                },
                .principal_2 => {
                    var p1: F = 0.0;
                    var p2: F = 0.0;
                    var ms: F = 0.0;
                    evalPrincipal2D(sxx, syy, sxy, &p1, &p2, &ms);
                    out_derived_ptr[out_idx] = p2;
                },
                .tresca, .max_shear => {
                    out_derived_ptr[out_idx] = evalTresca2D(sxx, syy, sxy);
                },
                .hydrostatic => {
                    out_derived_ptr[out_idx] = evalHydrostatic2D(sxx, syy);
                },
                .invariant_i1 => {
                    out_derived_ptr[out_idx] = sxx + syy;
                },
                .invariant_i2 => {
                    out_derived_ptr[out_idx] = sxx * syy - sxy * sxy;
                },
                else => {
                    out_derived_ptr[out_idx] = evalVonMises2D(sxx, syy, sxy);
                },
            }
        }
    }
}

pub fn transformTensorArray3D(
    raw_tensor_ptr: [*c]const F,
    num_points: usize,
    num_times: usize,
    inv_type: u32,
    out_derived_ptr: [*c]F,
) void {
    const itype: TensorInvariantType = @enumFromInt(inv_type);
    for (0..num_points) |ii| {
        for (0..num_times) |tt| {
            const idx_xx = (ii * 6 + 0) * num_times + tt;
            const idx_yy = (ii * 6 + 1) * num_times + tt;
            const idx_zz = (ii * 6 + 2) * num_times + tt;
            const idx_yz = (ii * 6 + 3) * num_times + tt;
            const idx_xz = (ii * 6 + 4) * num_times + tt;
            const idx_xy = (ii * 6 + 5) * num_times + tt;

            const sxx = raw_tensor_ptr[idx_xx];
            const syy = raw_tensor_ptr[idx_yy];
            const szz = raw_tensor_ptr[idx_zz];
            const syz = raw_tensor_ptr[idx_yz];
            const sxz = raw_tensor_ptr[idx_xz];
            const sxy = raw_tensor_ptr[idx_xy];

            const out_idx = ii * num_times + tt;

            switch (itype) {
                .von_mises => {
                    out_derived_ptr[out_idx] = evalVonMises3D(
                        sxx,
                        syy,
                        szz,
                        syz,
                        sxz,
                        sxy,
                    );
                },
                .principal_1 => {
                    var p1: F = 0.0;
                    var p2: F = 0.0;
                    var p3: F = 0.0;
                    evalPrincipal3D(
                        sxx,
                        syy,
                        szz,
                        syz,
                        sxz,
                        sxy,
                        &p1,
                        &p2,
                        &p3,
                    );
                    out_derived_ptr[out_idx] = p1;
                },
                .principal_2 => {
                    var p1: F = 0.0;
                    var p2: F = 0.0;
                    var p3: F = 0.0;
                    evalPrincipal3D(
                        sxx,
                        syy,
                        szz,
                        syz,
                        sxz,
                        sxy,
                        &p1,
                        &p2,
                        &p3,
                    );
                    out_derived_ptr[out_idx] = p2;
                },
                .principal_3 => {
                    var p1: F = 0.0;
                    var p2: F = 0.0;
                    var p3: F = 0.0;
                    evalPrincipal3D(
                        sxx,
                        syy,
                        szz,
                        syz,
                        sxz,
                        sxy,
                        &p1,
                        &p2,
                        &p3,
                    );
                    out_derived_ptr[out_idx] = p3;
                },
                .tresca, .max_shear => {
                    out_derived_ptr[out_idx] = evalTresca3D(
                        sxx,
                        syy,
                        szz,
                        syz,
                        sxz,
                        sxy,
                    );
                },
                .hydrostatic => {
                    out_derived_ptr[out_idx] = evalHydrostatic3D(
                        sxx,
                        syy,
                        szz,
                    );
                },
                .invariant_i1 => {
                    out_derived_ptr[out_idx] = sxx + syy + szz;
                },
                .invariant_i2 => {
                    out_derived_ptr[out_idx] = sxx * syy + syy * szz +
                        szz * sxx - sxy * sxy - syz * syz - sxz * sxz;
                },
                .invariant_i3 => {
                    out_derived_ptr[out_idx] = sxx * syy * szz +
                        2.0 * sxy * syz * sxz - sxx * syz * syz -
                        syy * sxz * sxz - szz * sxy * sxy;
                },
                else => {
                    out_derived_ptr[out_idx] = evalVonMises3D(
                        sxx,
                        syy,
                        szz,
                        syz,
                        sxz,
                        sxy,
                    );
                },
            }
        }
    }
}
