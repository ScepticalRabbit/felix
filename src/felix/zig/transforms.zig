// --------------------------------------------------------------------------------------
// Felix: A High Performance Sensor Simulation Core
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const F: type = f64;

// --------------------------------------------------------------------------------------
// Public Entry Points
// --------------------------------------------------------------------------------------

pub fn transformVector2D(
    rot_mat_22: *const [4]F,
    vx: F,
    vy: F,
    out_vx: *F,
    out_vy: *F,
) void {
    const r00 = rot_mat_22[0];
    const r01 = rot_mat_22[1];
    const r10 = rot_mat_22[2];
    const r11 = rot_mat_22[3];

    out_vx.* = r00 * vx + r01 * vy;
    out_vy.* = r10 * vx + r11 * vy;
}

pub fn transformVector3D(
    rot_mat_33: *const [9]F,
    vx: F,
    vy: F,
    vz: F,
    out_vx: *F,
    out_vy: *F,
    out_vz: *F,
) void {
    const r00 = rot_mat_33[0];
    const r01 = rot_mat_33[1];
    const r02 = rot_mat_33[2];
    const r10 = rot_mat_33[3];
    const r11 = rot_mat_33[4];
    const r12 = rot_mat_33[5];
    const r20 = rot_mat_33[6];
    const r21 = rot_mat_33[7];
    const r22 = rot_mat_33[8];

    out_vx.* = r00 * vx + r01 * vy + r02 * vz;
    out_vy.* = r10 * vx + r11 * vy + r12 * vz;
    out_vz.* = r20 * vx + r21 * vy + r22 * vz;
}

pub fn transformTensor2D(
    rot_mat_22: *const [4]F,
    s_xx: F,
    s_yy: F,
    s_xy: F,
    out_s_xx: *F,
    out_s_yy: *F,
    out_s_xy: *F,
) void {
    const r00 = rot_mat_22[0];
    const r01 = rot_mat_22[1];
    const r10 = rot_mat_22[2];
    const r11 = rot_mat_22[3];

    out_s_xx.* = r00 * r00 * s_xx + 2.0 * r00 * r01 * s_xy + r01 * r01 * s_yy;
    out_s_yy.* = r10 * r10 * s_xx + 2.0 * r10 * r11 * s_xy + r11 * r11 * s_yy;
    out_s_xy.* = r00 * r10 * s_xx + (r00 * r11 + r01 * r10) * s_xy + r01 * r11 * s_yy;
}

pub fn transformTensor3D(
    rot_mat_33: *const [9]F,
    in_tensor_6: *const [6]F,
    out_tensor_6: *[6]F,
) void {
    const t_xx = in_tensor_6[0];
    const t_yy = in_tensor_6[1];
    const t_zz = in_tensor_6[2];
    const t_xy = in_tensor_6[3];
    const t_xz = in_tensor_6[4];
    const t_yz = in_tensor_6[5];

    var full_t: [3][3]F = undefined;
    full_t[0][0] = t_xx;
    full_t[0][1] = t_xy;
    full_t[0][2] = t_xz;
    full_t[1][0] = t_xy;
    full_t[1][1] = t_yy;
    full_t[1][2] = t_yz;
    full_t[2][0] = t_xz;
    full_t[2][1] = t_yz;
    full_t[2][2] = t_zz;

    var r_mat: [3][3]F = undefined;
    r_mat[0][0] = rot_mat_33[0];
    r_mat[0][1] = rot_mat_33[1];
    r_mat[0][2] = rot_mat_33[2];
    r_mat[1][0] = rot_mat_33[3];
    r_mat[1][1] = rot_mat_33[4];
    r_mat[1][2] = rot_mat_33[5];
    r_mat[2][0] = rot_mat_33[6];
    r_mat[2][1] = rot_mat_33[7];
    r_mat[2][2] = rot_mat_33[8];

    var temp_m: [3][3]F = undefined;
    for (0..3) |ii| {
        for (0..3) |jj| {
            var sum_val: F = 0.0;
            for (0..3) |kk| {
                sum_val += full_t[ii][kk] * r_mat[jj][kk];
            }
            temp_m[ii][jj] = sum_val;
        }
    }

    var res_t: [3][3]F = undefined;
    for (0..3) |ii| {
        for (0..3) |jj| {
            var sum_val: F = 0.0;
            for (0..3) |kk| {
                sum_val += r_mat[ii][kk] * temp_m[kk][jj];
            }
            res_t[ii][jj] = sum_val;
        }
    }

    out_tensor_6[0] = res_t[0][0];
    out_tensor_6[1] = res_t[1][1];
    out_tensor_6[2] = res_t[2][2];
    out_tensor_6[3] = 0.5 * (res_t[0][1] + res_t[1][0]);
    out_tensor_6[4] = 0.5 * (res_t[0][2] + res_t[2][0]);
    out_tensor_6[5] = 0.5 * (res_t[1][2] + res_t[2][1]);
}
