// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("transforms_common.zig");
const scalar = @import("transforms_scalar.zig");

const F = common.F;
const VecSF = common.VecSF;
const SimdWidth = common.SimdWidth;

pub const Vec2Packet = common.Vec2Packet;
pub const Vec3Packet = common.Vec3Packet;
pub const Tensor2Packet = common.Tensor2Packet;
pub const Tensor3Packet = common.Tensor3Packet;

// --------------------------------------------------------------------------------------
// SIMD Packet Transforms
// --------------------------------------------------------------------------------------

pub fn transformVector2DPacket(
    rot_mat_22: *const [4]F,
    packet_in: Vec2Packet,
    out_packet: *Vec2Packet,
) void {
    const r00: VecSF = @splat(rot_mat_22[0]);
    const r01: VecSF = @splat(rot_mat_22[1]);
    const r10: VecSF = @splat(rot_mat_22[2]);
    const r11: VecSF = @splat(rot_mat_22[3]);

    out_packet.x = r00 * packet_in.x + r01 * packet_in.y;
    out_packet.y = r10 * packet_in.x + r11 * packet_in.y;
}

pub fn transformVector3DPacket(
    rot_mat_33: *const [9]F,
    packet_in: Vec3Packet,
    out_packet: *Vec3Packet,
) void {
    const r00: VecSF = @splat(rot_mat_33[0]);
    const r01: VecSF = @splat(rot_mat_33[1]);
    const r02: VecSF = @splat(rot_mat_33[2]);
    const r10: VecSF = @splat(rot_mat_33[3]);
    const r11: VecSF = @splat(rot_mat_33[4]);
    const r12: VecSF = @splat(rot_mat_33[5]);
    const r20: VecSF = @splat(rot_mat_33[6]);
    const r21: VecSF = @splat(rot_mat_33[7]);
    const r22: VecSF = @splat(rot_mat_33[8]);

    out_packet.x = r00 * packet_in.x + r01 * packet_in.y + r02 * packet_in.z;
    out_packet.y = r10 * packet_in.x + r11 * packet_in.y + r12 * packet_in.z;
    out_packet.z = r20 * packet_in.x + r21 * packet_in.y + r22 * packet_in.z;
}

pub fn transformTensor2DPacket(
    rot_mat_22: *const [4]F,
    packet_in: Tensor2Packet,
    out_packet: *Tensor2Packet,
) void {
    const r00: VecSF = @splat(rot_mat_22[0]);
    const r01: VecSF = @splat(rot_mat_22[1]);
    const r10: VecSF = @splat(rot_mat_22[2]);
    const r11: VecSF = @splat(rot_mat_22[3]);
    const two: VecSF = @splat(2.0);

    const s_xx = packet_in.xx;
    const s_yy = packet_in.yy;
    const s_xy = packet_in.xy;

    out_packet.xx = r00 * r00 * s_xx + two * r00 * r01 * s_xy + r01 * r01 * s_yy;
    out_packet.yy = r10 * r10 * s_xx + two * r10 * r11 * s_xy + r11 * r11 * s_yy;
    out_packet.xy = r00 * r10 * s_xx + (r00 * r11 + r01 * r10) * s_xy + r01 * r11 * s_yy;
}

pub fn transformTensor3DPacket(
    rot_mat_33: *const [9]F,
    packet_in: Tensor3Packet,
    out_packet: *Tensor3Packet,
) void {
    const r00: VecSF = @splat(rot_mat_33[0]);
    const r01: VecSF = @splat(rot_mat_33[1]);
    const r02: VecSF = @splat(rot_mat_33[2]);
    const r10: VecSF = @splat(rot_mat_33[3]);
    const r11: VecSF = @splat(rot_mat_33[4]);
    const r12: VecSF = @splat(rot_mat_33[5]);
    const r20: VecSF = @splat(rot_mat_33[6]);
    const r21: VecSF = @splat(rot_mat_33[7]);
    const r22: VecSF = @splat(rot_mat_33[8]);

    const s_xx = packet_in.xx;
    const s_yy = packet_in.yy;
    const s_zz = packet_in.zz;
    const s_xy = packet_in.xy;
    const s_xz = packet_in.xz;
    const s_yz = packet_in.yz;

    const ts00 = r00 * s_xx + r01 * s_xy + r02 * s_xz;
    const ts01 = r00 * s_xy + r01 * s_yy + r02 * s_yz;
    const ts02 = r00 * s_xz + r01 * s_yz + r02 * s_zz;

    const ts10 = r10 * s_xx + r11 * s_xy + r12 * s_xz;
    const ts11 = r10 * s_xy + r11 * s_yy + r12 * s_yz;
    const ts12 = r10 * s_xz + r11 * s_yz + r12 * s_zz;

    const ts20 = r20 * s_xx + r21 * s_xy + r22 * s_xz;
    const ts21 = r20 * s_xy + r21 * s_yy + r22 * s_yz;
    const ts22 = r20 * s_xz + r21 * s_yz + r22 * s_zz;

    out_packet.xx = ts00 * r00 + ts01 * r01 + ts02 * r02;
    out_packet.yy = ts10 * r10 + ts11 * r11 + ts12 * r12;
    out_packet.zz = ts20 * r20 + ts21 * r21 + ts22 * r22;
    out_packet.xy = ts00 * r10 + ts01 * r11 + ts02 * r12;
    out_packet.xz = ts00 * r20 + ts01 * r21 + ts02 * r22;
    out_packet.yz = ts10 * r20 + ts11 * r21 + ts12 * r22;
}

// --------------------------------------------------------------------------------------
// Re-export Scalar API for Universal Compatibility
// --------------------------------------------------------------------------------------

pub const transformVector2D = scalar.transformVector2D;
pub const transformVector3D = scalar.transformVector3D;
pub const transformTensor2D = scalar.transformTensor2D;
pub const transformTensor3D = scalar.transformTensor3D;

pub fn transformVector3DBatch(
    rot_mat_33: *const [9]F,
    vx: []const F,
    vy: []const F,
    vz: []const F,
    out_vx: []F,
    out_vy: []F,
    out_vz: []F,
) void {
    const total_len = vx.len;
    var offset: usize = 0;

    while (offset + SimdWidth <= total_len) : (offset += SimdWidth) {
        const in_packet = Vec3Packet{
            .x = vx[offset..][0..SimdWidth].*,
            .y = vy[offset..][0..SimdWidth].*,
            .z = vz[offset..][0..SimdWidth].*,
        };
        var out_packet: Vec3Packet = undefined;
        transformVector3DPacket(rot_mat_33, in_packet, &out_packet);
        out_vx[offset..][0..SimdWidth].* = out_packet.x;
        out_vy[offset..][0..SimdWidth].* = out_packet.y;
        out_vz[offset..][0..SimdWidth].* = out_packet.z;
    }

    while (offset < total_len) : (offset += 1) {
        transformVector3D(
            rot_mat_33,
            vx[offset],
            vy[offset],
            vz[offset],
            &out_vx[offset],
            &out_vy[offset],
            &out_vz[offset],
        );
    }
}

// --------------------------------------------------------------------------------------
// Tests: Direct SIMD vs Scalar Parity
// --------------------------------------------------------------------------------------

test "transforms SIMD vs scalar parity" {
    const rot_mat_33 = [9]F{
        0.8660254037844387, -0.5,               0.0,
        0.5,                0.8660254037844387, 0.0,
        0.0,                0.0,                1.0,
    };

    var vx_arr: [SimdWidth]F = undefined;
    var vy_arr: [SimdWidth]F = undefined;
    var vz_arr: [SimdWidth]F = undefined;

    for (0..SimdWidth) |ii| {
        vx_arr[ii] = @as(F, @floatFromInt(ii)) * 1.5 + 2.0;
        vy_arr[ii] = @as(F, @floatFromInt(ii)) * -0.7 + 1.0;
        vz_arr[ii] = @as(F, @floatFromInt(ii)) * 3.14;
    }

    const in_packet = Vec3Packet{
        .x = vx_arr,
        .y = vy_arr,
        .z = vz_arr,
    };
    var simd_out: Vec3Packet = undefined;
    transformVector3DPacket(&rot_mat_33, in_packet, &simd_out);

    const out_x_arr: [SimdWidth]F = simd_out.x;
    const out_y_arr: [SimdWidth]F = simd_out.y;
    const out_z_arr: [SimdWidth]F = simd_out.z;

    for (0..SimdWidth) |ii| {
        var scal_x: F = 0.0;
        var scal_y: F = 0.0;
        var scal_z: F = 0.0;
        scalar.transformVector3D(
            &rot_mat_33,
            vx_arr[ii],
            vy_arr[ii],
            vz_arr[ii],
            &scal_x,
            &scal_y,
            &scal_z,
        );
        try std.testing.expectApproxEqAbs(scal_x, out_x_arr[ii], 1e-12);
        try std.testing.expectApproxEqAbs(scal_y, out_y_arr[ii], 1e-12);
        try std.testing.expectApproxEqAbs(scal_z, out_z_arr[ii], 1e-12);
    }
}
