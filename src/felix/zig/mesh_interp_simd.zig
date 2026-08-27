// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("mesh_interp_common.zig");
const scalar = @import("mesh_interp_scalar.zig");

const F = common.F;
const VecSF = common.VecSF;
const SimdWidth = common.SimdWidth;

pub fn sampleCachedFEPointOverTimeSIMD(
    node_count: usize,
    weights: []const F,
    node_indices: []const usize,
    field_data: [*c]const F,
    num_nodes: usize,
    num_components: usize,
    num_times: usize,
    comp_idx: usize,
    time_offset: usize,
) VecSF {
    _ = num_nodes;
    var accum: VecSF = @splat(0.0);
    for (0..node_count) |nn| {
        const node_id = node_indices[nn];
        const base_offset = (node_id * num_components + comp_idx) * num_times + time_offset;
        const time_packet: VecSF = field_data[base_offset..][0..SimdWidth].*;
        const weight_vec: VecSF = @splat(weights[nn]);
        accum += weight_vec * time_packet;
    }
    return accum;
}

pub const interpTimeLinear = scalar.interpTimeLinear;
pub const sampleCachedFEPoint = scalar.sampleCachedFEPoint;

// --------------------------------------------------------------------------------------
// Tests: Parity Verification
// --------------------------------------------------------------------------------------

test "mesh_interp SIMD-over-time vs scalar parity" {
    const node_count = 4;
    const weights = [_]F{ 0.25, 0.25, 0.25, 0.25 };
    const node_indices = [_]usize{ 0, 1, 2, 3 };
    const num_nodes = 4;
    const num_components = 1;
    const num_times = SimdWidth;

    var field_data: [num_nodes * num_components * num_times]F = undefined;
    for (0..num_nodes) |nn| {
        for (0..num_times) |tt| {
            field_data[nn * num_times + tt] = @as(F, @floatFromInt(nn * 10 + tt));
        }
    }

    const simd_res = sampleCachedFEPointOverTimeSIMD(
        node_count,
        &weights,
        &node_indices,
        &field_data,
        num_nodes,
        num_components,
        num_times,
        0,
        0,
    );
    const simd_arr: [SimdWidth]F = simd_res;

    for (0..num_times) |tt| {
        const scal_val = scalar.sampleCachedFEPoint(
            node_count,
            &weights,
            &node_indices,
            &field_data,
            num_nodes,
            num_components,
            num_times,
            0,
            tt,
        );
        try std.testing.expectApproxEqAbs(scal_val, simd_arr[tt], 1e-12);
    }
}
