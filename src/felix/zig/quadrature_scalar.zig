// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("quadrature_common.zig");

const F = common.F;
const QuadRule = common.QuadRule;
const QuadSpec = common.QuadSpec;

pub fn getGaussLegendre1D(
    order: usize,
    out_nodes: []F,
    out_weights: []F,
) usize {
    switch (order) {
        1 => {
            out_nodes[0] = 0.0;
            out_weights[0] = 2.0;
            return 1;
        },
        2 => {
            const x = 0.57735026918962576451;
            out_nodes[0] = -x;
            out_nodes[1] = x;
            out_weights[0] = 1.0;
            out_weights[1] = 1.0;
            return 2;
        },
        3 => {
            const x = 0.77459666924148337704;
            out_nodes[0] = -x;
            out_nodes[1] = 0.0;
            out_nodes[2] = x;
            out_weights[0] = 0.55555555555555555556;
            out_weights[1] = 0.88888888888888888889;
            out_weights[2] = 0.55555555555555555556;
            return 3;
        },
        4 => {
            const x1 = 0.33998104358485626480;
            const x2 = 0.86113631159405257522;
            const w1 = 0.65214515486254614263;
            const w2 = 0.34785484513745385737;
            out_nodes[0] = -x2;
            out_nodes[1] = -x1;
            out_nodes[2] = x1;
            out_nodes[3] = x2;
            out_weights[0] = w2;
            out_weights[1] = w1;
            out_weights[2] = w1;
            out_weights[3] = w2;
            return 4;
        },
        5 => {
            const x1 = 0.53846931010568309104;
            const x2 = 0.90617984593866399280;
            const w0 = 0.56888888888888888889;
            const w1 = 0.47862867049936646804;
            const w2 = 0.23692688505618908751;
            out_nodes[0] = -x2;
            out_nodes[1] = -x1;
            out_nodes[2] = 0.0;
            out_nodes[3] = x1;
            out_nodes[4] = x2;
            out_weights[0] = w2;
            out_weights[1] = w1;
            out_weights[2] = w0;
            out_weights[3] = w1;
            out_weights[4] = w2;
            return 5;
        },
        6 => {
            const x1 = 0.23861918608319690863;
            const x2 = 0.66120938646626451366;
            const x3 = 0.93246951420315202781;
            const w1 = 0.46791393457269104739;
            const w2 = 0.36076157304813860757;
            const w3 = 0.17132449237917034504;
            out_nodes[0] = -x3;
            out_nodes[1] = -x2;
            out_nodes[2] = -x1;
            out_nodes[3] = x1;
            out_nodes[4] = x2;
            out_nodes[5] = x3;
            out_weights[0] = w3;
            out_weights[1] = w2;
            out_weights[2] = w1;
            out_weights[3] = w1;
            out_weights[4] = w2;
            out_weights[5] = w3;
            return 6;
        },
        else => {
            const n_f = @as(F, @floatFromInt(order));
            for (0..order) |ii| {
                const i_f = @as(F, @floatFromInt(ii));
                out_nodes[ii] = -1.0 + (2.0 * i_f + 1.0) / n_f;
                out_weights[ii] = 2.0 / n_f;
            }
            return order;
        },
    }
}

pub fn getMidpoint1D(
    divisions: usize,
    out_nodes: []F,
    out_weights: []F,
) usize {
    const divs = if (divisions > 0) divisions else 1;
    const dx = 2.0 / @as(F, @floatFromInt(divs));
    for (0..divs) |ii| {
        const i_f = @as(F, @floatFromInt(ii));
        out_nodes[ii] = -1.0 + (i_f + 0.5) * dx;
        out_weights[ii] = dx;
    }
    return divs;
}

pub fn getTrapezoidal1D(
    divisions: usize,
    out_nodes: []F,
    out_weights: []F,
) usize {
    const divs = if (divisions > 0) divisions else 1;
    const num_pts = divs + 1;
    const dx = 2.0 / @as(F, @floatFromInt(divs));
    for (0..num_pts) |ii| {
        const i_f = @as(F, @floatFromInt(ii));
        out_nodes[ii] = -1.0 + i_f * dx;
        if (ii == 0 or ii == divs) {
            out_weights[ii] = 0.5 * dx;
        } else {
            out_weights[ii] = dx;
        }
    }
    return num_pts;
}

pub fn getSimpson1D(
    divisions: usize,
    out_nodes: []F,
    out_weights: []F,
) usize {
    var divs = if (divisions > 0) divisions else 2;
    if (divs % 2 != 0) divs += 1;
    const num_pts = divs + 1;
    const dx = 2.0 / @as(F, @floatFromInt(divs));
    for (0..num_pts) |ii| {
        const i_f = @as(F, @floatFromInt(ii));
        out_nodes[ii] = -1.0 + i_f * dx;
        if (ii == 0 or ii == divs) {
            out_weights[ii] = (1.0 / 3.0) * dx;
        } else if (ii % 2 == 1) {
            out_weights[ii] = (4.0 / 3.0) * dx;
        } else {
            out_weights[ii] = (2.0 / 3.0) * dx;
        }
    }
    return num_pts;
}

pub fn generateNodesAndWeightsND(
    spec: *const QuadSpec,
    out_nodes_ptr: [*c]F,
    out_weights_ptr: [*c]F,
) usize {
    const rule: QuadRule = @enumFromInt(spec.rule);
    const order = if (spec.order > 0) spec.order else 2;
    const dims = if (spec.dims > 0) spec.dims else 1;

    var n_1d: [64]F = undefined;
    var w_1d: [64]F = undefined;

    const count_1d = switch (rule) {
        .gauss_legendre => getGaussLegendre1D(order, &n_1d, &w_1d),
        .midpoint => getMidpoint1D(order, &n_1d, &w_1d),
        .trapezoidal => getTrapezoidal1D(order, &n_1d, &w_1d),
        .simpson => getSimpson1D(order, &n_1d, &w_1d),
        .monte_carlo => getMidpoint1D(order, &n_1d, &w_1d),
    };

    if (dims == 1) {
        for (0..count_1d) |ii| {
            out_nodes_ptr[ii] = n_1d[ii];
            out_weights_ptr[ii] = w_1d[ii];
        }
        return count_1d;
    }

    if (dims == 2) {
        var pt_idx: usize = 0;
        for (0..count_1d) |ii| {
            for (0..count_1d) |jj| {
                out_nodes_ptr[pt_idx * 2] = n_1d[ii];
                out_nodes_ptr[pt_idx * 2 + 1] = n_1d[jj];
                out_weights_ptr[pt_idx] = w_1d[ii] * w_1d[jj];
                pt_idx += 1;
            }
        }
        return pt_idx;
    }

    if (dims == 3) {
        var pt_idx: usize = 0;
        for (0..count_1d) |ii| {
            for (0..count_1d) |jj| {
                for (0..count_1d) |kk| {
                    out_nodes_ptr[pt_idx * 3] = n_1d[ii];
                    out_nodes_ptr[pt_idx * 3 + 1] = n_1d[jj];
                    out_nodes_ptr[pt_idx * 3 + 2] = n_1d[kk];
                    out_weights_ptr[pt_idx] = w_1d[ii] * w_1d[jj] * w_1d[kk];
                    pt_idx += 1;
                }
            }
        }
        return pt_idx;
    }

    return 0;
}
