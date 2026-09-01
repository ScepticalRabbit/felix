// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const F: type = f64;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const max_elem_nodes: usize = 20;

pub const ElementType = enum(u32) {
    tri3 = 0,
    quad4 = 1,
    tet4 = 2,
    hex8 = 3,
    tri6 = 4,
    quad8 = 5,
    hex20 = 6,

    pub fn nodeCount(self: ElementType) usize {
        return switch (self) {
            .tri3 => 3,
            .quad4 => 4,
            .tet4 => 4,
            .hex8 => 8,
            .tri6 => 6,
            .quad8 => 8,
            .hex20 => 20,
        };
    }

    pub fn spatialDim(self: ElementType) usize {
        return switch (self) {
            .tri3, .quad4, .tri6, .quad8 => 2,
            .tet4, .hex8, .hex20 => 3,
        };
    }
};

pub const InverseResult = struct {
    inside: bool,
    weights: [max_elem_nodes]F,
    node_count: usize,
};

// --------------------------------------------------------------------------------------
// Shape Function Evaluators
// --------------------------------------------------------------------------------------

pub fn shapeTri3(
    xi_coord: F,
    eta_coord: F,
    out_weights: *[3]F,
) void {
    out_weights[0] = 1.0 - xi_coord - eta_coord;
    out_weights[1] = xi_coord;
    out_weights[2] = eta_coord;
}

pub fn shapeQuad4(
    xi_coord: F,
    eta_coord: F,
    out_weights: *[4]F,
    out_deriv_xi: ?*[4]F,
    out_deriv_eta: ?*[4]F,
) void {
    const Vec4 = @Vector(4, F);
    const xi_signs: Vec4 = .{ -1.0, 1.0, 1.0, -1.0 };
    const eta_signs: Vec4 = .{ -1.0, -1.0, 1.0, 1.0 };

    const one: Vec4 = @splat(1.0);
    const quarter: Vec4 = @splat(0.25);
    const xi_vec: Vec4 = @splat(xi_coord);
    const eta_vec: Vec4 = @splat(eta_coord);

    const factor_xi = one + xi_signs * xi_vec;
    const factor_eta = one + eta_signs * eta_vec;

    out_weights.* = quarter * factor_xi * factor_eta;

    if (out_deriv_xi) |deriv_xi| {
        deriv_xi.* = quarter * xi_signs * factor_eta;
    }
    if (out_deriv_eta) |deriv_eta| {
        deriv_eta.* = quarter * eta_signs * factor_xi;
    }
}

pub fn shapeQuad8(
    xi_coord: F,
    eta_coord: F,
    out_weights: *[8]F,
    out_deriv_xi: ?*[8]F,
    out_deriv_eta: ?*[8]F,
) void {
    const xi_c = [4]F{ -1.0, 1.0, 1.0, -1.0 };
    const eta_c = [4]F{ -1.0, -1.0, 1.0, 1.0 };

    for (0..4) |ii| {
        const f_xi = 1.0 + xi_c[ii] * xi_coord;
        const f_eta = 1.0 + eta_c[ii] * eta_coord;
        const term = xi_c[ii] * xi_coord + eta_c[ii] * eta_coord - 1.0;
        out_weights[ii] = 0.25 * f_xi * f_eta * term;

        if (out_deriv_xi) |deriv_xi| {
            deriv_xi[ii] = 0.25 * xi_c[ii] * f_eta * term + 0.25 * f_xi * f_eta * xi_c[ii];
        }
        if (out_deriv_eta) |deriv_eta| {
            deriv_eta[ii] = 0.25 * f_xi * eta_c[ii] * term + 0.25 * f_xi * f_eta * eta_c[ii];
        }
    }

    out_weights[4] = 0.5 * (1.0 - xi_coord * xi_coord) * (1.0 - eta_coord);
    out_weights[5] = 0.5 * (1.0 + xi_coord) * (1.0 - eta_coord * eta_coord);
    out_weights[6] = 0.5 * (1.0 - xi_coord * xi_coord) * (1.0 + eta_coord);
    out_weights[7] = 0.5 * (1.0 - xi_coord) * (1.0 - eta_coord * eta_coord);

    if (out_deriv_xi) |deriv_xi| {
        deriv_xi[4] = -xi_coord * (1.0 - eta_coord);
        deriv_xi[5] = 0.5 * (1.0 - eta_coord * eta_coord);
        deriv_xi[6] = -xi_coord * (1.0 + eta_coord);
        deriv_xi[7] = -0.5 * (1.0 - eta_coord * eta_coord);
    }
    if (out_deriv_eta) |deriv_eta| {
        deriv_eta[4] = -0.5 * (1.0 - xi_coord * xi_coord);
        deriv_eta[5] = -(1.0 + xi_coord) * eta_coord;
        deriv_eta[6] = 0.5 * (1.0 - xi_coord * xi_coord);
        deriv_eta[7] = -(1.0 - xi_coord) * eta_coord;
    }
}

pub fn shapeTet4(
    xi_coord: F,
    eta_coord: F,
    zeta_coord: F,
    out_weights: *[4]F,
) void {
    out_weights[0] = 1.0 - xi_coord - eta_coord - zeta_coord;
    out_weights[1] = xi_coord;
    out_weights[2] = eta_coord;
    out_weights[3] = zeta_coord;
}

pub fn shapeHex8(
    xi_coord: F,
    eta_coord: F,
    zeta_coord: F,
    out_weights: *[8]F,
    out_deriv_xi: ?*[8]F,
    out_deriv_eta: ?*[8]F,
    out_deriv_zeta: ?*[8]F,
) void {
    const Vec8 = @Vector(8, F);
    const xi_signs: Vec8 = .{ -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
    const eta_signs: Vec8 = .{ -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
    const zeta_signs: Vec8 = .{ -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };

    const one: Vec8 = @splat(1.0);
    const eighth: Vec8 = @splat(0.125);
    const xi_vec: Vec8 = @splat(xi_coord);
    const eta_vec: Vec8 = @splat(eta_coord);
    const zeta_vec: Vec8 = @splat(zeta_coord);

    const factor_xi = one + xi_signs * xi_vec;
    const factor_eta = one + eta_signs * eta_vec;
    const factor_zeta = one + zeta_signs * zeta_vec;

    const factor_eta_zeta = factor_eta * factor_zeta;
    const factor_xi_zeta = factor_xi * factor_zeta;
    const factor_xi_eta = factor_xi * factor_eta;

    out_weights.* = eighth * factor_xi * factor_eta_zeta;

    if (out_deriv_xi) |deriv_xi| {
        deriv_xi.* = eighth * xi_signs * factor_eta_zeta;
    }
    if (out_deriv_eta) |deriv_eta| {
        deriv_eta.* = eighth * eta_signs * factor_xi_zeta;
    }
    if (out_deriv_zeta) |deriv_zeta| {
        deriv_zeta.* = eighth * zeta_signs * factor_xi_eta;
    }
}

pub fn shapeHex20(
    xi_coord: F,
    eta_coord: F,
    zeta_coord: F,
    out_weights: *[20]F,
    out_deriv_xi: ?*[20]F,
    out_deriv_eta: ?*[20]F,
    out_deriv_zeta: ?*[20]F,
) void {
    // 8 corner nodes:
    const xi_c = [8]F{ -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
    const eta_c = [8]F{ -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
    const zeta_c = [8]F{ -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };

    for (0..8) |ii| {
        const f_xi = 1.0 + xi_c[ii] * xi_coord;
        const f_eta = 1.0 + eta_c[ii] * eta_coord;
        const f_zeta = 1.0 + zeta_c[ii] * zeta_coord;
        const term = xi_c[ii] * xi_coord + eta_c[ii] * eta_coord +
            zeta_c[ii] * zeta_coord - 2.0;

        out_weights[ii] = 0.125 * f_xi * f_eta * f_zeta * term;

        if (out_deriv_xi) |deriv_xi| {
            deriv_xi[ii] = 0.125 * xi_c[ii] * f_eta * f_zeta * term +
                0.125 * f_xi * f_eta * f_zeta * xi_c[ii];
        }
        if (out_deriv_eta) |deriv_eta| {
            deriv_eta[ii] = 0.125 * f_xi * eta_c[ii] * f_zeta * term +
                0.125 * f_xi * f_eta * f_zeta * eta_c[ii];
        }
        if (out_deriv_zeta) |deriv_zeta| {
            deriv_zeta[ii] = 0.125 * f_xi * f_eta * zeta_c[ii] * term +
                0.125 * f_xi * f_eta * f_zeta * zeta_c[ii];
        }
    }

    // 12 mid-edge nodes in VTK order:
    // 8..11 (bottom edges z=-1): (0,-1,-1), (1,0,-1), (0,1,-1), (-1,0,-1)
    // 12..15 (top edges z=1):    (0,-1,1), (1,0,1), (0,1,1), (-1,0,1)
    // 16..19 (vertical edges):   (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0)
    const mid_nodes = [12][3]F{
        .{ 0.0, -1.0, -1.0 }, .{ 1.0, 0.0, -1.0 }, .{ 0.0, 1.0, -1.0 }, .{ -1.0, 0.0, -1.0 },
        .{ 0.0, -1.0, 1.0 },  .{ 1.0, 0.0, 1.0 },  .{ 0.0, 1.0, 1.0 },  .{ -1.0, 0.0, 1.0 },
        .{ -1.0, -1.0, 0.0 }, .{ 1.0, -1.0, 0.0 }, .{ 1.0, 1.0, 0.0 },  .{ -1.0, 1.0, 0.0 },
    };

    for (0..12) |ii| {
        const idx = 8 + ii;
        const mx = mid_nodes[ii][0];
        const my = mid_nodes[ii][1];
        const mz = mid_nodes[ii][2];

        if (mx == 0.0) {
            // Edge along xi: (1 - xi^2) * (1 + eta_0*eta) * (1 + zeta_0*zeta) / 4
            const f_xi = 1.0 - xi_coord * xi_coord;
            const f_eta = 1.0 + my * eta_coord;
            const f_zeta = 1.0 + mz * zeta_coord;
            out_weights[idx] = 0.25 * f_xi * f_eta * f_zeta;

            if (out_deriv_xi) |deriv_xi| deriv_xi[idx] = -0.5 * xi_coord * f_eta * f_zeta;
            if (out_deriv_eta) |deriv_eta| deriv_eta[idx] = 0.25 * f_xi * my * f_zeta;
            if (out_deriv_zeta) |deriv_zeta| {
                deriv_zeta[idx] = 0.25 * f_xi * f_eta * mz;
            }
        } else if (my == 0.0) {
            // Edge along eta: (1 + xi_0*xi) * (1 - eta^2) * (1 + zeta_0*zeta) / 4
            const f_xi = 1.0 + mx * xi_coord;
            const f_eta = 1.0 - eta_coord * eta_coord;
            const f_zeta = 1.0 + mz * zeta_coord;
            out_weights[idx] = 0.25 * f_xi * f_eta * f_zeta;

            if (out_deriv_xi) |deriv_xi| deriv_xi[idx] = 0.25 * mx * f_eta * f_zeta;
            if (out_deriv_eta) |deriv_eta| deriv_eta[idx] = -0.5 * f_xi * eta_coord * f_zeta;
            if (out_deriv_zeta) |deriv_zeta| {
                deriv_zeta[idx] = 0.25 * f_xi * f_eta * mz;
            }
        } else {
            // Edge along zeta: (1 + xi_0*xi) * (1 + eta_0*eta) * (1 - zeta^2) / 4
            const f_xi = 1.0 + mx * xi_coord;
            const f_eta = 1.0 + my * eta_coord;
            const f_zeta = 1.0 - zeta_coord * zeta_coord;
            out_weights[idx] = 0.25 * f_xi * f_eta * f_zeta;

            if (out_deriv_xi) |deriv_xi| deriv_xi[idx] = 0.25 * mx * f_eta * f_zeta;
            if (out_deriv_eta) |deriv_eta| deriv_eta[idx] = 0.25 * f_xi * my * f_zeta;
            if (out_deriv_zeta) |deriv_zeta| {
                deriv_zeta[idx] = -0.5 * f_xi * f_eta * zeta_coord;
            }
        }
    }
}

// --------------------------------------------------------------------------------------
// Inverse Coordinate Solvers (Parametric Inversion & Inside Check)
// --------------------------------------------------------------------------------------

pub fn invertTri3(
    node_x: *const [3]F,
    node_y: *const [3]F,
    targ_x: F,
    targ_y: F,
    tol_eps: F,
    out_result: *InverseResult,
) void {
    const x0 = node_x[0];
    const y0 = node_y[0];
    const dx1 = node_x[1] - x0;
    const dy1 = node_y[1] - y0;
    const dx2 = node_x[2] - x0;
    const dy2 = node_y[2] - y0;

    const det_jac = dx1 * dy2 - dx2 * dy1;
    if (@abs(det_jac) < 1e-14) {
        out_result.inside = false;
        return;
    }

    const rx = targ_x - x0;
    const ry = targ_y - y0;

    const xi_val = (rx * dy2 - ry * dx2) / det_jac;
    const eta_val = (dx1 * ry - dy1 * rx) / det_jac;

    if (xi_val >= -tol_eps and eta_val >= -tol_eps and (xi_val + eta_val) <= (1.0 + tol_eps)) {
        out_result.inside = true;
        out_result.node_count = 3;
        shapeTri3(xi_val, eta_val, out_result.weights[0..3]);
    } else {
        out_result.inside = false;
    }
}

pub fn invertQuad4(
    node_x: *const [4]F,
    node_y: *const [4]F,
    targ_x: F,
    targ_y: F,
    tol_eps: F,
    out_result: *InverseResult,
) void {
    const Vec4 = @Vector(4, F);
    const nx: Vec4 = node_x.*;
    const ny: Vec4 = node_y.*;

    var xi_val: F = 0.0;
    var eta_val: F = 0.0;
    var weights: [4]F = undefined;
    var deriv_xi: [4]F = undefined;
    var deriv_eta: [4]F = undefined;

    const max_iters: usize = 25;
    var converged: bool = false;

    for (0..max_iters) |_| {
        shapeQuad4(xi_val, eta_val, &weights, &deriv_xi, &deriv_eta);

        const w_vec: Vec4 = weights;
        const dxi_vec: Vec4 = deriv_xi;
        const deta_vec: Vec4 = deriv_eta;

        const calc_x = @reduce(.Add, w_vec * nx);
        const calc_y = @reduce(.Add, w_vec * ny);
        const jac_11 = @reduce(.Add, dxi_vec * nx);
        const jac_12 = @reduce(.Add, deta_vec * nx);
        const jac_21 = @reduce(.Add, dxi_vec * ny);
        const jac_22 = @reduce(.Add, deta_vec * ny);

        const resid_x = targ_x - calc_x;
        const resid_y = targ_y - calc_y;

        if (@abs(resid_x) < 1e-11 and @abs(resid_y) < 1e-11) {
            converged = true;
            break;
        }

        const det_jac = jac_11 * jac_22 - jac_12 * jac_21;
        if (@abs(det_jac) < 1e-14) {
            break;
        }

        const delta_xi = (resid_x * jac_22 - resid_y * jac_12) / det_jac;
        const delta_eta = (jac_11 * resid_y - jac_21 * resid_x) / det_jac;

        xi_val += delta_xi;
        eta_val += delta_eta;

        if (@abs(delta_xi) < 1e-12 and @abs(delta_eta) < 1e-12) {
            converged = true;
            break;
        }
    }

    if (converged and
        xi_val >= (-1.0 - tol_eps) and xi_val <= (1.0 + tol_eps) and
        eta_val >= (-1.0 - tol_eps) and eta_val <= (1.0 + tol_eps))
    {
        out_result.inside = true;
        out_result.node_count = 4;
        shapeQuad4(xi_val, eta_val, out_result.weights[0..4], null, null);
    } else {
        out_result.inside = false;
    }
}

pub fn invertQuad8(
    node_x: *const [8]F,
    node_y: *const [8]F,
    targ_x: F,
    targ_y: F,
    tol_eps: F,
    out_result: *InverseResult,
) void {
    var xi_val: F = 0.0;
    var eta_val: F = 0.0;
    var weights: [8]F = undefined;
    var deriv_xi: [8]F = undefined;
    var deriv_eta: [8]F = undefined;

    const max_iters: usize = 30;
    var converged: bool = false;

    for (0..max_iters) |_| {
        shapeQuad8(xi_val, eta_val, &weights, &deriv_xi, &deriv_eta);

        var calc_x: F = 0.0;
        var calc_y: F = 0.0;
        var jac_11: F = 0.0;
        var jac_12: F = 0.0;
        var jac_21: F = 0.0;
        var jac_22: F = 0.0;

        for (0..8) |nn| {
            calc_x += weights[nn] * node_x[nn];
            calc_y += weights[nn] * node_y[nn];
            jac_11 += deriv_xi[nn] * node_x[nn];
            jac_12 += deriv_eta[nn] * node_x[nn];
            jac_21 += deriv_xi[nn] * node_y[nn];
            jac_22 += deriv_eta[nn] * node_y[nn];
        }

        const resid_x = targ_x - calc_x;
        const resid_y = targ_y - calc_y;

        if (@abs(resid_x) < 1e-11 and @abs(resid_y) < 1e-11) {
            converged = true;
            break;
        }

        const det_jac = jac_11 * jac_22 - jac_12 * jac_21;
        if (@abs(det_jac) < 1e-14) {
            break;
        }

        const delta_xi = (resid_x * jac_22 - resid_y * jac_12) / det_jac;
        const delta_eta = (jac_11 * resid_y - jac_21 * resid_x) / det_jac;

        xi_val += delta_xi;
        eta_val += delta_eta;

        if (@abs(delta_xi) < 1e-12 and @abs(delta_eta) < 1e-12) {
            converged = true;
            break;
        }
    }

    if (converged and
        xi_val >= (-1.0 - tol_eps) and xi_val <= (1.0 + tol_eps) and
        eta_val >= (-1.0 - tol_eps) and eta_val <= (1.0 + tol_eps))
    {
        out_result.inside = true;
        out_result.node_count = 8;
        shapeQuad8(xi_val, eta_val, out_result.weights[0..8], null, null);
    } else {
        out_result.inside = false;
    }
}

pub fn invertTet4(
    node_x: *const [4]F,
    node_y: *const [4]F,
    node_z: *const [4]F,
    targ_x: F,
    targ_y: F,
    targ_z: F,
    tol_eps: F,
    out_result: *InverseResult,
) void {
    const x0 = node_x[0];
    const y0 = node_y[0];
    const z0 = node_z[0];

    const j11 = node_x[1] - x0;
    const j21 = node_y[1] - y0;
    const j31 = node_z[1] - z0;

    const j12 = node_x[2] - x0;
    const j22 = node_y[2] - y0;
    const j32 = node_z[2] - z0;

    const j13 = node_x[3] - x0;
    const j23 = node_y[3] - y0;
    const j33 = node_z[3] - z0;

    const c11 = j22 * j33 - j23 * j32;
    const c12 = j23 * j31 - j21 * j33;
    const c13 = j21 * j32 - j22 * j31;

    const c21 = j13 * j32 - j12 * j33;
    const c22 = j11 * j33 - j13 * j31;
    const c23 = j12 * j31 - j11 * j32;

    const c31 = j12 * j23 - j13 * j22;
    const c32 = j13 * j21 - j11 * j23;
    const c33 = j11 * j22 - j12 * j21;

    const det_jac = j11 * c11 + j12 * c12 + j13 * c13;

    if (@abs(det_jac) < 1e-14) {
        out_result.inside = false;
        return;
    }

    const rx = targ_x - x0;
    const ry = targ_y - y0;
    const rz = targ_z - z0;

    const inv_det = 1.0 / det_jac;

    const xi_val = inv_det * (c11 * rx + c21 * ry + c31 * rz);
    const eta_val = inv_det * (c12 * rx + c22 * ry + c32 * rz);
    const zeta_val = inv_det * (c13 * rx + c23 * ry + c33 * rz);

    if (xi_val >= -tol_eps and
        eta_val >= -tol_eps and
        zeta_val >= -tol_eps and
        (xi_val + eta_val + zeta_val) <= (1.0 + tol_eps))
    {
        out_result.inside = true;
        out_result.node_count = 4;
        shapeTet4(xi_val, eta_val, zeta_val, out_result.weights[0..4]);
    } else {
        out_result.inside = false;
    }
}

pub fn invertHex8(
    node_x: *const [8]F,
    node_y: *const [8]F,
    node_z: *const [8]F,
    targ_x: F,
    targ_y: F,
    targ_z: F,
    tol_eps: F,
    out_result: *InverseResult,
) void {
    const Vec8 = @Vector(8, F);
    const nx: Vec8 = node_x.*;
    const ny: Vec8 = node_y.*;
    const nz: Vec8 = node_z.*;

    var xi_val: F = 0.0;
    var eta_val: F = 0.0;
    var zeta_val: F = 0.0;

    var weights: [8]F = undefined;
    var deriv_xi: [8]F = undefined;
    var deriv_eta: [8]F = undefined;
    var deriv_zeta: [8]F = undefined;

    const max_iters: usize = 30;
    var converged: bool = false;

    for (0..max_iters) |_| {
        shapeHex8(
            xi_val,
            eta_val,
            zeta_val,
            &weights,
            &deriv_xi,
            &deriv_eta,
            &deriv_zeta,
        );

        const w_vec: Vec8 = weights;
        const dxi_vec: Vec8 = deriv_xi;
        const deta_vec: Vec8 = deriv_eta;
        const dzeta_vec: Vec8 = deriv_zeta;

        const calc_x = @reduce(.Add, w_vec * nx);
        const calc_y = @reduce(.Add, w_vec * ny);
        const calc_z = @reduce(.Add, w_vec * nz);

        const j11 = @reduce(.Add, dxi_vec * nx);
        const j12 = @reduce(.Add, deta_vec * nx);
        const j13 = @reduce(.Add, dzeta_vec * nx);

        const j21 = @reduce(.Add, dxi_vec * ny);
        const j22 = @reduce(.Add, deta_vec * ny);
        const j23 = @reduce(.Add, dzeta_vec * ny);

        const j31 = @reduce(.Add, dxi_vec * nz);
        const j32 = @reduce(.Add, deta_vec * nz);
        const j33 = @reduce(.Add, dzeta_vec * nz);

        const resid_x = targ_x - calc_x;
        const resid_y = targ_y - calc_y;
        const resid_z = targ_z - calc_z;

        if (@abs(resid_x) < 1e-11 and
            @abs(resid_y) < 1e-11 and
            @abs(resid_z) < 1e-11)
        {
            converged = true;
            break;
        }

        const c11 = j22 * j33 - j23 * j32;
        const c12 = j23 * j31 - j21 * j33;
        const c13 = j21 * j32 - j22 * j31;

        const c21 = j13 * j32 - j12 * j33;
        const c22 = j11 * j33 - j13 * j31;
        const c23 = j12 * j31 - j11 * j32;

        const c31 = j12 * j23 - j13 * j22;
        const c32 = j13 * j21 - j11 * j23;
        const c33 = j11 * j22 - j12 * j21;

        const det_jac = j11 * c11 + j12 * c12 + j13 * c13;

        if (@abs(det_jac) < 1e-14) {
            break;
        }

        const inv_det = 1.0 / det_jac;

        const delta_xi = inv_det * (c11 * resid_x + c21 * resid_y + c31 * resid_z);
        const delta_eta = inv_det * (c12 * resid_x + c22 * resid_y + c32 * resid_z);
        const delta_zeta = inv_det * (c13 * resid_x + c23 * resid_y + c33 * resid_z);

        xi_val += delta_xi;
        eta_val += delta_eta;
        zeta_val += delta_zeta;

        if (@abs(delta_xi) < 1e-12 and
            @abs(delta_eta) < 1e-12 and
            @abs(delta_zeta) < 1e-12)
        {
            converged = true;
            break;
        }
    }

    if (converged and
        xi_val >= (-1.0 - tol_eps) and xi_val <= (1.0 + tol_eps) and
        eta_val >= (-1.0 - tol_eps) and eta_val <= (1.0 + tol_eps) and
        zeta_val >= (-1.0 - tol_eps) and zeta_val <= (1.0 + tol_eps))
    {
        out_result.inside = true;
        out_result.node_count = 8;
        shapeHex8(
            xi_val,
            eta_val,
            zeta_val,
            out_result.weights[0..8],
            null,
            null,
            null,
        );
    } else {
        out_result.inside = false;
    }
}

pub fn invertHex20(
    node_x: *const [20]F,
    node_y: *const [20]F,
    node_z: *const [20]F,
    targ_x: F,
    targ_y: F,
    targ_z: F,
    tol_eps: F,
    out_result: *InverseResult,
) void {
    var xi_val: F = 0.0;
    var eta_val: F = 0.0;
    var zeta_val: F = 0.0;

    var weights: [20]F = undefined;
    var deriv_xi: [20]F = undefined;
    var deriv_eta: [20]F = undefined;
    var deriv_zeta: [20]F = undefined;

    const max_iters: usize = 30;
    var converged: bool = false;

    for (0..max_iters) |_| {
        shapeHex20(
            xi_val,
            eta_val,
            zeta_val,
            &weights,
            &deriv_xi,
            &deriv_eta,
            &deriv_zeta,
        );

        var calc_x: F = 0.0;
        var calc_y: F = 0.0;
        var calc_z: F = 0.0;

        var j11: F = 0.0;
        var j12: F = 0.0;
        var j13: F = 0.0;
        var j21: F = 0.0;
        var j22: F = 0.0;
        var j23: F = 0.0;
        var j31: F = 0.0;
        var j32: F = 0.0;
        var j33: F = 0.0;

        for (0..20) |nn| {
            calc_x += weights[nn] * node_x[nn];
            calc_y += weights[nn] * node_y[nn];
            calc_z += weights[nn] * node_z[nn];

            j11 += deriv_xi[nn] * node_x[nn];
            j12 += deriv_eta[nn] * node_x[nn];
            j13 += deriv_zeta[nn] * node_x[nn];

            j21 += deriv_xi[nn] * node_y[nn];
            j22 += deriv_eta[nn] * node_y[nn];
            j23 += deriv_zeta[nn] * node_y[nn];

            j31 += deriv_xi[nn] * node_z[nn];
            j32 += deriv_eta[nn] * node_z[nn];
            j33 += deriv_zeta[nn] * node_z[nn];
        }

        const resid_x = targ_x - calc_x;
        const resid_y = targ_y - calc_y;
        const resid_z = targ_z - calc_z;

        if (@abs(resid_x) < 1e-11 and
            @abs(resid_y) < 1e-11 and
            @abs(resid_z) < 1e-11)
        {
            converged = true;
            break;
        }

        const c11 = j22 * j33 - j23 * j32;
        const c12 = j23 * j31 - j21 * j33;
        const c13 = j21 * j32 - j22 * j31;

        const c21 = j13 * j32 - j12 * j33;
        const c22 = j11 * j33 - j13 * j31;
        const c23 = j12 * j31 - j11 * j32;

        const c31 = j12 * j23 - j13 * j22;
        const c32 = j13 * j21 - j11 * j23;
        const c33 = j11 * j22 - j12 * j21;

        const det_jac = j11 * c11 + j12 * c12 + j13 * c13;

        if (@abs(det_jac) < 1e-14) {
            break;
        }

        const inv_det = 1.0 / det_jac;

        const delta_xi = inv_det * (c11 * resid_x + c21 * resid_y + c31 * resid_z);
        const delta_eta = inv_det * (c12 * resid_x + c22 * resid_y + c32 * resid_z);
        const delta_zeta = inv_det * (c13 * resid_x + c23 * resid_y + c33 * resid_z);

        xi_val += delta_xi;
        eta_val += delta_eta;
        zeta_val += delta_zeta;

        if (@abs(delta_xi) < 1e-12 and
            @abs(delta_eta) < 1e-12 and
            @abs(delta_zeta) < 1e-12)
        {
            converged = true;
            break;
        }
    }

    if (converged and
        xi_val >= (-1.0 - tol_eps) and xi_val <= (1.0 + tol_eps) and
        eta_val >= (-1.0 - tol_eps) and eta_val <= (1.0 + tol_eps) and
        zeta_val >= (-1.0 - tol_eps) and zeta_val <= (1.0 + tol_eps))
    {
        out_result.inside = true;
        out_result.node_count = 20;
        shapeHex20(
            xi_val,
            eta_val,
            zeta_val,
            out_result.weights[0..20],
            null,
            null,
            null,
        );
    } else {
        out_result.inside = false;
    }
}
