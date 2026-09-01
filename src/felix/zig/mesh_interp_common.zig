// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("buildconfig.zig");
const elements = @import("elements.zig");

pub const F = buildconfig.F;
pub const SimdWidth = buildconfig.SimdWidth;
pub const VecSF = buildconfig.VecSF;
pub const ElementType = elements.ElementType;
pub const InverseResult = elements.InverseResult;

// --------------------------------------------------------------------------------------
// Public Types
// --------------------------------------------------------------------------------------

pub const ElementBBox = struct {
    min_x: F,
    max_x: F,
    min_y: F,
    max_y: F,
    min_z: F,
    max_z: F,

    pub fn containsPoint(self: ElementBBox, px: F, py: F, pz: F, tol_eps: F) bool {
        return (px >= self.min_x - tol_eps and px <= self.max_x + tol_eps and
            py >= self.min_y - tol_eps and py <= self.max_y + tol_eps and
            pz >= self.min_z - tol_eps and pz <= self.max_z + tol_eps);
    }
};

pub const SensorLocation = struct {
    found: bool,
    elem_idx: usize,
    node_count: usize,
    weights: [elements.max_elem_nodes]F,
    node_indices: [elements.max_elem_nodes]usize,
};

// --------------------------------------------------------------------------------------
// Spatial Search Functions
// --------------------------------------------------------------------------------------

pub fn calcElementBBox(
    coords_ptr: [*c]const F,
    connect_elem: []const usize,
    nodes_per_elem: usize,
    out_bbox: *ElementBBox,
) void {
    var min_x: F = 1e30;
    var max_x: F = -1e30;
    var min_y: F = 1e30;
    var max_y: F = -1e30;
    var min_z: F = 1e30;
    var max_z: F = -1e30;

    for (0..nodes_per_elem) |nn| {
        const node_id = connect_elem[nn];
        const px = coords_ptr[node_id * 3 + 0];
        const py = coords_ptr[node_id * 3 + 1];
        const pz = coords_ptr[node_id * 3 + 2];

        if (px < min_x) min_x = px;
        if (px > max_x) max_x = px;
        if (py < min_y) min_y = py;
        if (py > max_y) max_y = py;
        if (pz < min_z) min_z = pz;
        if (pz > max_z) max_z = pz;
    }

    out_bbox.min_x = min_x;
    out_bbox.max_x = max_x;
    out_bbox.min_y = min_y;
    out_bbox.max_y = max_y;
    out_bbox.min_z = min_z;
    out_bbox.max_z = max_z;
}

pub fn locatePointInMesh(
    coords_ptr: [*c]const F,
    connect_ptr: [*c]const usize,
    num_elements: usize,
    elem_type: ElementType,
    targ_x: F,
    targ_y: F,
    targ_z: F,
    out_location: *SensorLocation,
) void {
    const nodes_per_elem = elem_type.nodeCount();
    const tol_eps: F = 1e-6;

    out_location.found = false;
    out_location.elem_idx = 0;
    out_location.node_count = nodes_per_elem;

    for (0..num_elements) |ee| {
        const elem_connect = connect_ptr[ee * nodes_per_elem .. (ee + 1) * nodes_per_elem];

        var bbox: ElementBBox = undefined;
        calcElementBBox(coords_ptr, elem_connect, nodes_per_elem, &bbox);

        if (!bbox.containsPoint(targ_x, targ_y, targ_z, 1e-4)) {
            continue;
        }

        var inv_res: InverseResult = undefined;
        invertElementDirect(
            coords_ptr,
            elem_connect,
            elem_type,
            targ_x,
            targ_y,
            targ_z,
            tol_eps,
            &inv_res,
        );

        if (inv_res.inside) {
            out_location.found = true;
            out_location.elem_idx = ee;
            out_location.node_count = nodes_per_elem;
            for (0..nodes_per_elem) |nn| {
                out_location.weights[nn] = inv_res.weights[nn];
                out_location.node_indices[nn] = elem_connect[nn];
            }
            return;
        }
    }
}

pub fn invertElementDirect(
    coords_ptr: [*c]const F,
    elem_connect: []const usize,
    elem_type: ElementType,
    targ_x: F,
    targ_y: F,
    targ_z: F,
    tol_eps: F,
    out_inv_res: *InverseResult,
) void {
    out_inv_res.inside = false;

    switch (elem_type) {
        .tri3 => {
            var node_x: [3]F = undefined;
            var node_y: [3]F = undefined;
            for (0..3) |nn| {
                const nid = elem_connect[nn];
                node_x[nn] = coords_ptr[nid * 3 + 0];
                node_y[nn] = coords_ptr[nid * 3 + 1];
            }
            elements.invertTri3(&node_x, &node_y, targ_x, targ_y, tol_eps, out_inv_res);
        },
        .quad4 => {
            var node_x: [4]F = undefined;
            var node_y: [4]F = undefined;
            for (0..4) |nn| {
                const nid = elem_connect[nn];
                node_x[nn] = coords_ptr[nid * 3 + 0];
                node_y[nn] = coords_ptr[nid * 3 + 1];
            }
            elements.invertQuad4(&node_x, &node_y, targ_x, targ_y, tol_eps, out_inv_res);
        },
        .tet4 => {
            var node_x: [4]F = undefined;
            var node_y: [4]F = undefined;
            var node_z: [4]F = undefined;
            for (0..4) |nn| {
                const nid = elem_connect[nn];
                node_x[nn] = coords_ptr[nid * 3 + 0];
                node_y[nn] = coords_ptr[nid * 3 + 1];
                node_z[nn] = coords_ptr[nid * 3 + 2];
            }
            elements.invertTet4(
                &node_x,
                &node_y,
                &node_z,
                targ_x,
                targ_y,
                targ_z,
                tol_eps,
                out_inv_res,
            );
        },
        .hex8 => {
            var node_x: [8]F = undefined;
            var node_y: [8]F = undefined;
            var node_z: [8]F = undefined;
            for (0..8) |nn| {
                const nid = elem_connect[nn];
                node_x[nn] = coords_ptr[nid * 3 + 0];
                node_y[nn] = coords_ptr[nid * 3 + 1];
                node_z[nn] = coords_ptr[nid * 3 + 2];
            }
            elements.invertHex8(
                &node_x,
                &node_y,
                &node_z,
                targ_x,
                targ_y,
                targ_z,
                tol_eps,
                out_inv_res,
            );
        },
        .quad8 => {
            var node_x: [8]F = undefined;
            var node_y: [8]F = undefined;
            for (0..8) |nn| {
                const nid = elem_connect[nn];
                node_x[nn] = coords_ptr[nid * 3 + 0];
                node_y[nn] = coords_ptr[nid * 3 + 1];
            }
            elements.invertQuad8(&node_x, &node_y, targ_x, targ_y, tol_eps, out_inv_res);
        },
        .hex20 => {
            var node_x: [20]F = undefined;
            var node_y: [20]F = undefined;
            var node_z: [20]F = undefined;
            for (0..20) |nn| {
                const nid = elem_connect[nn];
                node_x[nn] = coords_ptr[nid * 3 + 0];
                node_y[nn] = coords_ptr[nid * 3 + 1];
                node_z[nn] = coords_ptr[nid * 3 + 2];
            }
            elements.invertHex20(
                &node_x,
                &node_y,
                &node_z,
                targ_x,
                targ_y,
                targ_z,
                tol_eps,
                out_inv_res,
            );
        },
        else => {},
    }
}
