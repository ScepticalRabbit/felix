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
const mesh_interp_common = @import("mesh_interp_common.zig");

pub const F = buildconfig.F;
pub const ElementType = elements.ElementType;
pub const ElementBBox = mesh_interp_common.ElementBBox;
pub const SensorLocation = mesh_interp_common.SensorLocation;

// --------------------------------------------------------------------------------------
// Uniform Voxel Spatial Acceleration Grid
// --------------------------------------------------------------------------------------

pub const UniformVoxelGrid = struct {
    min_x: F,
    max_x: F,
    min_y: F,
    max_y: F,
    min_z: F,
    max_z: F,
    grid_dim_x: usize,
    grid_dim_y: usize,
    grid_dim_z: usize,
    inv_cell_dx: F,
    inv_cell_dy: F,
    inv_cell_dz: F,
    cell_offsets: []usize,
    cell_elements: []usize,

    pub fn init(
        outer_alloc: std.mem.Allocator,
        coords_ptr: [*c]const F,
        connect_ptr: [*c]const usize,
        num_elements: usize,
        elem_type: ElementType,
    ) !UniformVoxelGrid {
        if (num_elements == 0) {
            return UniformVoxelGrid{
                .min_x = 0.0,
                .max_x = 0.0,
                .min_y = 0.0,
                .max_y = 0.0,
                .min_z = 0.0,
                .max_z = 0.0,
                .grid_dim_x = 0,
                .grid_dim_y = 0,
                .grid_dim_z = 0,
                .inv_cell_dx = 0.0,
                .inv_cell_dy = 0.0,
                .inv_cell_dz = 0.0,
                .cell_offsets = try outer_alloc.alloc(usize, 1),
                .cell_elements = try outer_alloc.alloc(usize, 0),
            };
        }

        const nodes_per_elem = elem_type.nodeCount();

        var global_min_x: F = 1e30;
        var global_max_x: F = -1e30;
        var global_min_y: F = 1e30;
        var global_max_y: F = -1e30;
        var global_min_z: F = 1e30;
        var global_max_z: F = -1e30;

        var elem_bboxes = try outer_alloc.alloc(ElementBBox, num_elements);
        defer outer_alloc.free(elem_bboxes);

        for (0..num_elements) |ee| {
            const elem_conn = connect_ptr[ee * nodes_per_elem ..][0..nodes_per_elem];
            var bbox: ElementBBox = undefined;
            mesh_interp_common.calcElementBBox(
                coords_ptr,
                elem_conn,
                nodes_per_elem,
                &bbox,
            );
            elem_bboxes[ee] = bbox;

            if (bbox.min_x < global_min_x) global_min_x = bbox.min_x;
            if (bbox.max_x > global_max_x) global_max_x = bbox.max_x;
            if (bbox.min_y < global_min_y) global_min_y = bbox.min_y;
            if (bbox.max_y > global_max_y) global_max_y = bbox.max_y;
            if (bbox.min_z < global_min_z) global_min_z = bbox.min_z;
            if (bbox.max_z > global_max_z) global_max_z = bbox.max_z;
        }

        const range_x = @max(global_max_x - global_min_x, 1e-6);
        const range_y = @max(global_max_y - global_min_y, 1e-6);
        const range_z = @max(global_max_z - global_min_z, 1e-6);

        const is_2d = (range_z < 1e-5);
        const target_cells = @min(num_elements * 2, 500_000);

        var dim_x: usize = 1;
        var dim_y: usize = 1;
        var dim_z: usize = 1;

        if (is_2d) {
            const area = range_x * range_y;
            const cell_side = @sqrt(area / @as(F, @floatFromInt(target_cells)));
            dim_x = @max(1, @min(256, @as(usize, @intFromFloat(range_x / cell_side))));
            dim_y = @max(1, @min(256, @as(usize, @intFromFloat(range_y / cell_side))));
            dim_z = 1;
        } else {
            const volume = range_x * range_y * range_z;
            const cell_side = std.math.cbrt(
                volume / @as(F, @floatFromInt(target_cells)),
            );
            dim_x = @max(1, @min(128, @as(usize, @intFromFloat(range_x / cell_side))));
            dim_y = @max(1, @min(128, @as(usize, @intFromFloat(range_y / cell_side))));
            dim_z = @max(1, @min(128, @as(usize, @intFromFloat(range_z / cell_side))));
        }

        const inv_cell_dx = @as(F, @floatFromInt(dim_x)) / range_x;
        const inv_cell_dy = @as(F, @floatFromInt(dim_y)) / range_y;
        const inv_cell_dz = if (is_2d) 1.0 else @as(F, @floatFromInt(dim_z)) / range_z;

        const total_cells = dim_x * dim_y * dim_z;
        const cell_counts = try outer_alloc.alloc(usize, total_cells);
        defer outer_alloc.free(cell_counts);
        @memset(cell_counts, 0);

        for (0..num_elements) |ee| {
            const bb = elem_bboxes[ee];
            const i_min_x = @min(
                dim_x - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.min_x - global_min_x) * inv_cell_dx))),
            );
            const i_max_x = @min(
                dim_x - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.max_x - global_min_x) * inv_cell_dx))),
            );
            const i_min_y = @min(
                dim_y - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.min_y - global_min_y) * inv_cell_dy))),
            );
            const i_max_y = @min(
                dim_y - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.max_y - global_min_y) * inv_cell_dy))),
            );
            const i_min_z = if (is_2d) 0 else @min(
                dim_z - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.min_z - global_min_z) * inv_cell_dz))),
            );
            const i_max_z = if (is_2d) 0 else @min(
                dim_z - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.max_z - global_min_z) * inv_cell_dz))),
            );

            for (i_min_z..i_max_z + 1) |zz| {
                for (i_min_y..i_max_y + 1) |yy| {
                    for (i_min_x..i_max_x + 1) |xx| {
                        const cell_idx = xx + yy * dim_x + zz * dim_x * dim_y;
                        cell_counts[cell_idx] += 1;
                    }
                }
            }
        }

        const cell_offsets = try outer_alloc.alloc(usize, total_cells + 1);
        errdefer outer_alloc.free(cell_offsets);

        cell_offsets[0] = 0;
        for (0..total_cells) |ii| {
            cell_offsets[ii + 1] = cell_offsets[ii] + cell_counts[ii];
        }

        const total_elem_refs = cell_offsets[total_cells];
        const cell_elements = try outer_alloc.alloc(usize, total_elem_refs);
        errdefer outer_alloc.free(cell_elements);

        const write_heads = try outer_alloc.alloc(usize, total_cells);
        defer outer_alloc.free(write_heads);
        @memcpy(write_heads, cell_offsets[0..total_cells]);

        for (0..num_elements) |ee| {
            const bb = elem_bboxes[ee];
            const i_min_x = @min(
                dim_x - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.min_x - global_min_x) * inv_cell_dx))),
            );
            const i_max_x = @min(
                dim_x - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.max_x - global_min_x) * inv_cell_dx))),
            );
            const i_min_y = @min(
                dim_y - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.min_y - global_min_y) * inv_cell_dy))),
            );
            const i_max_y = @min(
                dim_y - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.max_y - global_min_y) * inv_cell_dy))),
            );
            const i_min_z = if (is_2d) 0 else @min(
                dim_z - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.min_z - global_min_z) * inv_cell_dz))),
            );
            const i_max_z = if (is_2d) 0 else @min(
                dim_z - 1,
                @as(usize, @intFromFloat(@max(0.0, (bb.max_z - global_min_z) * inv_cell_dz))),
            );

            for (i_min_z..i_max_z + 1) |zz| {
                for (i_min_y..i_max_y + 1) |yy| {
                    for (i_min_x..i_max_x + 1) |xx| {
                        const cell_idx = xx + yy * dim_x + zz * dim_x * dim_y;
                        const head = write_heads[cell_idx];
                        cell_elements[head] = ee;
                        write_heads[cell_idx] = head + 1;
                    }
                }
            }
        }

        return UniformVoxelGrid{
            .min_x = global_min_x,
            .max_x = global_max_x,
            .min_y = global_min_y,
            .max_y = global_max_y,
            .min_z = global_min_z,
            .max_z = global_max_z,
            .grid_dim_x = dim_x,
            .grid_dim_y = dim_y,
            .grid_dim_z = dim_z,
            .inv_cell_dx = inv_cell_dx,
            .inv_cell_dy = inv_cell_dy,
            .inv_cell_dz = inv_cell_dz,
            .cell_offsets = cell_offsets,
            .cell_elements = cell_elements,
        };
    }

    pub fn deinit(self: *UniformVoxelGrid, outer_alloc: std.mem.Allocator) void {
        outer_alloc.free(self.cell_offsets);
        outer_alloc.free(self.cell_elements);
        self.* = undefined;
    }

    pub fn locatePoint(
        self: *const UniformVoxelGrid,
        coords_ptr: [*c]const F,
        connect_ptr: [*c]const usize,
        elem_type: ElementType,
        px: F,
        py: F,
        pz: F,
        out_location: *SensorLocation,
    ) void {
        out_location.* = SensorLocation{
            .found = false,
            .elem_idx = 0,
            .node_count = 0,
            .weights = undefined,
            .node_indices = undefined,
        };

        if (px < self.min_x or px > self.max_x or
            py < self.min_y or py > self.max_y)
        {
            return;
        }

        const cell_ix = @min(
            self.grid_dim_x - 1,
            @as(usize, @intFromFloat(@max(0.0, (px - self.min_x) * self.inv_cell_dx))),
        );
        const cell_iy = @min(
            self.grid_dim_y - 1,
            @as(usize, @intFromFloat(@max(0.0, (py - self.min_y) * self.inv_cell_dy))),
        );
        const cell_iz = if (self.grid_dim_z <= 1) 0 else @min(
            self.grid_dim_z - 1,
            @as(usize, @intFromFloat(@max(0.0, (pz - self.min_z) * self.inv_cell_dz))),
        );

        const cell_idx = cell_ix +
            cell_iy * self.grid_dim_x +
            cell_iz * self.grid_dim_x * self.grid_dim_y;

        const start_idx = self.cell_offsets[cell_idx];
        const end_idx = self.cell_offsets[cell_idx + 1];
        const nodes_per_elem = elem_type.nodeCount();

        for (start_idx..end_idx) |idx| {
            const ee = self.cell_elements[idx];
            const elem_conn = connect_ptr[ee * nodes_per_elem ..][0..nodes_per_elem];

            var elem_inv: elements.InverseResult = undefined;
            mesh_interp_common.invertElementDirect(
                coords_ptr,
                elem_conn,
                elem_type,
                px,
                py,
                pz,
                1e-4,
                &elem_inv,
            );

            if (elem_inv.inside) {
                out_location.found = true;
                out_location.elem_idx = ee;
                out_location.node_count = elem_inv.node_count;
                for (0..elem_inv.node_count) |nn| {
                    out_location.weights[nn] = elem_inv.weights[nn];
                    out_location.node_indices[nn] = elem_conn[nn];
                }
                return;
            }
        }
    }
};
