// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../felix/zig/buildconfig.zig");
const meshio = @import("../felix/zig/meshio.zig");
const elements = @import("../felix/zig/elements.zig");
const sensor_sim = @import("../felix/zig/sensor_sim.zig");
const policy = @import("testpolicy.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const F = buildconfig.F;
pub const SimData = meshio.SimData;
pub const ElementType = elements.ElementType;
pub const SimMeshInput = sensor_sim.SimMeshInput;
pub const SensorArrayInput = sensor_sim.SensorArrayInput;

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn inferElementType(nodes_per_elem: usize, is_3d: bool) !ElementType {
    if (is_3d) {
        return switch (nodes_per_elem) {
            4 => .tet4,
            8 => .hex8,
            20 => .hex20,
            else => error.Unsupported3DElement,
        };
    } else {
        return switch (nodes_per_elem) {
            3 => .tri3,
            6 => .tri6,
            4 => .quad4,
            8 => .quad8,
            else => error.Unsupported2DElement,
        };
    }
}

pub fn loadCaseSimData(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    case: policy.DatasetCase,
    field_suffixes: ?[]const []const u8,
    disp_suffixes: ?[]const []const u8,
) !SimData {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const local_alloc = arena.allocator();

    const dir_path = policy.datasetPath(case);

    const coord_path = try std.fmt.allocPrint(
        local_alloc,
        "{s}coords.csv",
        .{dir_path},
    );

    const connect_path = try std.fmt.allocPrint(
        local_alloc,
        "{s}connectivity.csv",
        .{dir_path},
    );

    var field_paths: ?[][]const u8 = null;
    if (field_suffixes) |fs| {
        field_paths = try local_alloc.alloc([]const u8, fs.len);
        for (fs, 0..) |suffix, ii| {
            field_paths.?[ii] = try std.fmt.allocPrint(
                local_alloc,
                "{s}{s}",
                .{ dir_path, suffix },
            );
        }
    }

    var disp_paths: ?[][]const u8 = null;
    if (disp_suffixes) |ds| {
        disp_paths = try local_alloc.alloc([]const u8, ds.len);
        for (ds, 0..) |suffix, ii| {
            disp_paths.?[ii] = try std.fmt.allocPrint(
                local_alloc,
                "{s}{s}",
                .{ dir_path, suffix },
            );
        }
    }

    return try meshio.loadSimData(
        outer_alloc,
        io,
        coord_path,
        connect_path,
        field_paths,
        disp_paths,
    );
}

pub fn buildSimMeshInput(
    sim_data: *const SimData,
    elem_type: ElementType,
    sim_times: []const F,
) SimMeshInput {
    const coords_ptr = sim_data.coords.mem.ptr;
    const num_nodes = sim_data.coords.mat.rows_num;
    const connect_ptr = sim_data.connect.table_mem.ptr;
    const num_elements = sim_data.connect.getElemsNum();

    const field_ptr: [*c]const F = if (sim_data.field) |f| f.array_mem.ptr else null;
    const num_components: usize = if (sim_data.field) |f| f.getFieldsN() else 0;

    return .{
        .coords_ptr = coords_ptr,
        .num_nodes = num_nodes,
        .connect_ptr = connect_ptr,
        .num_elements = num_elements,
        .elem_type = @intFromEnum(elem_type),
        .nodal_fields_ptr = field_ptr,
        .num_components = num_components,
        .sim_times_ptr = sim_times.ptr,
        .num_sim_times = sim_times.len,
    };
}
