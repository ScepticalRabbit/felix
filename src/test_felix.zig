// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

pub const mesh_elements = @import("tests/test_mesh_elements.zig");
pub const field_transforms = @import("tests/test_field_transforms.zig");
pub const quadrature_kernels = @import("tests/test_quadrature_kernels.zig");
pub const stats = @import("tests/test_stats.zig");
pub const error_chains = @import("tests/test_error_chains.zig");
pub const err_graph = @import("tests/test_err_graph.zig");
pub const point_sensors = @import("tests/test_point_sensors.zig");
pub const monoblock_e2e = @import("tests/test_monoblock_e2e.zig");
pub const parachunkexec = @import("tests/test_parachunkexec.zig");
pub const experiment_sim = @import("tests/test_experiment_sim.zig");

test {
    std.testing.refAllDecls(@This());
}
