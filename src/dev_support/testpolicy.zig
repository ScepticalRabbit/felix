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

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const F = buildconfig.F;

pub const DatasetCase = enum {
    tri3_twoelems,
    tri6_twoelems,
    quad4_twoelems,
    quad8_twoelems,
    quad9_twoelems,
    cube_hex8,
    cube_hex20,
    tet4,
    tet10,
    plate_2d_mech,
    plate_2d_tm,
    monoblock_3d,
    cylinder_3d_mech,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn datasetPath(case: DatasetCase) []const u8 {
    return switch (case) {
        .tri3_twoelems => "data/min/tri3_twoelems/",
        .tri6_twoelems => "data/min/tri6_twoelems/",
        .quad4_twoelems => "data/min/quad4_twoelems/",
        .quad8_twoelems => "data/min/quad8_twoelems/",
        .quad9_twoelems => "data/min/quad9_twoelems/",
        .cube_hex8 => "data/cube_hex8/",
        .cube_hex20 => "data/cube_hex20/",
        .tet4 => "data/tet4/",
        .tet10 => "data/tet10/",
        .plate_2d_mech => "data/plate_2d_mech/",
        .plate_2d_tm => "data/plate_2d_tm/",
        .monoblock_3d => "data/monoblock_3d/",
        .cylinder_3d_mech => "data/cylinder_3d_mech/",
    };
}

pub fn goldRoot(suite_name: []const u8) []const u8 {
    _ = suite_name;
    return "data/gold/";
}
