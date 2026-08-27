// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");

pub const F = buildconfig.F;

pub const TensorInvariantType = enum(u32) {
    von_mises = 0,
    principal_1 = 1,
    principal_2 = 2,
    principal_3 = 3,
    tresca = 4,
    hydrostatic = 5,
    invariant_i1 = 6,
    invariant_i2 = 7,
    invariant_i3 = 8,
    invariant_j2 = 9,
    invariant_j3 = 10,
    max_shear = 11,
};
