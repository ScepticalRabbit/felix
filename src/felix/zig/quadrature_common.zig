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

pub const QuadRule = enum(u32) {
    gauss_legendre = 0,
    midpoint = 1,
    trapezoidal = 2,
    simpson = 3,
    monte_carlo = 4,
};

pub const QuadSpec = extern struct {
    rule: u32,
    order: usize,
    dims: usize,
};
