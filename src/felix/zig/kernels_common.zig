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
pub const SimdWidth = buildconfig.SimdWidth;
pub const VecSF = buildconfig.VecSF;

pub const KernelType = enum(u32) {
    uniform = 0,
    gaussian = 1,
    triangular = 2,
    cosine = 3,
    epanechnikov = 4,
    sinc = 5,
    lanczos = 6,
};

pub const KernelSpec = extern struct {
    kernel_type: u32,
    param0: F,
    param1: F,
    param2: F,
};
