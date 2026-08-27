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

pub fn calcMedian(sorted: []const F) F {
    const mid = sorted.len / 2;
    if (sorted.len % 2 == 1) return sorted[mid];
    return (sorted[mid - 1] + sorted[mid]) / 2.0;
}

pub fn lessThan(_: void, lhs: F, rhs: F) bool {
    return lhs < rhs;
}
