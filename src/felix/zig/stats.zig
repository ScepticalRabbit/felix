// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("stats_common.zig");
const stats_scalar = @import("stats_scalar.zig");
const stats_simd = @import("stats_simd.zig");

const cfg = buildconfig.config;
const stats_impl = if (cfg.simd == .on) stats_simd else stats_scalar;

pub const F = common.F;
pub const SimdWidth = common.SimdWidth;
pub const calcMedian = common.calcMedian;
pub const lessThan = common.lessThan;

pub const calcExperimentStats = stats_impl.calcExperimentStats;
