// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("err_graph_common.zig");
const err_graph_scalar = @import("err_graph_scalar.zig");
const err_graph_simd = @import("err_graph_simd.zig");

const cfg = buildconfig.config;
const err_graph_impl = if (cfg.simd == .on) err_graph_simd else err_graph_scalar;

pub const F = common.F;
pub const SimdWidth = common.SimdWidth;
pub const VecSF = common.VecSF;
pub const ErrOp = common.ErrOp;
pub const ErrGraphNodeSpec = common.ErrGraphNodeSpec;
pub const ErrGraphSpec = common.ErrGraphSpec;

pub const runErrGraphSimulation = err_graph_impl.runErrGraphSimulation;
