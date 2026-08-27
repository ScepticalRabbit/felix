// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const common = @import("err_graph_common.zig");
const scalar = @import("err_graph_scalar.zig");

pub const ErrOp = common.ErrOp;
pub const ErrGraphNodeSpec = common.ErrGraphNodeSpec;
pub const ErrGraphSpec = common.ErrGraphSpec;

pub const runErrGraphSimulation = scalar.runErrGraphSimulation;
