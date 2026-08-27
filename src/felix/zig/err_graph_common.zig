// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const errors = @import("errors.zig");

pub const F = buildconfig.F;
pub const SimdWidth = buildconfig.SimdWidth;
pub const VecSF = buildconfig.VecSF;
pub const ErrorSpec = errors.ErrorSpec;

pub const ErrOp = enum(u32) {
    add = 0,
    multiply = 1,
    replace = 2,
    custom_poly = 3,
    custom_table = 4,
};

pub const ErrGraphNodeSpec = extern struct {
    op: u32,
    num_inputs: usize,
    input_indices_ptr: [*c]const usize,
    error_spec: ErrorSpec,
};

pub const ErrGraphSpec = extern struct {
    num_nodes: usize,
    nodes_ptr: [*c]const ErrGraphNodeSpec,
    execution_order_ptr: [*c]const usize,
    num_leaves: usize,
    leaf_indices_ptr: [*c]const usize,
    store_node_outputs: u32,
};
