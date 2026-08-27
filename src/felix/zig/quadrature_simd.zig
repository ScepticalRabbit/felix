// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const common = @import("quadrature_common.zig");
const scalar = @import("quadrature_scalar.zig");

pub const QuadRule = common.QuadRule;
pub const QuadSpec = common.QuadSpec;

pub const getGaussLegendre1D = scalar.getGaussLegendre1D;
pub const getMidpoint1D = scalar.getMidpoint1D;
pub const getTrapezoidal1D = scalar.getTrapezoidal1D;
pub const getSimpson1D = scalar.getSimpson1D;
pub const generateNodesAndWeightsND = scalar.generateNodesAndWeightsND;
