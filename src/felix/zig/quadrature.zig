// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("quadrature_common.zig");
const quadrature_scalar = @import("quadrature_scalar.zig");
const quadrature_simd = @import("quadrature_simd.zig");

const cfg = buildconfig.config;
const quad_impl = if (cfg.simd == .on) quadrature_simd else quadrature_scalar;

pub const F = common.F;
pub const QuadRule = common.QuadRule;
pub const QuadSpec = common.QuadSpec;

pub const getGaussLegendre1D = quad_impl.getGaussLegendre1D;
pub const getMidpoint1D = quad_impl.getMidpoint1D;
pub const getTrapezoidal1D = quad_impl.getTrapezoidal1D;
pub const getSimpson1D = quad_impl.getSimpson1D;
pub const generateNodesAndWeightsND = quad_impl.generateNodesAndWeightsND;
