// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("errors_common.zig");
const errors_scalar = @import("errors_scalar.zig");
const errors_simd = @import("errors_simd.zig");

const cfg = buildconfig.config;
const errors_impl = if (cfg.simd == .on) errors_simd else errors_scalar;

// --------------------------------------------------------------------------------------
// Re-exported Types
// --------------------------------------------------------------------------------------

pub const F = common.F;
pub const SimdWidth = common.SimdWidth;
pub const VecSF = common.VecSF;
pub const ErrorType = common.ErrorType;
pub const ErrorDependence = common.ErrorDependence;
pub const DistType = common.DistType;
pub const RoundMethod = common.RoundMethod;
pub const DistributionSpec = common.DistributionSpec;
pub const FieldPerturbationSpec = common.FieldPerturbationSpec;
pub const ErrorSpec = common.ErrorSpec;

// --------------------------------------------------------------------------------------
// Public Error Evaluation Functions
// --------------------------------------------------------------------------------------

pub const evalRound = errors_impl.evalRound;
pub const evalPoly = errors_impl.evalPoly;
pub const evalTableLookup1D = errors_impl.evalTableLookup1D;
pub const applyOffsetSIMD = errors_simd.applyOffsetSIMD;
pub const applyPercentageSIMD = errors_simd.applyPercentageSIMD;
pub const applySaturationSIMD = errors_simd.applySaturationSIMD;
