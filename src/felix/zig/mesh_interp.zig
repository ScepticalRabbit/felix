// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("mesh_interp_common.zig");
const mesh_interp_scalar = @import("mesh_interp_scalar.zig");
const mesh_interp_simd = @import("mesh_interp_simd.zig");

const cfg = buildconfig.config;
const mesh_interp_impl = if (cfg.simd == .on) mesh_interp_simd else mesh_interp_scalar;

// --------------------------------------------------------------------------------------
// Re-exported Types & Helpers
// --------------------------------------------------------------------------------------

pub const F = common.F;
pub const SimdWidth = common.SimdWidth;
pub const VecSF = common.VecSF;
pub const ElementType = common.ElementType;
pub const InverseResult = common.InverseResult;
pub const ElementBBox = common.ElementBBox;
pub const SensorLocation = common.SensorLocation;

pub const calcElementBBox = common.calcElementBBox;
pub const locatePointInMesh = common.locatePointInMesh;
pub const interpTimeLinear = mesh_interp_impl.interpTimeLinear;
pub const sampleCachedFEPoint = mesh_interp_impl.sampleCachedFEPoint;
