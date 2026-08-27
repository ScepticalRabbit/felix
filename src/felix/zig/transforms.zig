// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("transforms_common.zig");
const transforms_scalar = @import("transforms_scalar.zig");
const transforms_simd = @import("transforms_simd.zig");

const cfg = buildconfig.config;
const transforms_impl = if (cfg.simd == .on) transforms_simd else transforms_scalar;

// --------------------------------------------------------------------------------------
// Re-exported Types
// --------------------------------------------------------------------------------------

pub const F = common.F;
pub const SimdWidth = common.SimdWidth;
pub const VecSF = common.VecSF;
pub const Vec2Packet = common.Vec2Packet;
pub const Vec3Packet = common.Vec3Packet;
pub const Tensor2Packet = common.Tensor2Packet;
pub const Tensor3Packet = common.Tensor3Packet;

// --------------------------------------------------------------------------------------
// Public Transformation Functions
// --------------------------------------------------------------------------------------

pub const transformVector2D = transforms_impl.transformVector2D;
pub const transformVector3D = transforms_impl.transformVector3D;
pub const transformTensor2D = transforms_impl.transformTensor2D;
pub const transformTensor3D = transforms_impl.transformTensor3D;
pub const transformVector3DBatch = transforms_impl.transformVector3DBatch;
