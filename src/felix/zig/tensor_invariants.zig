// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("tensor_invariants_common.zig");
const inv_scalar = @import("tensor_invariants_scalar.zig");
const inv_simd = @import("tensor_invariants_simd.zig");

const cfg = buildconfig.config;
const inv_impl = if (cfg.simd == .on) inv_simd else inv_scalar;

pub const F = common.F;
pub const TensorInvariantType = common.TensorInvariantType;

pub const evalPrincipal2D = inv_impl.evalPrincipal2D;
pub const evalPrincipal3D = inv_impl.evalPrincipal3D;
pub const evalVonMises2D = inv_impl.evalVonMises2D;
pub const evalVonMises3D = inv_impl.evalVonMises3D;
pub const evalHydrostatic2D = inv_impl.evalHydrostatic2D;
pub const evalHydrostatic3D = inv_impl.evalHydrostatic3D;
pub const evalTresca2D = inv_impl.evalTresca2D;
pub const evalTresca3D = inv_impl.evalTresca3D;
pub const transformTensorArray2D = inv_impl.transformTensorArray2D;
pub const transformTensorArray3D = inv_impl.transformTensorArray3D;
