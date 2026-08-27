// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const common = @import("tensor_invariants_common.zig");
const scalar = @import("tensor_invariants_scalar.zig");

pub const TensorInvariantType = common.TensorInvariantType;

pub const evalPrincipal2D = scalar.evalPrincipal2D;
pub const evalPrincipal3D = scalar.evalPrincipal3D;
pub const evalVonMises2D = scalar.evalVonMises2D;
pub const evalVonMises3D = scalar.evalVonMises3D;
pub const evalHydrostatic2D = scalar.evalHydrostatic2D;
pub const evalHydrostatic3D = scalar.evalHydrostatic3D;
pub const evalTresca2D = scalar.evalTresca2D;
pub const evalTresca3D = scalar.evalTresca3D;
pub const transformTensorArray2D = scalar.transformTensorArray2D;
pub const transformTensorArray3D = scalar.transformTensorArray3D;
