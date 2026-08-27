// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");

pub const F = buildconfig.F;
pub const SimdWidth = buildconfig.SimdWidth;
pub const VecSF = buildconfig.VecSF;

// --------------------------------------------------------------------------------------
// SIMD Structure-of-Arrays (SoA) Packet Types
// --------------------------------------------------------------------------------------

pub const Vec2Packet = struct {
    x: VecSF,
    y: VecSF,
};

pub const Vec3Packet = struct {
    x: VecSF,
    y: VecSF,
    z: VecSF,
};

pub const Tensor2Packet = struct {
    xx: VecSF,
    yy: VecSF,
    xy: VecSF,
};

pub const Tensor3Packet = struct {
    xx: VecSF,
    yy: VecSF,
    zz: VecSF,
    xy: VecSF,
    xz: VecSF,
    yz: VecSF,
};
