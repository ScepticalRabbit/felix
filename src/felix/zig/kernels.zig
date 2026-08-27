// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("kernels_common.zig");
const kernels_scalar = @import("kernels_scalar.zig");
const kernels_simd = @import("kernels_simd.zig");

const cfg = buildconfig.config;
const kernels_impl = if (cfg.simd == .on) kernels_simd else kernels_scalar;

pub const F = common.F;
pub const KernelType = common.KernelType;
pub const KernelSpec = common.KernelSpec;

pub const evalWeight = kernels_impl.evalWeight;
pub const evalWeightsBatch = kernels_impl.evalWeightsBatch;
