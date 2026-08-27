// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const common = @import("kernels_common.zig");
const scalar = @import("kernels_scalar.zig");

pub const KernelType = common.KernelType;
pub const KernelSpec = common.KernelSpec;

pub const evalWeight = scalar.evalWeight;
pub const evalWeightsBatch = scalar.evalWeightsBatch;
