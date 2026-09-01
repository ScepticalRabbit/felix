// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("../felix/zig/buildconfig.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const F = buildconfig.F;
pub const REL_TOL: F = if (F == f32) 1.0e-4 else 1.0e-8;
pub const ABS_TOL: F = if (F == f32) 1.0e-4 else 1.0e-8;
pub const TEST_CASE_VERBOSE: bool = false;
pub const TOTAL_THREADS: u16 = 1;
