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
// Error Enums & C-ABI Compatible Descriptors
// --------------------------------------------------------------------------------------

pub const ErrorType = enum(u32) {
    systematic = 0,
    random = 1,
};

pub const ErrorDependence = enum(u32) {
    independent = 0,
    dependent = 1,
};

pub const DistType = enum(u32) {
    none = 0,
    uniform = 1,
    normal = 2,
    triangular = 3,
    exponential = 4,
    gamma = 5,
    beta = 6,
    lognormal = 7,
};

pub const RoundMethod = enum(u32) {
    round = 0,
    floor = 1,
    ceil = 2,
};

pub const DistributionSpec = extern struct {
    dist_type: u32,
    param0: F,
    param1: F,
    param2: F,
    seed: u64,
    has_seed: u32,
};

pub const FieldPerturbationSpec = extern struct {
    pos_offset_ptr: [*c]const F,
    pos_lock_ptr: [*c]const u8,
    angle_offset_ptr: [*c]const F,
    angle_lock_ptr: [*c]const u8,
    time_offset_ptr: [*c]const F,
    pos_rand: [3]DistributionSpec,
    angle_rand: [3]DistributionSpec,
    time_rand: DistributionSpec,
    drift_kind: u32,
    drift_param0: F,
    drift_param1: F,
    drift_param2: F,
    drift_poly_ptr: [*c]const F,
    drift_poly_len: usize,
    spatial_kind: u32,
    spatial_dim_x: F,
    spatial_dim_y: F,
    spatial_dim_z: F,
};

pub const ErrorSpec = extern struct {
    kind: u32,
    err_type: u32,
    err_dep: u32,
    dist_type: u32,
    param0: F,
    param1: F,
    param2: F,
    seed: u64,
    has_seed: u32,
    table_ptr: [*c]const F,
    table_rows: usize,
    poly_coeffs_ptr: [*c]const F,
    poly_coeffs_len: usize,
    field_spec_ptr: [*c]const FieldPerturbationSpec,
};
