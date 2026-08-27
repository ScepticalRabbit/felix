// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const root = @import("root");

const build_options = if (@hasDecl(root, "build_options"))
    root.build_options
else
    struct {
        pub const precision = "f64";
        pub const simd = "on";
        pub const simd_vec_width: comptime_int = 0;
    };

// --------------------------------------------------------------------------------------
// Main Configuration Structure
// --------------------------------------------------------------------------------------

pub const Config = struct {
    simd: SimdMode = default_simd,
    simd_vec_width: comptime_int = defaultSimdVecWidthForPrecision(F),
    precision: type = F,
};

pub const default_precision = parsePrecision(build_options.precision);
pub const F = default_precision;
pub const Scal = F;
pub const Scalar = Scal;

pub const default_simd = parseSimd(build_options.simd);
pub const config = configForPrecision(F);

pub const SimdWidth = config.simd_vec_width;

// --------------------------------------------------------------------------------------
// Public Enums and Helpers
// --------------------------------------------------------------------------------------

pub const SimdMode = enum {
    off,
    on,
};

pub fn defaultSimdVecWidthForPrecision(comptime precision_type: type) comptime_int {
    const opt_width = comptime buildOptionsSimdVecWidth();
    if (opt_width > 0) {
        return opt_width;
    }
    return switch (precision_type) {
        f32 => 16,
        f64 => 8,
        else => @compileError("Only f32 and f64 precision are supported in Felix."),
    };
}

pub fn configForPrecision(comptime precision_type: type) Config {
    return .{
        .precision = precision_type,
        .simd = default_simd,
        .simd_vec_width = defaultSimdVecWidthForPrecision(precision_type),
    };
}

fn parsePrecision(comptime precision_str: []const u8) type {
    if (std.mem.eql(u8, precision_str, "f32")) {
        return f32;
    }
    if (std.mem.eql(u8, precision_str, "f64")) {
        return f64;
    }
    @compileError("build_options.precision must be \"f32\" or \"f64\".");
}

fn buildOptionsSimdVecWidth() comptime_int {
    if (@hasDecl(build_options, "simd_vec_width")) {
        return build_options.simd_vec_width;
    }
    if (@hasDecl(build_options, "simd_vector_width")) {
        return build_options.simd_vector_width;
    }
    return 0;
}

fn parseSimd(comptime simd_str: []const u8) SimdMode {
    if (std.mem.eql(u8, simd_str, "on")) {
        return .on;
    }
    if (std.mem.eql(u8, simd_str, "off")) {
        return .off;
    }
    @compileError("build_options.simd must be \"on\" or \"off\".");
}

// --------------------------------------------------------------------------------------
// Generic Vector Primitives
// --------------------------------------------------------------------------------------

pub const VecSF = @Vector(SimdWidth, F);
pub const VecSU = @Vector(SimdWidth, usize);
pub const VecSI = @Vector(SimdWidth, isize);
pub const VecSB = @Vector(SimdWidth, bool);
pub const VecSU8 = @Vector(SimdWidth, u8);
