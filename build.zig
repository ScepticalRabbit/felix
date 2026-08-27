// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const precision = b.option([]const u8, "precision", "Precision: f64 or f32") orelse "f64";
    const simd = b.option([]const u8, "simd", "SIMD mode: on or off") orelse "on";
    const simd_vector_width = b.option(
        u32,
        "simd-vector-width",
        "SIMD vector width (0 for default)",
    ) orelse 0;

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "precision", precision);
    build_options.addOption([]const u8, "simd", simd);
    build_options.addOption(u32, "simd_vector_width", simd_vector_width);
    const build_options_mod = build_options.createModule();

    const felix_mod = b.addModule("felix", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    const shared_lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "c_felix",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/felix/zig/c-abi.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    b.installArtifact(shared_lib);

    const test_files = [_][]const u8{
        "src/felix/zig/transforms_simd.zig",
        "src/felix/zig/stats_simd.zig",
        "src/felix/zig/errors_simd.zig",
        "src/felix/zig/mesh_interp_simd.zig",
    };

    const test_step = b.step("test", "Run all Felix unit and parity tests");

    for (test_files) |test_path| {
        const test_exe = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(test_path),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .imports = &.{
                    .{ .name = "build_options", .module = build_options_mod },
                    .{ .name = "felix", .module = felix_mod },
                },
            }),
        });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }
}
