// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const common = @import("sensor_sim_common.zig");
const sensor_sim_scalar = @import("sensor_sim_scalar.zig");
const sensor_sim_simd = @import("sensor_sim_simd.zig");

const cfg = buildconfig.config;
const sensor_sim_impl = if (cfg.simd == .on) sensor_sim_simd else sensor_sim_scalar;

// --------------------------------------------------------------------------------------
// Re-exported Types & Structs
// --------------------------------------------------------------------------------------

pub const F = common.F;
pub const SimdWidth = common.SimdWidth;
pub const VecSF = common.VecSF;
pub const SimMeshInput = common.SimMeshInput;
pub const SensorArrayInput = common.SensorArrayInput;
pub const ErrorSpec = common.ErrorSpec;
pub const ElementType = common.ElementType;
pub const SensorLocation = common.SensorLocation;
pub const SensorMeshBinding = common.SensorMeshBinding;
pub const bindSensorsToMesh = common.bindSensorsToMesh;
pub const freeSensorMeshBinding = common.freeSensorMeshBinding;

// --------------------------------------------------------------------------------------
// Public Simulation Execution
// --------------------------------------------------------------------------------------

pub const runSensorSimulation = sensor_sim_impl.runSensorSimulation;
