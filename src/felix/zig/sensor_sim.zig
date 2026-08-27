// --------------------------------------------------------------------------------------
// Felix: A High Performance Sensor Simulation Core
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const elements = @import("elements.zig");
const mesh_interp = @import("mesh_interp.zig");
const transforms = @import("transforms.zig");
const errors = @import("errors.zig");
const random = @import("random.zig");

const F: type = f64;
const ElementType = elements.ElementType;
const ErrorSpec = errors.ErrorSpec;
const RandomStream = random.RandomStream;
const SensorLocation = mesh_interp.SensorLocation;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const SimMeshInput = extern struct {
    coords_ptr: [*c]const F,
    num_nodes: usize,
    connect_ptr: [*c]const usize,
    num_elements: usize,
    elem_type: u32,
    nodal_fields_ptr: [*c]const F,
    num_components: usize,
    sim_times_ptr: [*c]const F,
    num_sim_times: usize,
};

pub const SensorArrayInput = extern struct {
    positions_ptr: [*c]const F,
    num_sensors: usize,
    sample_times_ptr: [*c]const F,
    num_sample_times: usize,
    rot_matrices_ptr: [*c]const F,
    num_rot_matrices: usize,
    spatial_dims: u32,
    is_tensor: u32,
    work_positions_ptr: [*c]F,
    work_times_ptr: [*c]F,
    work_rot_matrices_ptr: [*c]F,
    scratch_positions_ptr: [*c]F,
    scratch_times_ptr: [*c]F,
    scratch_rot_matrices_ptr: [*c]F,
};

// --------------------------------------------------------------------------------------
// Public Simulation Entry Point
// --------------------------------------------------------------------------------------

pub fn runSensorSimulation(
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    error_specs_ptr: [*c]const ErrorSpec,
    num_errors: usize,
    out_truth: [*c]F,
    out_measurements: [*c]F,
    out_errs_sys: ?[*c]F,
    out_errs_rand: ?[*c]F,
    out_errs_total: ?[*c]F,
    seed_offset: u64,
) void {
    const num_sensors = sensor_in.num_sensors;
    const num_comps = mesh_in.num_components;
    const num_out_times = if (sensor_in.num_sample_times > 0)
        sensor_in.num_sample_times
    else
        mesh_in.num_sim_times;

    const total_elements = num_sensors * num_comps * num_out_times;

    // 1. Calculate Ground Truth
    var sim_time_buf: [256]F = undefined;
    const num_sim_times = mesh_in.num_sim_times;
    const sim_times = mesh_in.sim_times_ptr[0..num_sim_times];

    const elem_type: ElementType = @enumFromInt(mesh_in.elem_type);

    for (0..num_sensors) |ss| {
        const px = sensor_in.positions_ptr[ss * 3 + 0];
        const py = sensor_in.positions_ptr[ss * 3 + 1];
        const pz = sensor_in.positions_ptr[ss * 3 + 2];

        var loc: SensorLocation = undefined;
        mesh_interp.locatePointInMesh(
            mesh_in.coords_ptr,
            mesh_in.connect_ptr,
            mesh_in.num_elements,
            elem_type,
            px,
            py,
            pz,
            &loc,
        );

        for (0..num_comps) |cc| {
            if (!loc.found) {
                for (0..num_out_times) |tt| {
                    const out_idx = ss * (num_comps * num_out_times) + cc * num_out_times + tt;
                    out_truth[out_idx] = 0.0;
                }
                continue;
            }

            // Sample across sim time steps
            for (0..num_sim_times) |tt| {
                var field_val: F = 0.0;
                for (0..loc.node_count) |nn| {
                    const nid = loc.node_indices[nn];
                    const node_field_idx = nid * (num_comps * num_sim_times) +
                        cc * num_sim_times + tt;
                    field_val += loc.weights[nn] * mesh_in.nodal_fields_ptr[node_field_idx];
                }
                sim_time_buf[tt] = field_val;
            }

            // Interpolate to output sample times
            if (sensor_in.num_sample_times > 0) {
                for (0..num_out_times) |tt| {
                    const targ_time = sensor_in.sample_times_ptr[tt];
                    const interp_val = mesh_interp.interpTimeLinear(
                        sim_times,
                        sim_time_buf[0..num_sim_times],
                        targ_time,
                    );
                    const out_idx = ss * (num_comps * num_out_times) + cc * num_out_times + tt;
                    out_truth[out_idx] = interp_val;
                }
            } else {
                for (0..num_out_times) |tt| {
                    const out_idx = ss * (num_comps * num_out_times) + cc * num_out_times + tt;
                    out_truth[out_idx] = sim_time_buf[tt];
                }
            }
        }
    }

    // 2. Coordinate Transformations (Sensor Angles / Rotations)
    if (sensor_in.num_rot_matrices > 0) {
        for (0..num_sensors) |ss| {
            const rot_idx = if (sensor_in.num_rot_matrices == 1) 0 else ss;
            const rot_mat_ptr = sensor_in.rot_matrices_ptr + rot_idx * 9;

            if (sensor_in.is_tensor == 0) {
                if (sensor_in.spatial_dims == 2 and num_comps == 2) {
                    var r22 = [4]F{
                        rot_mat_ptr[0], rot_mat_ptr[1],
                        rot_mat_ptr[3], rot_mat_ptr[4],
                    };
                    for (0..num_out_times) |tt| {
                        const base_idx = ss * (num_comps * num_out_times) + tt;
                        const idx_x = base_idx + 0 * num_out_times;
                        const idx_y = base_idx + 1 * num_out_times;
                        var tx: F = undefined;
                        var ty: F = undefined;
                        transforms.transformVector2D(
                            &r22,
                            out_truth[idx_x],
                            out_truth[idx_y],
                            &tx,
                            &ty,
                        );
                        out_truth[idx_x] = tx;
                        out_truth[idx_y] = ty;
                    }
                } else if (num_comps == 3) {
                    var r33: [9]F = undefined;
                    for (0..9) |ii| r33[ii] = rot_mat_ptr[ii];
                    for (0..num_out_times) |tt| {
                        const base_idx = ss * (num_comps * num_out_times) + tt;
                        const idx_x = base_idx + 0 * num_out_times;
                        const idx_y = base_idx + 1 * num_out_times;
                        const idx_z = base_idx + 2 * num_out_times;
                        var tx: F = undefined;
                        var ty: F = undefined;
                        var tz: F = undefined;
                        transforms.transformVector3D(
                            &r33,
                            out_truth[idx_x],
                            out_truth[idx_y],
                            out_truth[idx_z],
                            &tx,
                            &ty,
                            &tz,
                        );
                        out_truth[idx_x] = tx;
                        out_truth[idx_y] = ty;
                        out_truth[idx_z] = tz;
                    }
                }
            } else {
                // Tensor transformation
                if (sensor_in.spatial_dims == 2 and num_comps == 3) {
                    var r22 = [4]F{
                        rot_mat_ptr[0], rot_mat_ptr[1],
                        rot_mat_ptr[3], rot_mat_ptr[4],
                    };
                    for (0..num_out_times) |tt| {
                        const base_idx = ss * (num_comps * num_out_times) + tt;
                        const idx_xx = base_idx + 0 * num_out_times;
                        const idx_yy = base_idx + 1 * num_out_times;
                        const idx_xy = base_idx + 2 * num_out_times;
                        var t_xx: F = undefined;
                        var t_yy: F = undefined;
                        var t_xy: F = undefined;
                        transforms.transformTensor2D(
                            &r22,
                            out_truth[idx_xx],
                            out_truth[idx_yy],
                            out_truth[idx_xy],
                            &t_xx,
                            &t_yy,
                            &t_xy,
                        );
                        out_truth[idx_xx] = t_xx;
                        out_truth[idx_yy] = t_yy;
                        out_truth[idx_xy] = t_xy;
                    }
                } else if (num_comps == 6) {
                    var r33: [9]F = undefined;
                    for (0..9) |ii| r33[ii] = rot_mat_ptr[ii];
                    for (0..num_out_times) |tt| {
                        var in_t: [6]F = undefined;
                        const base_idx = ss * (num_comps * num_out_times) + tt;
                        for (0..6) |cc| {
                            const idx = base_idx + cc * num_out_times;
                            in_t[cc] = out_truth[idx];
                        }
                        var out_t: [6]F = undefined;
                        transforms.transformTensor3D(&r33, &in_t, &out_t);
                        for (0..6) |cc| {
                            const idx = base_idx + cc * num_out_times;
                            out_truth[idx] = out_t[cc];
                        }
                    }
                }
            }
        }
    }

    // 3. Error Chain Integration
    var err_total_buf: [16384]F = undefined;
    var err_sys_buf: [16384]F = undefined;
    var err_rand_buf: [16384]F = undefined;

    const total_slice = if (total_elements <= 16384)
        err_total_buf[0..total_elements]
    else
        out_measurements[0..total_elements];

    const sys_slice = if (out_errs_sys) |ptr|
        ptr[0..total_elements]
    else if (total_elements <= 16384)
        err_sys_buf[0..total_elements]
    else
        total_slice;

    const rand_slice = if (out_errs_rand) |ptr|
        ptr[0..total_elements]
    else if (total_elements <= 16384)
        err_rand_buf[0..total_elements]
    else
        total_slice;

    @memset(total_slice, 0.0);
    if (out_errs_sys != null) @memset(sys_slice, 0.0);
    if (out_errs_rand != null) @memset(rand_slice, 0.0);

    @memcpy(
        sensor_in.work_positions_ptr[0 .. num_sensors * 3],
        sensor_in.positions_ptr[0 .. num_sensors * 3],
    );
    if (sensor_in.num_sample_times > 0) {
        @memcpy(
            sensor_in.work_times_ptr[0..num_out_times],
            sensor_in.sample_times_ptr[0..num_out_times],
        );
    } else {
        @memcpy(
            sensor_in.work_times_ptr[0..num_out_times],
            mesh_in.sim_times_ptr[0..num_out_times],
        );
    }
    initialiseRotations(sensor_in);

    for (0..num_errors) |ee| {
        const spec = error_specs_ptr[ee];
        var stream = if (spec.has_seed != 0)
            RandomStream.init(spec.seed +% seed_offset)
        else
            RandomStream.init(123456789 + @as(u64, ee) +% seed_offset);

        if (spec.kind == 12 and spec.field_spec_ptr != null) {
            prepareFieldPerturbation(
                mesh_in,
                sensor_in,
                spec.field_spec_ptr,
                spec.err_dep == 1,
                ee,
                seed_offset,
            );
        }

        // Evaluate error values for each sensor, component, time
        for (0..num_sensors) |ss| {
            // For systematic generator (ErrSysGen), sample once per sensor
            // (or per sensor-comp)
            var sensor_rand_val: F = 0.0;
            if (spec.kind == 2 and spec.err_type == 0) {
                // Systematic generator
                sensor_rand_val = sampleDistribution(
                    &stream,
                    spec.dist_type,
                    spec.param0,
                    spec.param1,
                    spec.param2,
                );
            }

            for (0..num_comps) |cc| {
                for (0..num_out_times) |tt| {
                    const idx = ss * (num_comps * num_out_times) + cc * num_out_times + tt;
                    const basis_val = if (spec.err_dep == 1)
                        out_truth[idx] + total_slice[idx]
                    else
                        out_truth[idx];

                    const t_val = if (sensor_in.num_sample_times > 0)
                        sensor_in.sample_times_ptr[tt]
                    else
                        mesh_in.sim_times_ptr[tt];

                    var err_val: F = 0.0;

                    switch (spec.kind) {
                        0 => {
                            // Constant offset
                            err_val = spec.param0;
                        },
                        1 => {
                            // Percentage offset
                            err_val = (spec.param0 / 100.0) * basis_val;
                        },
                        2 => {
                            // Systematic generator (ErrSysGen)
                            err_val = sensor_rand_val;
                        },
                        3 => {
                            // Systematic generator percentage (ErrSysGenPercent)
                            err_val = (sensor_rand_val / 100.0) * basis_val;
                        },
                        4 => {
                            // Random generator (ErrRandGen)
                            const r_sample = sampleDistribution(
                                &stream,
                                spec.dist_type,
                                spec.param0,
                                spec.param1,
                                spec.param2,
                            );
                            err_val = r_sample;
                        },
                        5 => {
                            // Random generator percentage (ErrRandGenPercent)
                            const r_sample = sampleDistribution(
                                &stream,
                                spec.dist_type,
                                spec.param0,
                                spec.param1,
                                spec.param2,
                            );
                            err_val = (r_sample / 100.0) * basis_val;
                        },
                        6 => {
                            // Roundoff
                            const round_u32 = @as(u32, @intFromFloat(spec.param1));
                            const round_method: errors.RoundMethod = @enumFromInt(round_u32);
                            const base_val = spec.param0;
                            const rounded = errors.evalRound(
                                basis_val,
                                base_val,
                                round_method,
                            );
                            err_val = rounded - basis_val;
                        },
                        7 => {
                            // Digitisation
                            const units_per_bit = spec.param0;
                            const round_u32 = @as(u32, @intFromFloat(spec.param1));
                            const round_method: errors.RoundMethod = @enumFromInt(round_u32);
                            const digitised = errors.evalRound(
                                basis_val,
                                units_per_bit,
                                round_method,
                            );
                            err_val = digitised - basis_val;
                        },
                        8 => {
                            // Saturation
                            const meas_min = spec.param0;
                            const meas_max = spec.param1;
                            const clamped = std.math.clamp(basis_val, meas_min, meas_max);
                            err_val = clamped - basis_val;
                        },
                        9 => {
                            // Calibration table
                            if (spec.table_ptr != null and spec.table_rows > 0) {
                                const table_slice = spec.table_ptr[0 .. spec.table_rows * 2];
                                const assumed_y = errors.evalTableLookup1D(
                                    table_slice,
                                    spec.table_rows,
                                    basis_val,
                                );
                                err_val = assumed_y - basis_val;
                            }
                        },
                        10 => {
                            // Linear drift: rate * (t - t0) + offset
                            const rate = spec.param0;
                            const t0 = spec.param1;
                            const offset = spec.param2;
                            err_val = rate * (t_val - t0) + offset;
                        },
                        11 => {
                            // Polynomial drift
                            if (spec.poly_coeffs_ptr != null and spec.poly_coeffs_len > 0) {
                                const coeffs = spec.poly_coeffs_ptr[0..spec.poly_coeffs_len];
                                const t0 = spec.param0;
                                err_val = errors.evalPoly(coeffs, t_val - t0);
                            }
                        },
                        12 => {
                            var sampled: [6]F = [_]F{0.0} ** 6;
                            samplePerturbedSensor(
                                mesh_in,
                                sensor_in,
                                spec.field_spec_ptr,
                                ss,
                                tt,
                                &sampled,
                            );
                            err_val = sampled[cc] - basis_val;
                        },
                        else => {},
                    }

                    if (out_errs_sys != null and spec.err_type == 0) {
                        sys_slice[idx] += err_val;
                    }
                    if (out_errs_rand != null and spec.err_type == 1) {
                        rand_slice[idx] += err_val;
                    }

                    total_slice[idx] += err_val;
                }
            }
        }
    }

    // 4. Calculate Final Measurements
    for (0..total_elements) |ii| {
        out_measurements[ii] = out_truth[ii] + total_slice[ii];
        if (out_errs_total) |ptr| {
            ptr[ii] = total_slice[ii];
        }
    }
}

fn initialiseRotations(sensor_in: *const SensorArrayInput) void {
    for (0..sensor_in.num_sensors) |ss| {
        const out_mat = sensor_in.work_rot_matrices_ptr + ss * 9;
        if (sensor_in.num_rot_matrices > 0) {
            const rot_idx = if (sensor_in.num_rot_matrices == 1) 0 else ss;
            const in_mat = sensor_in.rot_matrices_ptr + rot_idx * 9;
            @memcpy(out_mat[0..9], in_mat[0..9]);
        } else {
            setIdentity(out_mat);
        }
    }
}

fn prepareFieldPerturbation(
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    field_spec: *const errors.FieldPerturbationSpec,
    is_dependent: bool,
    error_index: usize,
    seed_offset: u64,
) void {
    const num_sensors = sensor_in.num_sensors;
    const num_times = if (sensor_in.num_sample_times > 0)
        sensor_in.num_sample_times
    else
        mesh_in.num_sim_times;
    const source_pos = if (is_dependent)
        sensor_in.work_positions_ptr
    else
        sensor_in.positions_ptr;
    const source_times = if (is_dependent)
        sensor_in.work_times_ptr
    else if (sensor_in.num_sample_times > 0)
        sensor_in.sample_times_ptr
    else
        mesh_in.sim_times_ptr;
    const source_rots = if (is_dependent)
        sensor_in.work_rot_matrices_ptr
    else
        null;

    var pos_streams: [3]RandomStream = undefined;
    var angle_streams: [3]RandomStream = undefined;
    for (0..3) |aa| {
        pos_streams[aa] = initDistributionStream(
            field_spec.pos_rand[aa],
            error_index,
            aa,
            seed_offset,
        );
        angle_streams[aa] = initDistributionStream(
            field_spec.angle_rand[aa],
            error_index,
            aa + 3,
            seed_offset,
        );
    }
    var time_stream = initDistributionStream(
        field_spec.time_rand,
        error_index,
        6,
        seed_offset,
    );

    for (0..num_sensors) |ss| {
        for (0..3) |aa| {
            const idx = ss * 3 + aa;
            var val = source_pos[idx];
            if (field_spec.pos_offset_ptr != null) val += field_spec.pos_offset_ptr[idx];
            const dist = field_spec.pos_rand[aa];
            if (dist.dist_type != 0) {
                val += sampleDistribution(
                    &pos_streams[aa],
                    dist.dist_type,
                    dist.param0,
                    dist.param1,
                    dist.param2,
                );
            }
            if (field_spec.pos_lock_ptr != null and field_spec.pos_lock_ptr[idx] != 0) {
                val = source_pos[idx];
            }
            sensor_in.scratch_positions_ptr[idx] = val;
        }

        const nominal = if (source_rots) |rots|
            rots + ss * 9
        else if (sensor_in.num_rot_matrices > 0)
            sensor_in.rot_matrices_ptr +
                (if (sensor_in.num_rot_matrices == 1) 0 else ss * 9)
        else
            null;
        var angles = [3]F{ 0.0, 0.0, 0.0 };
        for (0..3) |aa| {
            const idx = ss * 3 + aa;
            if (field_spec.angle_offset_ptr != null) angles[aa] += field_spec.angle_offset_ptr[idx];
            const dist = field_spec.angle_rand[aa];
            if (dist.dist_type != 0) {
                angles[aa] += sampleDistribution(
                    &angle_streams[aa],
                    dist.dist_type,
                    dist.param0,
                    dist.param1,
                    dist.param2,
                );
            }
            if (field_spec.angle_lock_ptr != null and field_spec.angle_lock_ptr[idx] != 0) {
                angles[aa] = 0.0;
            }
        }
        buildPerturbedRotation(
            nominal,
            angles,
            sensor_in.scratch_rot_matrices_ptr + ss * 9,
        );
    }

    for (0..num_times) |tt| {
        var val = source_times[tt];
        if (field_spec.time_offset_ptr != null) val += field_spec.time_offset_ptr[tt];
        if (field_spec.time_rand.dist_type != 0) {
            const dist = field_spec.time_rand;
            val += sampleDistribution(
                &time_stream,
                dist.dist_type,
                dist.param0,
                dist.param1,
                dist.param2,
            );
        }
        val += calcFieldDrift(field_spec, val);
        sensor_in.scratch_times_ptr[tt] = val;
    }

    if (is_dependent) {
        @memcpy(
            sensor_in.work_positions_ptr[0 .. num_sensors * 3],
            sensor_in.scratch_positions_ptr[0 .. num_sensors * 3],
        );
        @memcpy(
            sensor_in.work_rot_matrices_ptr[0 .. num_sensors * 9],
            sensor_in.scratch_rot_matrices_ptr[0 .. num_sensors * 9],
        );
        if (num_times > 0) {
            @memcpy(
                sensor_in.work_times_ptr[0..num_times],
                sensor_in.scratch_times_ptr[0..num_times],
            );
        }
    }
}

fn samplePerturbedSensor(
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    field_spec: *const errors.FieldPerturbationSpec,
    sensor_index: usize,
    time_index: usize,
    out_values: *[6]F,
) void {
    const num_comps = mesh_in.num_components;
    const num_out_times = if (sensor_in.num_sample_times > 0)
        sensor_in.num_sample_times
    else
        mesh_in.num_sim_times;
    _ = num_out_times;
    const position = sensor_in.scratch_positions_ptr + sensor_index * 3;
    const rotation = sensor_in.scratch_rot_matrices_ptr + sensor_index * 9;
    const sample_time = sensor_in.scratch_times_ptr[time_index];

    const point_count = spatialPointCount(field_spec.spatial_kind);
    var weight_sum: F = 0.0;
    for (0..point_count) |pp| {
        var offset: [3]F = undefined;
        const weight = spatialPoint(
            field_spec,
            pp,
            &offset,
        );
        const world_x = position[0] + rotation[0] * offset[0] +
            rotation[3] * offset[1] + rotation[6] * offset[2];
        const world_y = position[1] + rotation[1] * offset[0] +
            rotation[4] * offset[1] + rotation[7] * offset[2];
        const world_z = position[2] + rotation[2] * offset[0] +
            rotation[5] * offset[1] + rotation[8] * offset[2];
        for (0..num_comps) |cc| {
            out_values[cc] += weight * sampleMeshComponent(
                mesh_in,
                cc,
                sample_time,
                world_x,
                world_y,
                world_z,
            );
        }
        weight_sum += weight;
    }
    if (weight_sum != 0.0) {
        for (0..num_comps) |cc| out_values[cc] /= weight_sum;
    }
    transformSample(sensor_in, num_comps, rotation, out_values);
}

fn sampleMeshComponent(
    mesh_in: *const SimMeshInput,
    component: usize,
    sample_time: F,
    px: F,
    py: F,
    pz: F,
) F {
    const elem_type: ElementType = @enumFromInt(mesh_in.elem_type);
    var loc: SensorLocation = undefined;
    mesh_interp.locatePointInMesh(
        mesh_in.coords_ptr,
        mesh_in.connect_ptr,
        mesh_in.num_elements,
        elem_type,
        px,
        py,
        pz,
        &loc,
    );
    if (!loc.found) return 0.0;

    var values: [256]F = undefined;
    for (0..mesh_in.num_sim_times) |tt| {
        var val: F = 0.0;
        for (0..loc.node_count) |nn| {
            const node = loc.node_indices[nn];
            const idx = node * (mesh_in.num_components * mesh_in.num_sim_times) +
                component * mesh_in.num_sim_times + tt;
            val += loc.weights[nn] * mesh_in.nodal_fields_ptr[idx];
        }
        values[tt] = val;
    }
    return mesh_interp.interpTimeLinear(
        mesh_in.sim_times_ptr[0..mesh_in.num_sim_times],
        values[0..mesh_in.num_sim_times],
        sample_time,
    );
}

fn initDistributionStream(
    spec: errors.DistributionSpec,
    error_index: usize,
    stream_index: usize,
    seed_offset: u64,
) RandomStream {
    const seed = if (spec.has_seed != 0)
        spec.seed
    else
        123456789 + @as(u64, @intCast(error_index * 7 + stream_index));
    return RandomStream.init(seed +% seed_offset);
}

fn calcFieldDrift(
    spec: *const errors.FieldPerturbationSpec,
    time: F,
) F {
    return switch (spec.drift_kind) {
        1 => spec.drift_param0,
        2 => spec.drift_param0 * (time - spec.drift_param1) + spec.drift_param2,
        3 => if (spec.drift_poly_ptr != null)
            errors.evalPoly(
                spec.drift_poly_ptr[0..spec.drift_poly_len],
                time - spec.drift_param0,
            )
        else
            0.0,
        else => 0.0,
    };
}

fn buildPerturbedRotation(
    nominal: ?[*c]const F,
    angles_degrees: [3]F,
    out: [*c]F,
) void {
    const degrees_to_radians = std.math.pi / 180.0;
    const zz = angles_degrees[0] * degrees_to_radians;
    const yy = angles_degrees[1] * degrees_to_radians;
    const xx = angles_degrees[2] * degrees_to_radians;
    const cz = @cos(zz);
    const sz = @sin(zz);
    const cy = @cos(yy);
    const sy = @sin(yy);
    const cx = @cos(xx);
    const sx = @sin(xx);

    const perturb_t = [9]F{
        cy * cz,                cy * sz,                -sy,
        sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy,
        cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy,
    };
    if (nominal) |nominal_mat| {
        multiplyMatrices(nominal_mat, &perturb_t, out);
    } else {
        @memcpy(out[0..9], perturb_t[0..9]);
    }
}

fn multiplyMatrices(lhs: [*c]const F, rhs: *const [9]F, out: [*c]F) void {
    for (0..3) |rr| {
        for (0..3) |cc| {
            var val: F = 0.0;
            for (0..3) |kk| val += lhs[rr * 3 + kk] * rhs[kk * 3 + cc];
            out[rr * 3 + cc] = val;
        }
    }
}

fn setIdentity(out: [*c]F) void {
    const identity = [9]F{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    @memcpy(out[0..9], identity[0..9]);
}

fn spatialPointCount(kind: u32) usize {
    return switch (kind) {
        2, 4 => 4,
        3, 5 => 9,
        else => 1,
    };
}

fn spatialPoint(
    spec: *const errors.FieldPerturbationSpec,
    point_index: usize,
    out_offset: *[3]F,
) F {
    const half = [_]F{ -0.5, 0.5 };
    const third = [_]F{ -1.0 / 3.0, 0.0, 1.0 / 3.0 };
    const gauss_2 = [_]F{ -1.0 / @sqrt(3.0), 1.0 / @sqrt(3.0) };
    const root_06 = @sqrt(0.6);
    const gauss_3 = [_]F{ -root_06, root_06, 0.0 };

    var x_factor: F = 0.0;
    var y_factor: F = 0.0;
    var weight: F = 1.0;
    switch (spec.spatial_kind) {
        2 => {
            x_factor = half[point_index / 2];
            y_factor = half[point_index % 2];
        },
        3 => {
            x_factor = third[point_index / 3];
            y_factor = third[point_index % 3];
        },
        4 => {
            x_factor = gauss_2[point_index / 2];
            y_factor = gauss_2[point_index % 2];
        },
        5 => {
            const pairs = [9][2]usize{
                .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 },
                .{ 0, 2 }, .{ 2, 0 }, .{ 2, 1 }, .{ 1, 2 },
                .{ 2, 2 },
            };
            x_factor = gauss_3[pairs[point_index][0]];
            y_factor = gauss_3[pairs[point_index][1]];
            weight = if (point_index < 4)
                25.0 / 81.0
            else if (point_index < 8)
                40.0 / 81.0
            else
                64.0 / 81.0;
        },
        else => {},
    }
    out_offset.* = .{
        x_factor * spec.spatial_dim_x,
        y_factor * spec.spatial_dim_y,
        0.0 * spec.spatial_dim_z,
    };
    return weight;
}

fn transformSample(
    sensor_in: *const SensorArrayInput,
    num_components: usize,
    rotation: [*c]const F,
    values: *[6]F,
) void {
    if (sensor_in.is_tensor == 0) {
        if (sensor_in.spatial_dims == 2 and num_components == 2) {
            const r22 = [4]F{ rotation[0], rotation[1], rotation[3], rotation[4] };
            var xx: F = undefined;
            var yy: F = undefined;
            transforms.transformVector2D(&r22, values[0], values[1], &xx, &yy);
            values[0] = xx;
            values[1] = yy;
        } else if (num_components == 3) {
            var r33: [9]F = undefined;
            @memcpy(r33[0..9], rotation[0..9]);
            var xx: F = undefined;
            var yy: F = undefined;
            var zz: F = undefined;
            transforms.transformVector3D(
                &r33,
                values[0],
                values[1],
                values[2],
                &xx,
                &yy,
                &zz,
            );
            values[0] = xx;
            values[1] = yy;
            values[2] = zz;
        }
    } else if (sensor_in.spatial_dims == 2) {
        const r22 = [4]F{ rotation[0], rotation[1], rotation[3], rotation[4] };
        var xx: F = undefined;
        var yy: F = undefined;
        var xy: F = undefined;
        transforms.transformTensor2D(
            &r22,
            values[0],
            values[1],
            values[2],
            &xx,
            &yy,
            &xy,
        );
        values[0] = xx;
        values[1] = yy;
        values[2] = xy;
    } else {
        var r33: [9]F = undefined;
        @memcpy(r33[0..9], rotation[0..9]);
        var transformed: [6]F = undefined;
        transforms.transformTensor3D(&r33, values, &transformed);
        values.* = transformed;
    }
}

fn sampleDistribution(
    stream: *RandomStream,
    dist_type_val: u32,
    p0: F,
    p1: F,
    p2: F,
) F {
    const dist: errors.DistType = @enumFromInt(dist_type_val);
    return switch (dist) {
        .none => 0.0,
        .uniform => stream.uniform(p0, p1),
        .normal => stream.normal(p0, p1),
        .triangular => stream.triangular(p0, p1, p2),
        .exponential => stream.exponential(p0),
        .gamma => stream.gamma(p0, p1),
        .beta => stream.beta(p0, p1),
        .lognormal => stream.logNormal(p0, p1),
    };
}
