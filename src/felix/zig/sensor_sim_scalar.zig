// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("sensor_sim_common.zig");
const elements = @import("elements.zig");
const mesh_interp = @import("mesh_interp.zig");
const transforms = @import("transforms.zig");
const errors = @import("errors.zig");
const random = @import("random.zig");

const F = common.F;
const SimMeshInput = common.SimMeshInput;
const SensorArrayInput = common.SensorArrayInput;
const ErrorSpec = common.ErrorSpec;
const ElementType = common.ElementType;
const SensorLocation = common.SensorLocation;

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
        var is_found: bool = false;
        var node_count: usize = 0;
        var node_indices: [elements.max_elem_nodes]usize = undefined;
        var weights: [elements.max_elem_nodes]F = undefined;

        if (sensor_in.binding_ptr) |b| {
            is_found = b.found_mask_ptr[ss] != 0;
            node_count = b.node_counts_ptr[ss];
            const base_node_offset = ss * elements.max_elem_nodes;
            for (0..node_count) |nn| {
                node_indices[nn] = b.node_indices_ptr[base_node_offset + nn];
                weights[nn] = b.weights_ptr[base_node_offset + nn];
            }
        } else {
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
            is_found = loc.found;
            node_count = loc.node_count;
            for (0..node_count) |nn| {
                node_indices[nn] = loc.node_indices[nn];
                weights[nn] = loc.weights[nn];
            }
        }

        const sens_base_idx = ss * (num_comps * num_out_times);

        for (0..num_comps) |cc| {
            const comp_out_ptr = out_truth + sens_base_idx + cc * num_out_times;

            if (!is_found) {
                @memset(comp_out_ptr[0..num_out_times], 0.0);
                continue;
            }

            @memset(sim_time_buf[0..num_sim_times], 0.0);
            for (0..node_count) |nn| {
                const weight_val = weights[nn];
                const node_id = node_indices[nn];
                const node_offset = node_id * (num_comps * num_sim_times) + cc * num_sim_times;
                const node_data = mesh_in.nodal_fields_ptr + node_offset;

                for (0..num_sim_times) |tt| {
                    sim_time_buf[tt] += weight_val * node_data[tt];
                }
            }

            if (sensor_in.num_sample_times > 0) {
                mesh_interp.interpTimesLinear(
                    sim_times,
                    sim_time_buf[0..num_sim_times],
                    sensor_in.sample_times_ptr[0..num_out_times],
                    comp_out_ptr[0..num_out_times],
                );
            } else {
                @memcpy(comp_out_ptr[0..num_out_times], sim_time_buf[0..num_out_times]);
            }
        }
    }

    // 2. Coordinate Transformations
    if (sensor_in.num_rot_matrices > 0) {
        for (0..num_sensors) |ss| {
            const rot_idx = if (sensor_in.num_rot_matrices == 1) 0 else ss;
            const rot_mat_ptr = sensor_in.rot_matrices_ptr + rot_idx * 9;
            const base_idx = ss * (num_comps * num_out_times);

            if (sensor_in.is_tensor == 0) {
                if (sensor_in.spatial_dims == 2 and num_comps == 2) {
                    var r22 = [4]F{
                        rot_mat_ptr[0], rot_mat_ptr[1],
                        rot_mat_ptr[3], rot_mat_ptr[4],
                    };
                    const ptr_x = out_truth + base_idx + 0 * num_out_times;
                    const ptr_y = out_truth + base_idx + 1 * num_out_times;
                    for (0..num_out_times) |tt| {
                        var tx: F = undefined;
                        var ty: F = undefined;
                        transforms.transformVector2D(
                            &r22,
                            ptr_x[tt],
                            ptr_y[tt],
                            &tx,
                            &ty,
                        );
                        ptr_x[tt] = tx;
                        ptr_y[tt] = ty;
                    }
                } else if (num_comps == 3) {
                    var r33: [9]F = undefined;
                    for (0..9) |ii| r33[ii] = rot_mat_ptr[ii];
                    const ptr_x = out_truth + base_idx + 0 * num_out_times;
                    const ptr_y = out_truth + base_idx + 1 * num_out_times;
                    const ptr_z = out_truth + base_idx + 2 * num_out_times;
                    for (0..num_out_times) |tt| {
                        var tx: F = undefined;
                        var ty: F = undefined;
                        var tz: F = undefined;
                        transforms.transformVector3D(
                            &r33,
                            ptr_x[tt],
                            ptr_y[tt],
                            ptr_z[tt],
                            &tx,
                            &ty,
                            &tz,
                        );
                        ptr_x[tt] = tx;
                        ptr_y[tt] = ty;
                        ptr_z[tt] = tz;
                    }
                }
            } else {
                if (sensor_in.spatial_dims == 2 and num_comps == 3) {
                    var r22 = [4]F{
                        rot_mat_ptr[0], rot_mat_ptr[1],
                        rot_mat_ptr[3], rot_mat_ptr[4],
                    };
                    const ptr_xx = out_truth + base_idx + 0 * num_out_times;
                    const ptr_yy = out_truth + base_idx + 1 * num_out_times;
                    const ptr_xy = out_truth + base_idx + 2 * num_out_times;
                    for (0..num_out_times) |tt| {
                        var t_xx: F = undefined;
                        var t_yy: F = undefined;
                        var t_xy: F = undefined;
                        transforms.transformTensor2D(
                            &r22,
                            ptr_xx[tt],
                            ptr_yy[tt],
                            ptr_xy[tt],
                            &t_xx,
                            &t_yy,
                            &t_xy,
                        );
                        ptr_xx[tt] = t_xx;
                        ptr_yy[tt] = t_yy;
                        ptr_xy[tt] = t_xy;
                    }
                } else if (num_comps == 6) {
                    var r33: [9]F = undefined;
                    for (0..9) |ii| r33[ii] = rot_mat_ptr[ii];
                    const ptr_xx = out_truth + base_idx + 0 * num_out_times;
                    const ptr_yy = out_truth + base_idx + 1 * num_out_times;
                    const ptr_zz = out_truth + base_idx + 2 * num_out_times;
                    const ptr_xy = out_truth + base_idx + 3 * num_out_times;
                    const ptr_xz = out_truth + base_idx + 4 * num_out_times;
                    const ptr_yz = out_truth + base_idx + 5 * num_out_times;
                    for (0..num_out_times) |tt| {
                        var in_t: [6]F = undefined;
                        in_t[0] = ptr_xx[tt];
                        in_t[1] = ptr_yy[tt];
                        in_t[2] = ptr_zz[tt];
                        in_t[3] = ptr_xy[tt];
                        in_t[4] = ptr_xz[tt];
                        in_t[5] = ptr_yz[tt];

                        var out_t: [6]F = undefined;
                        transforms.transformTensor3D(&r33, &in_t, &out_t);

                        ptr_xx[tt] = out_t[0];
                        ptr_yy[tt] = out_t[1];
                        ptr_zz[tt] = out_t[2];
                        ptr_xy[tt] = out_t[3];
                        ptr_xz[tt] = out_t[4];
                        ptr_yz[tt] = out_t[5];
                    }
                }
            }
        }
    }

    // 3. Error Chain Integration
    var err_total_buf: [16384]F = undefined;
    var err_sys_buf: [16384]F = undefined;
    var err_rand_buf: [16384]F = undefined;

    const use_stack = total_elements <= 16384;
    var heap_err_total: ?[]F = null;
    var heap_err_sys: ?[]F = null;
    var heap_err_rand: ?[]F = null;

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    if (!use_stack) {
        heap_err_total = alloc.alloc(F, total_elements) catch @panic("OOM");
        heap_err_sys = alloc.alloc(F, total_elements) catch @panic("OOM");
        heap_err_rand = alloc.alloc(F, total_elements) catch @panic("OOM");
    }

    const err_total_slice = if (use_stack)
        err_total_buf[0..total_elements]
    else
        heap_err_total.?;
    const err_sys_slice = if (use_stack)
        err_sys_buf[0..total_elements]
    else
        heap_err_sys.?;
    const err_rand_slice = if (use_stack)
        err_rand_buf[0..total_elements]
    else
        heap_err_rand.?;

    @memset(err_total_slice, 0.0);
    @memset(err_sys_slice, 0.0);
    @memset(err_rand_slice, 0.0);

    @memcpy(out_measurements[0..total_elements], out_truth[0..total_elements]);

    if (sensor_in.work_positions_ptr != null and sensor_in.scratch_positions_ptr != null) {
        @memcpy(
            sensor_in.work_positions_ptr[0 .. num_sensors * 3],
            sensor_in.positions_ptr[0 .. num_sensors * 3],
        );
        if (sensor_in.work_rot_matrices_ptr != null) {
            if (sensor_in.num_rot_matrices > 0 and sensor_in.rot_matrices_ptr != null) {
                for (0..num_sensors) |ss| {
                    const src_idx = if (sensor_in.num_rot_matrices == 1) 0 else ss;
                    @memcpy(
                        sensor_in.work_rot_matrices_ptr[ss * 9 .. (ss + 1) * 9],
                        sensor_in.rot_matrices_ptr[src_idx * 9 .. (src_idx + 1) * 9],
                    );
                }
            } else {
                for (0..num_sensors) |ss| {
                    common.setIdentity(sensor_in.work_rot_matrices_ptr + ss * 9);
                }
            }
        }
        if (sensor_in.work_times_ptr != null) {
            if (sensor_in.num_sample_times > 0 and sensor_in.sample_times_ptr != null) {
                @memcpy(
                    sensor_in.work_times_ptr[0..num_out_times],
                    sensor_in.sample_times_ptr[0..num_out_times],
                );
            } else if (mesh_in.sim_times_ptr != null) {
                @memcpy(
                    sensor_in.work_times_ptr[0..num_sim_times],
                    mesh_in.sim_times_ptr[0..num_sim_times],
                );
            }
        }
    }

    var rand_streams: [64]common.RandomStream = undefined;
    for (0..num_errors) |ee| {
        const spec = error_specs_ptr[ee];
        if (spec.dist_type != 0) {
            const seed = if (spec.has_seed != 0)
                spec.seed
            else
                123456789 + @as(u64, @intCast(ee));
            rand_streams[ee] = common.RandomStream.init(seed +% seed_offset);
        }

        if (spec.kind == 12) {
            if (spec.field_spec_ptr != null) {
                common.applyFieldPerturbations(
                    mesh_in,
                    sensor_in,
                    spec.field_spec_ptr,
                    ee,
                    seed_offset,
                    spec.err_dep == 1,
                );
            }
        }

        for (0..num_sensors) |ss| {
            var sensor_rand_val: F = 0.0;
            if (spec.kind == 2 and spec.err_type == 0 and spec.dist_type != 0) {
                sensor_rand_val = common.sampleDistribution(
                    &rand_streams[ee],
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
                        out_measurements[idx]
                    else
                        out_truth[idx];

                    const t_val = if (sensor_in.num_sample_times > 0)
                        sensor_in.sample_times_ptr[tt]
                    else
                        mesh_in.sim_times_ptr[tt];

                    var err_val: F = 0.0;
                    switch (spec.kind) {
                        0 => {
                            err_val = spec.param0;
                        },
                        1 => {
                            err_val = (spec.param0 / 100.0) * basis_val;
                        },
                        2 => {
                            err_val = sensor_rand_val;
                        },
                        3 => {
                            err_val = (sensor_rand_val / 100.0) * basis_val;
                        },
                        4 => {
                            if (spec.dist_type != 0) {
                                err_val = common.sampleDistribution(
                                    &rand_streams[ee],
                                    spec.dist_type,
                                    spec.param0,
                                    spec.param1,
                                    spec.param2,
                                );
                            }
                        },
                        5 => {
                            if (spec.dist_type != 0) {
                                const r_samp = common.sampleDistribution(
                                    &rand_streams[ee],
                                    spec.dist_type,
                                    spec.param0,
                                    spec.param1,
                                    spec.param2,
                                );
                                err_val = (r_samp / 100.0) * basis_val;
                            }
                        },
                        6 => {
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
                            const meas_min = spec.param0;
                            const meas_max = spec.param1;
                            const clamped = std.math.clamp(basis_val, meas_min, meas_max);
                            err_val = clamped - basis_val;
                        },
                        9 => {
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
                            const rate = spec.param0;
                            const t0 = spec.param1;
                            const offset = spec.param2;
                            err_val = rate * (t_val - t0) + offset;
                        },
                        11 => {
                            if (spec.poly_coeffs_ptr != null and spec.poly_coeffs_len > 0) {
                                const coeffs = spec.poly_coeffs_ptr[0..spec.poly_coeffs_len];
                                const t0 = spec.param0;
                                err_val = errors.evalPoly(coeffs, t_val - t0);
                            }
                        },
                        12 => {
                            var sampled: [6]F = [_]F{0.0} ** 6;
                            common.samplePerturbedSensor(
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

                    out_measurements[idx] += err_val;
                    err_total_slice[idx] += err_val;

                    if (spec.err_type == 0) {
                        err_sys_slice[idx] += err_val;
                    } else {
                        err_rand_slice[idx] += err_val;
                    }
                }
            }
        }
    }

    if (out_errs_sys) |ptr| {
        if (ptr != null) @memcpy(ptr[0..total_elements], err_sys_slice[0..total_elements]);
    }
    if (out_errs_rand) |ptr| {
        if (ptr != null) @memcpy(ptr[0..total_elements], err_rand_slice[0..total_elements]);
    }
    if (out_errs_total) |ptr| {
        if (ptr != null) @memcpy(ptr[0..total_elements], err_total_slice[0..total_elements]);
    }
}
