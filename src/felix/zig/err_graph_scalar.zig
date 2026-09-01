// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const common = @import("err_graph_common.zig");
const sensor_common = @import("sensor_sim_common.zig");
const errors = @import("errors.zig");

const F = common.F;
const ErrOp = common.ErrOp;
const ErrGraphSpec = common.ErrGraphSpec;
const SimMeshInput = sensor_common.SimMeshInput;
const SensorArrayInput = sensor_common.SensorArrayInput;

pub fn runErrGraphSimulation(
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    graph_spec: *const ErrGraphSpec,
    truth_values: [*c]const F,
    out_measurements: [*c]F,
    out_errs_sys: ?[*c]F,
    out_errs_rand: ?[*c]F,
    out_errs_total: ?[*c]F,
    out_node_outputs: ?[*c]F,
    seed_offset: u64,
) void {
    const num_sensors = sensor_in.num_sensors;
    const num_comps = mesh_in.num_components;
    const num_out_times = if (sensor_in.num_sample_times > 0)
        sensor_in.num_sample_times
    else
        mesh_in.num_sim_times;

    const total_elements = num_sensors * num_comps * num_out_times;
    const num_nodes = graph_spec.num_nodes;
    if (num_nodes == 0) {
        @memcpy(out_measurements[0..total_elements], truth_values[0..total_elements]);
        if (out_errs_sys) |p| if (p != null) @memset(p[0..total_elements], 0.0);
        if (out_errs_rand) |p| if (p != null) @memset(p[0..total_elements], 0.0);
        if (out_errs_total) |p| if (p != null) @memset(p[0..total_elements], 0.0);
        return;
    }

    const alloc = std.heap.page_allocator;
    const node_states = alloc.alloc(F, num_nodes * total_elements) catch @panic("OOM");
    defer alloc.free(node_states);

    const node_errors = alloc.alloc(F, num_nodes * total_elements) catch @panic("OOM");
    defer alloc.free(node_errors);

    const in_state_buf = alloc.alloc(F, total_elements) catch @panic("OOM");
    defer alloc.free(in_state_buf);

    const err_sys_buf = alloc.alloc(F, total_elements) catch @panic("OOM");
    defer alloc.free(err_sys_buf);
    @memset(err_sys_buf, 0.0);

    const err_rand_buf = alloc.alloc(F, total_elements) catch @panic("OOM");
    defer alloc.free(err_rand_buf);
    @memset(err_rand_buf, 0.0);

    var rand_streams: [64]sensor_common.RandomStream = undefined;
    for (0..num_nodes) |nn| {
        const spec = graph_spec.nodes_ptr[nn].error_spec;
        if (spec.dist_type != 0) {
            const seed = if (spec.has_seed != 0)
                spec.seed
            else
                123456789 + @as(u64, @intCast(nn));
            rand_streams[nn] = sensor_common.RandomStream.init(seed +% seed_offset);
        }
    }

    for (0..num_nodes) |step_idx| {
        const node_idx = graph_spec.execution_order_ptr[step_idx];
        const node_spec = graph_spec.nodes_ptr[node_idx];
        const spec = node_spec.error_spec;

        // 1. Resolve Input Signal State
        if (node_spec.num_inputs == 0) {
            @memcpy(in_state_buf[0..total_elements], truth_values[0..total_elements]);
        } else if (node_spec.num_inputs == 1) {
            const parent_idx = node_spec.input_indices_ptr[0];
            const p_offset = parent_idx * total_elements;
            @memcpy(
                in_state_buf[0..total_elements],
                node_states[p_offset .. p_offset + total_elements],
            );
        } else {
            @memcpy(in_state_buf[0..total_elements], truth_values[0..total_elements]);
            for (0..node_spec.num_inputs) |inp_i| {
                const parent_idx = node_spec.input_indices_ptr[inp_i];
                const p_offset = parent_idx * total_elements;
                for (0..total_elements) |elem_i| {
                    in_state_buf[elem_i] += (node_states[p_offset + elem_i] -
                        truth_values[elem_i]);
                }
            }
        }

        // 2. Prepare Field Perturbations if any
        if (spec.kind == 12 and spec.field_spec_ptr != null) {
            sensor_common.applyFieldPerturbations(
                mesh_in,
                sensor_in,
                spec.field_spec_ptr,
                node_idx,
                seed_offset,
                spec.err_dep == 1,
            );
        }

        const curr_node_state = node_states[node_idx * total_elements ..][0..total_elements];
        const curr_node_err = node_errors[node_idx * total_elements ..][0..total_elements];

        // 3. Evaluate Error Simulator & Operator
        for (0..num_sensors) |ss| {
            var sensor_rand_val: F = 0.0;
            if (spec.kind == 2 and spec.err_type == 0 and spec.dist_type != 0) {
                sensor_rand_val = sensor_common.sampleDistribution(
                    &rand_streams[node_idx],
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
                        in_state_buf[idx]
                    else
                        truth_values[idx];

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
                                err_val = sensor_common.sampleDistribution(
                                    &rand_streams[node_idx],
                                    spec.dist_type,
                                    spec.param0,
                                    spec.param1,
                                    spec.param2,
                                );
                            }
                        },
                        5 => {
                            if (spec.dist_type != 0) {
                                const r_samp = sensor_common.sampleDistribution(
                                    &rand_streams[node_idx],
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
                            sensor_common.samplePerturbedSensor(
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

                    curr_node_err[idx] = err_val;

                    const op: ErrOp = @enumFromInt(node_spec.op);
                    switch (op) {
                        .add => {
                            curr_node_state[idx] = in_state_buf[idx] + err_val;
                        },
                        .multiply => {
                            curr_node_state[idx] = in_state_buf[idx] * (1.0 + err_val);
                        },
                        .replace => {
                            curr_node_state[idx] = err_val;
                        },
                        .custom_poly => {
                            if (spec.poly_coeffs_ptr != null and spec.poly_coeffs_len > 0) {
                                const coeffs = spec.poly_coeffs_ptr[0..spec.poly_coeffs_len];
                                curr_node_state[idx] = errors.evalPoly(
                                    coeffs,
                                    in_state_buf[idx],
                                );
                            } else {
                                curr_node_state[idx] = in_state_buf[idx] + err_val;
                            }
                        },
                        .custom_table => {
                            if (spec.table_ptr != null and spec.table_rows > 0) {
                                const table_slice = spec.table_ptr[0 .. spec.table_rows * 2];
                                curr_node_state[idx] = errors.evalTableLookup1D(
                                    table_slice,
                                    spec.table_rows,
                                    in_state_buf[idx],
                                );
                            } else {
                                curr_node_state[idx] = in_state_buf[idx] + err_val;
                            }
                        },
                    }

                    if (spec.err_type == 0) {
                        err_sys_buf[idx] += err_val;
                    } else {
                        err_rand_buf[idx] += err_val;
                    }
                }
            }
        }
    }

    // 4. Resolve Final Leaf Output States
    if (graph_spec.num_leaves == 1) {
        const leaf_idx = graph_spec.leaf_indices_ptr[0];
        const l_offset = leaf_idx * total_elements;
        @memcpy(
            out_measurements[0..total_elements],
            node_states[l_offset .. l_offset + total_elements],
        );
    } else if (graph_spec.num_leaves > 1) {
        @memcpy(out_measurements[0..total_elements], truth_values[0..total_elements]);
        for (0..graph_spec.num_leaves) |lf_i| {
            const leaf_idx = graph_spec.leaf_indices_ptr[lf_i];
            const l_offset = leaf_idx * total_elements;
            for (0..total_elements) |elem_i| {
                out_measurements[elem_i] += (node_states[l_offset + elem_i] -
                    truth_values[elem_i]);
            }
        }
    } else {
        @memcpy(out_measurements[0..total_elements], truth_values[0..total_elements]);
    }

    if (out_errs_total) |ptr| {
        if (ptr != null) {
            for (0..total_elements) |elem_i| {
                ptr[elem_i] = out_measurements[elem_i] - truth_values[elem_i];
            }
        }
    }
    if (out_errs_sys) |ptr| {
        if (ptr != null) @memcpy(ptr[0..total_elements], err_sys_buf[0..total_elements]);
    }
    if (out_errs_rand) |ptr| {
        if (ptr != null) @memcpy(ptr[0..total_elements], err_rand_buf[0..total_elements]);
    }
    if (out_node_outputs) |ptr| {
        if (ptr != null and graph_spec.store_node_outputs != 0) {
            @memcpy(ptr[0 .. num_nodes * total_elements], node_states[0 .. num_nodes * total_elements]);
        }
    }
}
