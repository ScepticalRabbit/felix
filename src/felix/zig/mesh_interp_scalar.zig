// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const common = @import("mesh_interp_common.zig");

const F = common.F;

pub fn interpTimeLinear(
    sim_times: []const F,
    time_series_data: []const F,
    target_time: F,
) F {
    const num_times = sim_times.len;
    if (num_times == 0) return 0.0;
    if (num_times == 1 or target_time <= sim_times[0]) return time_series_data[0];
    if (target_time >= sim_times[num_times - 1]) return time_series_data[num_times - 1];

    for (0..num_times - 1) |tt| {
        const t0 = sim_times[tt];
        const t1 = sim_times[tt + 1];

        if (target_time >= t0 and target_time <= t1) {
            const dt = t1 - t0;
            if (dt < 1e-14) {
                return time_series_data[tt];
            }
            const alpha_val = (target_time - t0) / dt;
            return (1.0 - alpha_val) * time_series_data[tt] +
                alpha_val * time_series_data[tt + 1];
        }
    }

    return time_series_data[num_times - 1];
}

pub fn interpTimesLinear(
    sim_times: []const F,
    sim_data: []const F,
    sample_times: []const F,
    out_vals: []F,
) void {
    const num_sim_times = sim_times.len;
    const num_sample_times = sample_times.len;
    if (num_sim_times == 0 or num_sample_times == 0) return;

    if (num_sim_times == 1) {
        @memset(out_vals, sim_data[0]);
        return;
    }

    var curr_sim_idx: usize = 0;
    const last_sim_idx = num_sim_times - 1;

    for (sample_times, 0..) |targ_time, ii| {
        if (targ_time <= sim_times[0]) {
            out_vals[ii] = sim_data[0];
            continue;
        }
        if (targ_time >= sim_times[last_sim_idx]) {
            out_vals[ii] = sim_data[last_sim_idx];
            continue;
        }

        if (targ_time < sim_times[curr_sim_idx]) {
            curr_sim_idx = 0;
        }

        while (curr_sim_idx < last_sim_idx and targ_time > sim_times[curr_sim_idx + 1]) {
            curr_sim_idx += 1;
        }

        const t0 = sim_times[curr_sim_idx];
        const t1 = sim_times[curr_sim_idx + 1];
        const dt = t1 - t0;
        if (dt < 1e-14) {
            out_vals[ii] = sim_data[curr_sim_idx];
        } else {
            const alpha_val = (targ_time - t0) / dt;
            out_vals[ii] = (1.0 - alpha_val) * sim_data[curr_sim_idx] +
                alpha_val * sim_data[curr_sim_idx + 1];
        }
    }
}

pub fn sampleCachedFEPoint(
    node_count: usize,
    weights: []const F,
    node_indices: []const usize,
    field_data: [*c]const F,
    num_nodes: usize,
    num_components: usize,
    num_times: usize,
    comp_idx: usize,
    time_idx: usize,
) F {
    _ = num_nodes;
    var val: F = 0.0;
    for (0..node_count) |nn| {
        const node_id = node_indices[nn];
        const field_offset = (node_id * num_components + comp_idx) * num_times + time_idx;
        val += weights[nn] * field_data[field_offset];
    }
    return val;
}
