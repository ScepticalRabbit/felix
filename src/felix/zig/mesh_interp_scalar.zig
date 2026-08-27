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
