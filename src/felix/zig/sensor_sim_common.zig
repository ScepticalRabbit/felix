// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("buildconfig.zig");
const elements = @import("elements.zig");
const mesh_interp = @import("mesh_interp.zig");
const transforms = @import("transforms.zig");
const errors = @import("errors.zig");
const random = @import("random.zig");
const spatial_grid = @import("spatial_grid.zig");

pub const F = buildconfig.F;
pub const SimdWidth = buildconfig.SimdWidth;
pub const VecSF = buildconfig.VecSF;
pub const ElementType = elements.ElementType;
pub const ErrorSpec = errors.ErrorSpec;
pub const RandomStream = random.RandomStream;
pub const SensorLocation = mesh_interp.SensorLocation;

// --------------------------------------------------------------------------------------
// C-ABI Compatible Simulation Input Structures
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

pub const SensorMeshBinding = extern struct {
    num_sensors: usize,
    found_mask_ptr: [*c]const u8,
    elem_indices_ptr: [*c]const usize,
    node_counts_ptr: [*c]const usize,
    node_indices_ptr: [*c]const usize,
    weights_ptr: [*c]const F,
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
    binding_ptr: ?*const SensorMeshBinding = null,
};

pub fn bindSensorsToMesh(
    outer_alloc: std.mem.Allocator,
    mesh_in: *const SimMeshInput,
    positions_ptr: [*c]const F,
    num_sensors: usize,
    out_binding: *SensorMeshBinding,
) !void {
    const elem_type: ElementType = @enumFromInt(mesh_in.elem_type);

    const found_mask = try outer_alloc.alloc(u8, num_sensors);
    errdefer outer_alloc.free(found_mask);

    const elem_indices = try outer_alloc.alloc(usize, num_sensors);
    errdefer outer_alloc.free(elem_indices);

    const node_counts = try outer_alloc.alloc(usize, num_sensors);
    errdefer outer_alloc.free(node_counts);

    const node_indices = try outer_alloc.alloc(
        usize,
        num_sensors * elements.max_elem_nodes,
    );
    errdefer outer_alloc.free(node_indices);

    const weights = try outer_alloc.alloc(
        F,
        num_sensors * elements.max_elem_nodes,
    );
    errdefer outer_alloc.free(weights);

    @memset(node_indices, 0);
    @memset(weights, 0.0);

    var grid = try spatial_grid.UniformVoxelGrid.init(
        outer_alloc,
        mesh_in.coords_ptr,
        mesh_in.connect_ptr,
        mesh_in.num_elements,
        elem_type,
    );
    defer grid.deinit(outer_alloc);

    for (0..num_sensors) |ss| {
        const px = positions_ptr[ss * 3 + 0];
        const py = positions_ptr[ss * 3 + 1];
        const pz = positions_ptr[ss * 3 + 2];

        var loc: SensorLocation = undefined;
        grid.locatePoint(
            mesh_in.coords_ptr,
            mesh_in.connect_ptr,
            elem_type,
            px,
            py,
            pz,
            &loc,
        );

        found_mask[ss] = if (loc.found) 1 else 0;
        elem_indices[ss] = loc.elem_idx;
        node_counts[ss] = loc.node_count;

        const base_node_offset = ss * elements.max_elem_nodes;
        for (0..loc.node_count) |nn| {
            node_indices[base_node_offset + nn] = loc.node_indices[nn];
            weights[base_node_offset + nn] = loc.weights[nn];
        }
    }

    out_binding.* = SensorMeshBinding{
        .num_sensors = num_sensors,
        .found_mask_ptr = found_mask.ptr,
        .elem_indices_ptr = elem_indices.ptr,
        .node_counts_ptr = node_counts.ptr,
        .node_indices_ptr = node_indices.ptr,
        .weights_ptr = weights.ptr,
    };
}

pub fn freeSensorMeshBinding(
    outer_alloc: std.mem.Allocator,
    binding: *SensorMeshBinding,
) void {
    if (binding.found_mask_ptr != null) {
        outer_alloc.free(binding.found_mask_ptr[0..binding.num_sensors]);
    }
    if (binding.elem_indices_ptr != null) {
        outer_alloc.free(binding.elem_indices_ptr[0..binding.num_sensors]);
    }
    if (binding.node_counts_ptr != null) {
        outer_alloc.free(binding.node_counts_ptr[0..binding.num_sensors]);
    }
    if (binding.node_indices_ptr != null) {
        outer_alloc.free(
            binding.node_indices_ptr[0 .. binding.num_sensors * elements.max_elem_nodes],
        );
    }
    if (binding.weights_ptr != null) {
        outer_alloc.free(
            binding.weights_ptr[0 .. binding.num_sensors * elements.max_elem_nodes],
        );
    }
    binding.* = std.mem.zeroes(SensorMeshBinding);
}

// --------------------------------------------------------------------------------------
// Perturbation & Sampling Helpers
// --------------------------------------------------------------------------------------

pub fn applyFieldPerturbations(
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    field_spec: *const errors.FieldPerturbationSpec,
    error_index: usize,
    seed_offset: u64,
    is_dependent: bool,
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

pub fn samplePerturbedSensor(
    mesh_in: *const SimMeshInput,
    sensor_in: *const SensorArrayInput,
    field_spec: *const errors.FieldPerturbationSpec,
    sensor_index: usize,
    time_index: usize,
    out_values: *[6]F,
) void {
    const num_comps = mesh_in.num_components;
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

pub fn sampleMeshComponent(
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

pub fn initDistributionStream(
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

pub fn calcFieldDrift(
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

pub fn buildPerturbedRotation(
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

pub fn multiplyMatrices(lhs: [*c]const F, rhs: *const [9]F, out: [*c]F) void {
    for (0..3) |rr| {
        for (0..3) |cc| {
            var val: F = 0.0;
            for (0..3) |kk| val += lhs[rr * 3 + kk] * rhs[kk * 3 + cc];
            out[rr * 3 + cc] = val;
        }
    }
}

pub fn setIdentity(out: [*c]F) void {
    const identity = [9]F{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    @memcpy(out[0..9], identity[0..9]);
}

pub fn spatialPointCount(kind: u32) usize {
    return switch (kind) {
        2, 4 => 4,
        3, 5 => 9,
        else => 1,
    };
}

pub fn spatialPoint(
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

pub fn transformSample(
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

pub fn sampleDistribution(
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
