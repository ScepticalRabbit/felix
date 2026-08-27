// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const F: type = f64;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const RandomStream = struct {
    prng: std.Random.Xoshiro256,
    random: std.Random,
    has_spare_normal: bool = false,
    spare_normal_val: F = 0.0,

    const Self = @This();

    pub fn init(seed_val: u64) Self {
        var stream = Self{
            .prng = std.Random.Xoshiro256.init(seed_val),
            .random = undefined,
            .has_spare_normal = false,
            .spare_normal_val = 0.0,
        };
        stream.random = stream.prng.random();
        return stream;
    }

    pub fn reseed(self: *Self, seed_val: u64) void {
        self.prng = std.Random.Xoshiro256.init(seed_val);
        self.random = self.prng.random();
        self.has_spare_normal = false;
        self.spare_normal_val = 0.0;
    }

    pub fn uniform01(self: *Self) F {
        return self.random.float(F);
    }

    pub fn uniform(self: *Self, low_val: F, high_val: F) F {
        const rand_u01 = self.uniform01();
        return low_val + (high_val - low_val) * rand_u01;
    }

    pub fn normal(self: *Self, mean_val: F, std_dev: F) F {
        if (self.has_spare_normal) {
            self.has_spare_normal = false;
            return mean_val + std_dev * self.spare_normal_val;
        }

        var rand_u1 = self.uniform01();
        while (rand_u1 <= 1e-15) {
            rand_u1 = self.uniform01();
        }
        const rand_u2 = self.uniform01();

        const radius = @sqrt(-2.0 * @log(rand_u1));
        const theta = 2.0 * std.math.pi * rand_u2;

        const z0 = radius * @cos(theta);
        const z1 = radius * @sin(theta);

        self.spare_normal_val = z1;
        self.has_spare_normal = true;

        return mean_val + std_dev * z0;
    }

    pub fn exponential(self: *Self, rate_lambda: F) F {
        var u_val = self.uniform01();
        while (u_val <= 1e-15) {
            u_val = self.uniform01();
        }
        return -@log(u_val) / rate_lambda;
    }

    pub fn triangular(
        self: *Self,
        left_val: F,
        mode_val: F,
        right_val: F,
    ) F {
        const u_val = self.uniform01();
        const split_ratio = (mode_val - left_val) / (right_val - left_val);

        if (u_val < split_ratio) {
            return left_val + @sqrt(u_val * (right_val - left_val) * (mode_val - left_val));
        } else {
            return right_val - @sqrt(
                (1.0 - u_val) * (right_val - left_val) * (right_val - mode_val),
            );
        }
    }

    pub fn gamma(self: *Self, shape_k: F, scale_theta: F) F {
        if (shape_k < 1.0) {
            const u_val = self.uniform01();
            const g_val = self.gamma(shape_k + 1.0, scale_theta);
            return g_val * std.math.pow(F, u_val, 1.0 / shape_k);
        }

        const dd = shape_k - 1.0 / 3.0;
        const cc = 1.0 / @sqrt(9.0 * dd);

        while (true) {
            var z_val: F = undefined;
            var vv: F = undefined;

            while (true) {
                z_val = self.normal(0.0, 1.0);
                vv = 1.0 + cc * z_val;
                if (vv > 0.0) {
                    break;
                }
            }

            vv = vv * vv * vv;
            const uu = self.uniform01();

            if (uu < 1.0 - 0.0331 * z_val * z_val * z_val * z_val) {
                return dd * vv * scale_theta;
            }

            if (@log(uu) < 0.5 * z_val * z_val + dd * (1.0 - vv + @log(vv))) {
                return dd * vv * scale_theta;
            }
        }
    }

    pub fn beta(self: *Self, alpha_val: F, beta_val: F) F {
        const x_gamma = self.gamma(alpha_val, 1.0);
        const y_gamma = self.gamma(beta_val, 1.0);
        if (x_gamma + y_gamma == 0.0) {
            return 0.5;
        }
        return x_gamma / (x_gamma + y_gamma);
    }

    pub fn logNormal(self: *Self, mean_mu: F, sigma_val: F) F {
        const norm_val = self.normal(mean_mu, sigma_val);
        return @exp(norm_val);
    }

    pub fn fillUniform(
        self: *Self,
        low_val: F,
        high_val: F,
        out_slice: []F,
    ) void {
        for (out_slice) |*elem_ptr| {
            elem_ptr.* = self.uniform(low_val, high_val);
        }
    }

    pub fn fillNormal(
        self: *Self,
        mean_val: F,
        std_dev: F,
        out_slice: []F,
    ) void {
        for (out_slice) |*elem_ptr| {
            elem_ptr.* = self.normal(mean_val, std_dev);
        }
    }
};

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "RandomStream basic sampling" {
    var stream = RandomStream.init(12345);

    const u_val = stream.uniform(0.0, 10.0);
    try std.testing.expect(u_val >= 0.0 and u_val <= 10.0);

    const n_val = stream.normal(5.0, 1.0);
    try std.testing.expect(n_val > -10.0 and n_val < 20.0);

    const g_val = stream.gamma(2.0, 1.0);
    try std.testing.expect(g_val > 0.0);

    const b_val = stream.beta(2.0, 5.0);
    try std.testing.expect(b_val >= 0.0 and b_val <= 1.0);
}
