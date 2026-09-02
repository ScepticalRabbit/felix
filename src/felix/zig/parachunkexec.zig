// --------------------------------------------------------------------------------------
// Felix: A virtual sensor laboratory
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

// Parallel Chunk Executioner
// --------------------------------------------------------------------------------------
// Execution helper for "parallel for" work by partitioning the domain range into
// chunks and executing them across a worker thread pool.
// runStaticRange:  Each worker thread executes a statically assigned range chunk.
// runDynRange:     Workers dynamically steal grains using an atomic counter.
// --------------------------------------------------------------------------------------

pub const RangeFn = *const fn (
    ctx_ptr: *anyopaque,
    worker_idx: usize,
    range_start: usize,
    range_end: usize,
) void;

pub const RangeWorkerFnErr = *const fn (
    ctx_ptr: *anyopaque,
    worker_idx: usize,
    range_start: usize,
    range_end: usize,
) anyerror!void;

const WorkerErrState = struct {
    has_err: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    mutex: std.atomic.Mutex = .unlocked,
    first_err: ?anyerror = null,

    fn setFirst(
        self: *WorkerErrState,
        err: anyerror,
    ) void {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        if (self.first_err == null) {
            self.first_err = err;
            self.has_err.store(true, .release);
        }
    }

    fn hasErr(self: *WorkerErrState) bool {
        return self.has_err.load(.acquire);
    }

    fn getFirst(self: *WorkerErrState) ?anyerror {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        return self.first_err;
    }
};

pub const ParaChunkExecutor = struct {
    io: std.Io,
    workers_num: usize,

    pub fn init(
        io: std.Io,
        workers_num: u16,
    ) ParaChunkExecutor {
        return .{
            .io = io,
            .workers_num = @intCast(workers_num),
        };
    }

    pub fn runStaticRange(
        self: *ParaChunkExecutor,
        ctx_ptr: *anyopaque,
        job_func: RangeFn,
        dom_len: usize,
        chunk_size: usize,
    ) !void {
        _ = chunk_size;
        if (dom_len == 0) {
            return;
        }

        std.debug.assert(self.workers_num > 0);

        var group: std.Io.Group = .init;
        errdefer group.cancel(self.io);

        const helper_workers_num = self.workers_num - 1;
        for (0..helper_workers_num) |ww| {
            group.async(
                self.io,
                runStaticWorkerTask,
                .{
                    ctx_ptr,
                    job_func,
                    ww,
                    self.workers_num,
                    dom_len,
                },
            );
        }

        try runStaticWorkerTask(
            ctx_ptr,
            job_func,
            helper_workers_num,
            self.workers_num,
            dom_len,
        );

        try group.await(self.io);
    }

    pub fn runDynRange(
        self: *ParaChunkExecutor,
        ctx_ptr: *anyopaque,
        job_func: RangeFn,
        dom_len: usize,
        grain_size: usize,
    ) !void {
        if (dom_len == 0) {
            return;
        }

        std.debug.assert(grain_size > 0);
        std.debug.assert(self.workers_num > 0);

        var next_start = std.atomic.Value(usize).init(0);
        var group: std.Io.Group = .init;
        errdefer group.cancel(self.io);

        const helper_workers_num = self.workers_num - 1;
        for (0..helper_workers_num) |ww| {
            group.async(
                self.io,
                runDynChunkTask,
                .{
                    ctx_ptr,
                    job_func,
                    ww,
                    &next_start,
                    dom_len,
                    grain_size,
                },
            );
        }

        try runDynChunkTask(
            ctx_ptr,
            job_func,
            helper_workers_num,
            &next_start,
            dom_len,
            grain_size,
        );

        try group.await(self.io);
    }

    pub fn runDynRangeWithWorkerErr(
        self: *ParaChunkExecutor,
        ctx_ptr: *anyopaque,
        job_func: RangeWorkerFnErr,
        dom_len: usize,
        grain_size: usize,
    ) !void {
        if (dom_len == 0) {
            return;
        }

        std.debug.assert(grain_size > 0);
        std.debug.assert(self.workers_num > 0);

        var next_start = std.atomic.Value(usize).init(0);
        var err_state = WorkerErrState{};
        var group: std.Io.Group = .init;
        errdefer group.cancel(self.io);

        const helper_workers_num = self.workers_num - 1;
        for (0..helper_workers_num) |ww| {
            group.async(
                self.io,
                runDynChunkTaskWithWorkerErr,
                .{
                    ctx_ptr,
                    job_func,
                    ww,
                    &err_state,
                    &next_start,
                    dom_len,
                    grain_size,
                },
            );
        }

        try runDynChunkTaskWithWorkerErr(
            ctx_ptr,
            job_func,
            helper_workers_num,
            &err_state,
            &next_start,
            dom_len,
            grain_size,
        );

        try group.await(self.io);
        if (err_state.getFirst()) |err| {
            return err;
        }
    }
};

fn runStaticWorkerTask(
    ctx_ptr: *anyopaque,
    job_func: RangeFn,
    worker_idx: usize,
    workers_num: usize,
    dom_len: usize,
) std.Io.Cancelable!void {
    const range_start = (dom_len * worker_idx) / workers_num;
    const range_end = (dom_len * (worker_idx + 1)) / workers_num;
    if (range_start != range_end) {
        job_func(ctx_ptr, worker_idx, range_start, range_end);
    }
}

fn runDynChunkTask(
    ctx_ptr: *anyopaque,
    job_func: RangeFn,
    worker_idx: usize,
    next_start: *std.atomic.Value(usize),
    dom_len: usize,
    grain_size: usize,
) std.Io.Cancelable!void {
    while (true) {
        const range_start = next_start.fetchAdd(grain_size, .monotonic);
        if (range_start >= dom_len) {
            return;
        }
        const range_end = @min(dom_len, range_start + grain_size);
        job_func(ctx_ptr, worker_idx, range_start, range_end);
    }
}

fn runDynChunkTaskWithWorkerErr(
    ctx_ptr: *anyopaque,
    job_func: RangeWorkerFnErr,
    worker_idx: usize,
    err_state: *WorkerErrState,
    next_start: *std.atomic.Value(usize),
    dom_len: usize,
    grain_size: usize,
) std.Io.Cancelable!void {
    while (true) {
        if (err_state.hasErr()) {
            return;
        }
        const range_start = next_start.fetchAdd(grain_size, .monotonic);
        if (range_start >= dom_len) {
            return;
        }
        const range_end = @min(dom_len, range_start + grain_size);
        job_func(ctx_ptr, worker_idx, range_start, range_end) catch |err| {
            err_state.setFirst(err);
            return;
        };
    }
}

pub fn initThreadedIo(
    outer_alloc: std.mem.Allocator,
    total_threads: u16,
) std.Io.Threaded {
    const threads = @max(@as(u16, 1), total_threads);
    const limit: std.Io.Limit = if (threads <= 1)
        .nothing
    else
        .limited(threads - 1);

    return std.Io.Threaded.init(outer_alloc, .{
        .argv0 = .empty,
        .environ = .empty,
        .async_limit = limit,
        .concurrent_limit = limit,
    });
}
