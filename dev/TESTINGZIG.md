# Zig Native Testing Guide

Felix provides native Zig testing orchestration modelled after the Riley test framework. All native tests run in debug mode to verify numerical correctness, SIMD/scalar parity, and catch memory safety issues or leaks with `std.heap.DebugAllocator`.

---

## Running Native Zig Tests

Execute the complete native test suite from the repository root:

```bash
zig test ./src/test_felix.zig
```

Or via the Zig build system:

```bash
zig build test
```

To run a single focused test suite:

```bash
zig test ./src/tests/test_mesh_elements.zig
zig test ./src/tests/test_field_transforms.zig
zig test ./src/tests/test_quadrature_kernels.zig
zig test ./src/tests/test_stats.zig
zig test ./src/tests/test_error_chains.zig
zig test ./src/tests/test_err_graph.zig
zig test ./src/tests/test_point_sensors.zig
zig test ./src/tests/test_monoblock_e2e.zig
```

---

## Memory Allocator Architecture & Pattern

Felix follows strict allocator discipline throughout the codebase:

1. **Outer Default Allocator**:
   At the outermost entry points (C-ABI boundaries and top-level runners), use `std.heap.smp_allocator` as the root allocator.
2. **Arena Wrapping**:
   Immediately wrap the outer allocator in an `ArenaAllocator`:
   ```zig
   var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
   defer arena.deinit();
   const alloc = arena.allocator();
   ```
3. **Function Parameter Ordering**:
   Pass allocators as the first non-comptime arguments:
   `pub fn myFunction(outer_alloc: std.mem.Allocator, io: std.Io, ...) !ReturnType`
4. **Function-Scoped Local Arena**:
   If a library function needs scratch or temporary allocations, it must create a function-local arena wrapping `outer_alloc` and defer deinitialization:
   ```zig
   var arena = std.heap.ArenaAllocator.init(outer_alloc);
   defer arena.deinit();
   const local_alloc = arena.allocator();
   ```
   All temporary buffers, string formatting, and intermediate slices allocate from `local_alloc`. Returned/owned memory allocates directly from `outer_alloc`.
5. **Debug Leak Verification in Tests**:
   Tests must initialize `std.heap.DebugAllocator` and assert clean teardown:
   ```zig
   var gpa: std.heap.DebugAllocator(.{}) = .init;
   const allocator = gpa.allocator();
   defer {
       const deinit_status = gpa.deinit();
       std.testing.expect(deinit_status == .ok) catch @panic("Memory leak detected!");
   }
   ```

---

## Test Orchestration & Dev Support (`src/dev_support/`)

Common testing helpers and case orchestration are located in `src/dev_support/`:

- `testconfig.zig`: Global test precision (`f64` / `f32`), tolerances (`REL_TOL`, `ABS_TOL`), and execution options.
- `testpolicy.zig`: `DatasetCase` enum and path routing to simulation datasets in `data/`.
- `tests.zig`: Numerical comparison utilities:
  - `isApproxEqual(a, b, rel_tol, abs_tol)`
  - `assertSlicesClose(actual, expected, rel_tol, abs_tol)`
  - `compareNDArrayToGold(allocator, io, actual, gold_path)`
- `orchestration.zig`: High-level case loaders:
  - `loadCaseSimData`: Loads coordinate, connectivity, and field CSVs into `SimData`.
  - `buildSimMeshInput`: Constructs `SimMeshInput` descriptor for sensor simulations.

---

## Simulation Datasets (`data/`)

Simulation test cases are converted from Exodus `.e` files into CSV mesh representations using `data/exodus_to_csv.py`:

- `data/cube_hex8`: 8-node linear hex mesh.
- `data/cube_hex20`: 20-node quadratic hex mesh.
- `data/tet4`: 4-node linear tet mesh.
- `data/tet10`: 10-node quadratic tet mesh.
- `data/plate_2d_mech`: 2D quadratic plate with hole (QUAD9).
- `data/plate_2d_tm`: 2D thermomechanical plate (QUAD4).
- `data/monoblock_3d`: Full 3D fusion divertor monoblock (HEX20, 5,152 nodes, 13 physical fields).
- `data/min/`: 2-element benchmark cases (TRI3, TRI6, QUAD4, QUAD8, QUAD9).

To regenerate or extract datasets:

```bash
.venv/bin/python data/exodus_to_csv.py
```

---

## Required Checks

Before completing any Zig change:

1. Run the native test suite:
   ```bash
   zig test ./src/test_felix.zig
   ```
2. Run the build system test step:
   ```bash
   zig build test
   ```
3. Format Zig code (< 95 columns line limit, 4 spaces indentation):
   ```bash
   zig fmt src/
   ```
4. Verify all tests pass with zero memory leaks.
