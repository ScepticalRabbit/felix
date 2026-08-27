# Mesh Convention (Riley)

This document defines the mesh convention used by Riley's rasteriser and shape functions. Felix adopts this convention verbatim.
Felix re‑uses Riley's mesh‑conversion utilities (`riley.python.meshconv`) for validation and enforcement.


`riley.python.meshconv` is the authoritative implementation and validator.

## Mesh representation

- Coordinates are an `N x 3` array named `coords`.
- A connectivity table is a row-major `E x P` integer array: one element per
  row and one local node slot per column.
- Connectivity is zero-based. Every index must satisfy `0 <= index < N`.
- A mesh must use either surface or volume connectivity, not both. Mixed
  element meshes are not supported by the convention converter.

## Element types

- Felix supports these element families:

| Family | Types | Corner slots | Remaining local slots |
| --- | --- | --- | --- |
| Triangle | TRI3, TRI6, TRI7 | `0..2` | TRI6 edges `3..5`; TRI7 edges `3..5`, centre `6` |
| Quadrilateral | QUAD4, QUAD8, QUAD9 | `0..3` | QUAD8 edges `4..7`; QUAD9 edges `4..7`, centre `8` |
| Tetrahedron | TET4, TET10 | `0..3` | TET10 edges `4..9` |
| Hexahedron | HEX8, HEX20, HEX27 | `0..7` | HEX20 edges `8..19`; HEX27 edges `8..19`, faces `20..25`, cell centre `26` |

Other node counts and topologies are rejected. Degenerate elements are input
errors and must not be repaired by reordering.

## Orientation

### Surface elements

Surface connectivity must be consistently material-facing:

- A closed exterior shell has outward pointing normals.
- A cavity boundary has normals pointing into the void. This is correct for a
  plate-with-hole bore wall and must not be reversed merely because it points
  towards the model centre.
- Adjacent faces must traverse a shared edge in opposite directions.
- The core tracks topology by global node identity, not coordinate equality.
Coincident-coordinate nodes at UV seams and poles are valid if they are
distinct node IDs. Non-manifold face sets do not have a unique shell
orientation.

### Volume elements

TET and HEX connectivity must have a positive right-handed signed metric.
This local ordering is significant: it determines the shape-function
coordinate system, interpolation of nodal fields, and extracted surface faces.

## High-order node roles

Higher-order nodes are not interchangeable. Their slots must match the
reference element's edge, face, and centre roles. Moving a mid-edge node to a
different edge slot changes interpolation even though the element contains the
same global node IDs.

For a supplied known source convention, use `MeshConvention` to map each source slot to the target slot according to the Riley mesh convention. Keep exporter‑specific mappings at the import boundary (for example, the PyVale Exodus adapter).

## HEX27 and VTK

- The convention adopts VTK HEX27 local roles (as defined by Riley).

| Slots | Role |
| --- | --- |
| `0..7` | corners |
| `8..19` | edge nodes |
| `20` | front face centre, corners `(0, 1, 5, 4)` |
| `21` | right face centre, corners `(1, 2, 6, 5)` |
| `22` | back face centre, corners `(2, 3, 7, 6)` |
| `23` | left face centre, corners `(3, 0, 4, 7)` |
| `24` | bottom face centre, corners `(0, 1, 2, 3)` |
| `25` | top face centre, corners `(4, 5, 6, 7)` |
| `26` | cell centre |

This is not compatible with every Exodus style HEX27 ordering. Such data needs an explicit source-to-Riley permutation before use when importing Riley mesh data.

## Validation and enforcement

Use the Python API before emitting or consuming mesh CSV data:

```python
from riley.python import meshconv

report = meshconv.check_mesh_convention(mesh)
if report:
    raise ValueError(report)

mesh = meshconv.enforce_mesh_convention(mesh)
```

For a known non-Riley source order, declare it explicitly:

```python
source = meshconv.MeshConvention({
    meshconv.EElementType.HEX20: (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        16, 17, 18, 19, 12, 13, 14, 15,
    ),
})
mesh = meshconv.enforce_mesh_convention(mesh, source)
```

`infer_mesh_convention(mesh)` is an opt in diagnostic. It can infer some
simple affine layouts and rejects ambiguous layouts. It is not yet the default
source order conversion path for `check_mesh_convention` or
`enforce_mesh_convention`; callers with a known source convention should pass
it explicitly.

## Data-generation rule

Prefer this order of work:

1. Enforce/emit Riley-conforming connectivity.
2. Generate UVs, displacement fields, and other nodal data against that mesh.
3. Render and compare against the relevant regression baseline.

If existing connectivity is locally reordered, global node fields do not need
to be reordered. They do need to be regenerated or visually checked when their
meaning depends on the element's local reference coordinates. Regenerate gold
only after render parity has been reviewed; save scaled TIFF alongside FIMG for
new gold generation.
