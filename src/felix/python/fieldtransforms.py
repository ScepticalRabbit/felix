# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Field transformation operators and transformed field adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
from pyvale.dataio.simdata import SimData
from scipy.spatial.transform import Rotation

import felix.cython.felix as fc
from felix.python.enums import EDim
from felix.python.fieldspecs import IField


class IFieldTransform(ABC):
    """Abstract interface for field transformation operators."""

    @abstractmethod
    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Returns output component names given input components."""

    @abstractmethod
    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        """Transforms raw interpolated field values into derived quantities."""


class FieldTransformVonMises(IFieldTransform):
    """Computes scalar von Mises stress invariant from stress tensors."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "von_mises") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        s_dim = 2 if raw_samples.shape[1] == 3 else 3
        return fc.transform_tensor_invariants(raw_samples, 0, s_dim)


class FieldTransformPrincipal(IFieldTransform):
    """Computes ordered principal eigenvalues and maximum shear."""

    __slots__ = ("_return_max_shear",)

    def __init__(self, return_max_shear: bool = True) -> None:
        self._return_max_shear = return_max_shear

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        n_in = len(input_components)
        prefix = (
            "eps"
            if len(input_components) > 0
            and "strain" in input_components[0].lower()
            else "sigma"
        )
        shear_key = "gamma_max" if prefix == "eps" else "tau_max"

        if n_in == 3:
            names = [f"{prefix}_1", f"{prefix}_2"]
            if self._return_max_shear:
                names.append(shear_key)
            return tuple(names)
        names = [f"{prefix}_1", f"{prefix}_2", f"{prefix}_3"]
        if self._return_max_shear:
            names.append(shear_key)
        return tuple(names)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_comps = raw_samples.shape[1]
        s_dim = 2 if n_comps == 3 else 3

        p1 = fc.transform_tensor_invariants(raw_samples, 1, s_dim)
        p2 = fc.transform_tensor_invariants(raw_samples, 2, s_dim)

        if s_dim == 2:
            out_list = [p1, p2]
            if self._return_max_shear:
                max_s = fc.transform_tensor_invariants(raw_samples, 11, s_dim)
                out_list.append(max_s)
            return np.concatenate(out_list, axis=1)

        p3 = fc.transform_tensor_invariants(raw_samples, 3, s_dim)
        out_list = [p1, p2, p3]
        if self._return_max_shear:
            tresca = fc.transform_tensor_invariants(raw_samples, 4, s_dim)
            out_list.append(tresca)
        return np.concatenate(out_list, axis=1)


class FieldTransformHydrostatic(IFieldTransform):
    """Computes hydrostatic mean stress / pressure p = (1/d)*tr(sigma)."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "hydrostatic") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        s_dim = 2 if raw_samples.shape[1] == 3 else 3
        return fc.transform_tensor_invariants(raw_samples, 5, s_dim)


class FieldTransformTresca(IFieldTransform):
    """Computes Tresca maximum shear stress tau_max."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "tresca") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        s_dim = 2 if raw_samples.shape[1] == 3 else 3
        return fc.transform_tensor_invariants(raw_samples, 4, s_dim)


class FieldTransformTraction(IFieldTransform):
    """Computes surface traction vector t = sigma . n."""

    __slots__ = ("_include_scalar_projections",)

    def __init__(self, include_scalar_projections: bool = True) -> None:
        self._include_scalar_projections = include_scalar_projections

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        names = ["traction_x", "traction_y", "traction_z"]
        if self._include_scalar_projections:
            names.extend(["traction_normal", "traction_shear"])
        return tuple(names)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_pts, n_comps, n_times = raw_samples.shape
        normals = np.zeros((n_pts, 3), dtype=np.float64)
        if angles is not None:
            if len(angles) == 1:
                normals[:] = angles[0].apply(np.array([0.0, 0.0, 1.0]))
            else:
                for idx, rot in enumerate(angles):
                    normals[idx, :] = rot.apply(np.array([0.0, 0.0, 1.0]))
        else:
            normals[:, 2] = 1.0

        if n_comps == 3:
            s_xx = raw_samples[:, 0, :]
            s_yy = raw_samples[:, 1, :]
            s_xy = raw_samples[:, 2, :]
            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]

            tx = s_xx * nx + s_xy * ny
            ty = s_xy * nx + s_yy * ny
            tz = np.zeros_like(tx)
        elif n_comps >= 6:
            s_xx = raw_samples[:, 0, :]
            s_yy = raw_samples[:, 1, :]
            s_zz = raw_samples[:, 2, :]
            s_xy = raw_samples[:, 3, :]
            s_xz = raw_samples[:, 4, :]
            s_yz = raw_samples[:, 5, :]

            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]
            nz = normals[:, 2, np.newaxis]

            tx = s_xx * nx + s_xy * ny + s_xz * nz
            ty = s_xy * nx + s_yy * ny + s_yz * nz
            tz = s_xz * nx + s_yz * ny + szz * nz
        else:
            raise ValueError(
                f"Traction transform expects 3 or 6 components, got {n_comps}."
            )

        tx = tx[:, np.newaxis, :]
        ty = ty[:, np.newaxis, :]
        tz = tz[:, np.newaxis, :]
        out_list = [tx, ty, tz]

        if self._include_scalar_projections:
            nx = normals[:, 0, np.newaxis, np.newaxis]
            ny = normals[:, 1, np.newaxis, np.newaxis]
            nz = normals[:, 2, np.newaxis, np.newaxis]

            t_n = tx * nx + ty * ny + tz * nz
            t_sq = tx**2 + ty**2 + tz**2
            t_s = np.sqrt(np.maximum(0.0, t_sq - t_n**2))
            out_list.extend([t_n, t_s])

        return np.concatenate(out_list, axis=1)


class FieldTransformFlux(IFieldTransform):
    """Computes normal scalar flux q_n = q . n from a vector field."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "flux_normal") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_pts, n_comps, n_times = raw_samples.shape
        normals = np.zeros((n_pts, 3), dtype=np.float64)
        if angles is not None:
            if len(angles) == 1:
                normals[:] = angles[0].apply(np.array([0.0, 0.0, 1.0]))
            else:
                for idx, rot in enumerate(angles):
                    normals[idx, :] = rot.apply(np.array([0.0, 0.0, 1.0]))
        else:
            normals[:, 2] = 1.0

        if n_comps == 2:
            qx = raw_samples[:, 0, :]
            qy = raw_samples[:, 1, :]
            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]
            qn = qx * nx + qy * ny
        elif n_comps >= 3:
            qx = raw_samples[:, 0, :]
            qy = raw_samples[:, 1, :]
            qz = raw_samples[:, 2, :]
            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]
            nz = normals[:, 2, np.newaxis]
            qn = qx * nx + qy * ny + qz * nz
        else:
            raise ValueError(
                f"Flux transform expects 2 or 3 components, got {n_comps}."
            )

        return qn[:, np.newaxis, :]


class FieldTransformMagnitude(IFieldTransform):
    """Computes Euclidean magnitude ||u|| of a vector field."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "magnitude") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        mag = np.sqrt(np.sum(raw_samples**2, axis=1, keepdims=True))
        return mag


class FieldTransformDirectional(IFieldTransform):
    """Projects a vector field onto a specified direction vector u . d."""

    __slots__ = ("_direction", "_component_name")

    def __init__(
        self,
        direction: tuple[float, float, float] | np.ndarray,
        component_name: str = "directional_projection",
    ) -> None:
        dir_arr = np.asarray(direction, dtype=np.float64)
        self._direction = dir_arr / np.linalg.norm(dir_arr)
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_comps = raw_samples.shape[1]
        d = self._direction[:n_comps]
        proj = np.tensordot(d, raw_samples, axes=(0, 1))
        return proj[:, np.newaxis, :]


class FieldTransformChain(IFieldTransform):
    """Chains multiple IFieldTransform operators sequentially."""

    __slots__ = ("_transforms",)

    def __init__(self, transforms: list[IFieldTransform]) -> None:
        self._transforms = transforms

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        current_names = input_components
        for tr in self._transforms:
            current_names = tr.get_component_names(current_names)
        return current_names

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        current_data = raw_samples
        for tr in self._transforms:
            current_data = tr.transform(current_data, points, times, angles)
        return current_data


class FieldTransformCustom(IFieldTransform):
    """Applies a custom user callable to transform raw sampled field arrays."""

    __slots__ = ("_func", "_component_names")

    def __init__(
        self,
        func: Callable[
            [
                np.ndarray,
                np.ndarray,
                np.ndarray,
                tuple[Rotation, ...] | None,
            ],
            np.ndarray,
        ]
        | Callable[[np.ndarray], np.ndarray],
        component_names: tuple[str, ...] = ("custom",),
    ) -> None:
        self._func = func
        self._component_names = component_names

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return self._component_names

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        try:
            return self._func(raw_samples, points, times, angles)
        except TypeError:
            return self._func(raw_samples)


class FieldTransformed(IField):
    """Wraps any physical field and dynamically applies a field transform."""

    __slots__ = ("_underlying_field", "_transform")

    def __init__(
        self,
        base_field: IField | None = None,
        transform: IFieldTransform | None = None,
        field: IField | None = None,
    ) -> None:
        b_field = base_field if base_field is not None else field
        if b_field is None or transform is None:
            raise ValueError("Both base field and transform must be provided")
        self._underlying_field = b_field
        self._transform = transform

    def get_base_field(self) -> IField:
        return self._underlying_field

    def get_transform(self) -> IFieldTransform:
        return self._transform

    def set_sim_data(self, sim_data: SimData) -> None:
        self._underlying_field.set_sim_data(sim_data)

    def get_sim_data(self) -> SimData:
        return self._underlying_field.get_sim_data()

    def get_spatial_dims(self) -> EDim:
        return self._underlying_field.get_spatial_dims()

    def get_time_steps(self) -> np.ndarray:
        return self._underlying_field.get_time_steps()

    def get_visualiser(self) -> object:
        return self._underlying_field.get_visualiser()

    def get_all_components(self) -> tuple[str, ...]:
        in_comps = self._underlying_field.get_all_components()
        return self._transform.get_component_names(in_comps)

    def get_component_index(self, comp_key: str) -> int:
        comps = self.get_all_components()
        return comps.index(comp_key)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        raw_samples = self._underlying_field.sample_field(
            points=points, times=times, angles=angles
        )
        sample_times = (
            times
            if times is not None
            else self._underlying_field.get_time_steps()
        )
        return self._transform.transform(
            raw_samples=raw_samples,
            points=points,
            times=sample_times,
            angles=angles,
        )


class FieldMultiTransformed(IField):
    """Fuses multiple IField instances through a multi-field transform."""

    __slots__ = ("_fields", "_transform_func", "_component_names")

    def __init__(
        self,
        fields: dict[str, IField],
        transform_func: Callable[
            [dict[str, np.ndarray], np.ndarray, np.ndarray], np.ndarray
        ],
        component_names: tuple[str, ...],
    ) -> None:
        self._fields = fields
        self._transform_func = transform_func
        self._component_names = component_names

    def set_sim_data(self, sim_data: SimData) -> None:
        for f in self._fields.values():
            f.set_sim_data(sim_data)

    def get_sim_data(self) -> SimData:
        first_field = next(iter(self._fields.values()))
        return first_field.get_sim_data()

    def get_spatial_dims(self) -> EDim:
        first_field = next(iter(self._fields.values()))
        return first_field.get_spatial_dims()

    def get_time_steps(self) -> np.ndarray:
        first_field = next(iter(self._fields.values()))
        return first_field.get_time_steps()

    def get_visualiser(self) -> object:
        first_field = next(iter(self._fields.values()))
        return first_field.get_visualiser()

    def get_all_components(self) -> tuple[str, ...]:
        return self._component_names

    def get_component_index(self, comp_key: str) -> int:
        return self._component_names.index(comp_key)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        samples = {}
        for k, f in self._fields.items():
            samples[k] = f.sample_field(points, times, angles)

        first_field = next(iter(self._fields.values()))
        sample_times = (
            times if times is not None else first_field.get_time_steps()
        )
        return self._transform_func(samples, points, sample_times)
