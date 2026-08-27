# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from pyvale.dataio.simdata import SimData

from felix.python.enums import EDim
from felix.python.fieldspecs import FieldScalar, FieldTensor, FieldVector
from felix.python.sensordata import SensorData
from felix.python.sensordescriptor import SensorDescriptor
from felix.python.sensorspoint import SensorsPoint


class SensorFactory:
    @staticmethod
    def scalar_point(
        sim_data: SimData,
        sens_data: SensorData,
        comp_key: str,
        spatial_dims: EDim = EDim.THREED,
        descriptor: SensorDescriptor | None = None,
    ) -> SensorsPoint:
        return SensorsPoint(
            sens_data,
            FieldScalar(sim_data, comp_key, spatial_dims),
            descriptor,
        )

    @staticmethod
    def vector_point(
        sim_data: SimData,
        sens_data: SensorData,
        comp_keys: tuple[str, ...],
        spatial_dims: EDim = EDim.THREED,
        descriptor: SensorDescriptor | None = None,
    ) -> SensorsPoint:
        return SensorsPoint(
            sens_data,
            FieldVector(sim_data, comp_keys, spatial_dims),
            descriptor,
        )

    @staticmethod
    def tensor_point(
        sim_data: SimData,
        sens_data: SensorData,
        norm_comp_keys: tuple[str, ...],
        dev_comp_keys: tuple[str, ...],
        spatial_dims: EDim = EDim.THREED,
        descriptor: SensorDescriptor | None = None,
    ) -> SensorsPoint:
        return SensorsPoint(
            sens_data,
            FieldTensor(
                sim_data,
                norm_comp_keys,
                dev_comp_keys,
                spatial_dims,
            ),
            descriptor,
        )


class DescriptorFactory:
    @staticmethod
    def temperature(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor("Temp.", r"^{\circ}C", symbol="T", tag="TC")

    @staticmethod
    def scalar(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor("scalar", "units", symbol="scal.")

    @staticmethod
    def displacement(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor(
            "Disp.",
            "mm",
            symbol="u",
            tag="DS",
            components=("x", "y", "z"),
        )

    @staticmethod
    def vector(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor(
            "vector",
            "unit",
            symbol="vect.",
            tag="V",
            components=("x", "y", "z"),
        )

    @staticmethod
    def strain(spatial_dims: object = EDim.THREED) -> SensorDescriptor:
        return _tensor_descriptor("Strain", r"\varepsilon", "-", spatial_dims)

    @staticmethod
    def tensor(spatial_dims: object = EDim.THREED) -> SensorDescriptor:
        return _tensor_descriptor("tensor", "tens.", "unit", spatial_dims)


def _tensor_descriptor(
    name: str,
    symbol: str,
    units: str,
    spatial_dims: object,
) -> SensorDescriptor:
    is_2d = getattr(spatial_dims, "value", spatial_dims) == 2
    components = (
        ("xx", "yy", "xy")
        if is_2d
        else ("xx", "yy", "zz", "xy", "yz", "xz")
    )
    return SensorDescriptor(
        name,
        units,
        symbol=symbol,
        tag="SG" if name == "Strain" else "T",
        components=components,
    )
