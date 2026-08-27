# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from felix.sensorsim.enums import EDim
from felix.sensorsim.sensordescriptor import SensorDescriptor


class DescriptorFactory:
    @staticmethod
    def temperature(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor(
            name="Temp.",
            symbol="T",
            units=r"^{\circ}C",
            tag="TC",
        )

    @staticmethod
    def scalar(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor(
            name="scalar",
            symbol="scal.",
            units=r"units",
            tag="S",
        )

    @staticmethod
    def displacement(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor(
            name="Disp.",
            symbol="u",
            units=r"mm",
            tag="DS",
            components=("x", "y", "z"),
        )

    @staticmethod
    def vector(spatial_dims: object = None) -> SensorDescriptor:
        return SensorDescriptor(
            name="vector",
            symbol="vect.",
            units=r"unit",
            tag="V",
            components=("x", "y", "z"),
        )

    @staticmethod
    def strain(spatial_dims: object = EDim.THREED) -> SensorDescriptor:
        is_2d = (
            spatial_dims == EDim.TWOD
            or getattr(spatial_dims, "value", None) == 2
            or spatial_dims == 2
        )
        comps = ("xx", "yy", "xy") if is_2d else (
            "xx",
            "yy",
            "zz",
            "xy",
            "yz",
            "xz",
        )
        return SensorDescriptor(
            name="Strain",
            symbol=r"\varepsilon",
            units=r"-",
            tag="SG",
            components=comps,
        )

    @staticmethod
    def tensor(spatial_dims: object = EDim.THREED) -> SensorDescriptor:
        is_2d = (
            spatial_dims == EDim.TWOD
            or getattr(spatial_dims, "value", None) == 2
            or spatial_dims == 2
        )
        comps = ("xx", "yy", "xy") if is_2d else (
            "xx",
            "yy",
            "zz",
            "xy",
            "yz",
            "xz",
        )
        return SensorDescriptor(
            name="tensor",
            symbol=r"tens.",
            units=r"unit",
            tag="T",
            components=comps,
        )
