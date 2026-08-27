# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from pyvale.dataio.simdata import SimData
from felix.sensorsim.enums import EDim
from felix.sensorsim.sensordata import SensorData
from felix.sensorsim.sensordescriptor import SensorDescriptor
from felix.sensorsim.fieldscalar import FieldScalar
from felix.sensorsim.fieldvector import FieldVector
from felix.sensorsim.fieldtensor import FieldTensor
from felix.sensorsim.sensorspoint import SensorsPoint


class SensorFactory:
    @staticmethod
    def scalar_point(
        sim_data: SimData,
        sens_data: SensorData,
        comp_key: str,
        spatial_dims: EDim = EDim.THREED,
        descriptor: SensorDescriptor | None = None,
    ) -> SensorsPoint:
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        return SensorsPoint(sens_data, field, descriptor)

    @staticmethod
    def vector_point(
        sim_data: SimData,
        sens_data: SensorData,
        comp_keys: tuple[str, ...],
        spatial_dims: EDim = EDim.THREED,
        descriptor: SensorDescriptor | None = None,
    ) -> SensorsPoint:
        field = FieldVector(sim_data, comp_keys, spatial_dims)
        return SensorsPoint(sens_data, field, descriptor)

    @staticmethod
    def tensor_point(
        sim_data: SimData,
        sens_data: SensorData,
        norm_comp_keys: tuple[str, ...],
        dev_comp_keys: tuple[str, ...],
        spatial_dims: EDim = EDim.THREED,
        descriptor: SensorDescriptor | None = None,
    ) -> SensorsPoint:
        field = FieldTensor(
            sim_data, norm_comp_keys, dev_comp_keys, spatial_dims
        )
        return SensorsPoint(sens_data, field, descriptor)
