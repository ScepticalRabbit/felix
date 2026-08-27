# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from dataclasses import dataclass


@dataclass(slots=True)
class SensorDescriptor:
    name: str = "Measured Value"
    units: str = r"-"
    time_units: str = r"s"
    symbol: str = r"m"
    tag: str = "S"
    components: tuple[str, ...] | None = None

    def create_label(self, comp_ind: int | None = None) -> str:
        return self._create_label(comp_ind)

    def create_label_flat(self, comp_ind: int | None = None) -> str:
        return self._create_label(comp_ind)

    def create_sensor_tags(self, num_sensors: int) -> list[str]:
        width = len(str(num_sensors))
        return [
            f"{self.tag}{sensor + 1:0{width}d}"
            for sensor in range(num_sensors)
        ]

    def _create_label(self, comp_ind: int | None) -> str:
        label = f"{self.name} " if self.name else ""
        symbol = self.symbol
        if comp_ind is not None and self.components is not None:
            symbol = f"{symbol}_{{{self.components[comp_ind]}}}"
        if symbol:
            label += rf"${symbol}$ "
        if self.units:
            label += rf" [${self.units}$]"
        return label
