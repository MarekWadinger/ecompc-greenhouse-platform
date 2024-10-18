from abc import ABC

import casadi as ca
from numpy import clip


class Actuator(ABC):
    """Actuator model interface

    Interface for actuator models

    Args:
        max_unit: Maximum reachable actuation [unit]
        power_per_unit: Power per Actuation Unit [W/unit]
        energy_cost: Cost of Energy [EUR/kWh]
        co2_intensity: Carbon Intensity [gCO₂eq/kWh]  https://app.electricitymaps.com/
        co2_cost: Social cost of CO2 [EUR/gCO₂eq]  https://www.theguardian.com/environment/article/2024/may/17/economic-damage-climate-change-report
        dt: Duration [s]
    """

    def __init__(
        self,
        max_unit: float,
        power_per_unit: float = 1.0,
        energy_cost: float = 0.0612,
        co2_intensity: float = 200.0,
        co2_cost: float = 0.001,
        efficiency: float = 0.8,
        dt: float = 1.0,
    ):
        self.max_unit = max_unit  # Maximum value of actuation
        self.power_per_unit = power_per_unit
        self.energy_cost = energy_cost
        self.co2_intensity = co2_intensity
        self.co2_cost = co2_cost
        self.efficiency = efficiency
        self.dt = dt

    def signal_to_actuation(
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.GenericExpressionCommon:
        if isinstance(signal, ca.GenericExpressionCommon):
            signal = ca.if_else(signal < 0, 0, signal)
            signal = ca.if_else(signal > 100, 100, signal)
        else:
            signal = clip(signal, 0.0, 100.0)

        # Calculate heating power based on the signal
        return (signal / 100.0) * self.max_unit

    def signal_to_power(
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.GenericExpressionCommon:
        return (
            self.power_per_unit
            * self.signal_to_actuation(signal)
            / self.efficiency
        )

    def signal_to_eur(
        self,
        signal: float | ca.GenericExpressionCommon,
        energy_cost: float | None = None,
    ) -> float | ca.GenericExpressionCommon:
        if energy_cost is None:
            energy_cost = self.energy_cost
        return (
            energy_cost * self.signal_to_power(signal) / 1000 * self.dt / 3600
        )

    def signal_to_co2(
        self,
        signal: float | ca.GenericExpressionCommon,
        co2_intensity: float | None = None,
    ) -> float | ca.GenericExpressionCommon:
        if co2_intensity is None:
            co2_intensity = self.co2_intensity
        return (
            co2_intensity
            * self.signal_to_power(signal)
            / 1000
            * self.dt
            / 3600
        )

    def signal_to_co2_eur(
        self,
        signal: float | ca.GenericExpressionCommon,
        co2_intensity: float | None = None,
    ) -> float | ca.GenericExpressionCommon:
        return self.co2_cost * self.signal_to_co2(signal, co2_intensity)


class SimpleHeater(Actuator):
    """Heater model

    Simple Heater model that takes a signal percentage and returns the heating power.

    Examples:
    >>> max_unit = 1000  # Maximum heating power in W
    >>> heater = SimpleHeater(max_unit)
    >>> heater.signal_to_actuation(50)
    500.0
    """

    def __init__(self, max_unit, *args, **kwargs):
        super().__init__(max_unit, *args, **kwargs)


class SimpleFan(Actuator):
    """Fan model

    Simple Fan model that takes a signal percentage and returns the airflow.

    Examples:
    >>> max_unit = 1000  # Maximum airflow in m^3/s
    >>> ventilation = SimpleFan(max_unit)
    >>> ventilation.signal_to_actuation(50)
    500.0
    """

    def __init__(self, max_unit, power_per_unit=5, *args, **kwargs):
        super().__init__(max_unit, power_per_unit, *args, **kwargs)


class SimpleEvaporativeHumidifier(Actuator):
    """EvaporativeHumidifier model

    Simple EvaporativeHumidifier model that takes a signal percentage and returns the humidification output.

    # TODO: influences air speed as well
    https://growershouse.com/ideal-air-pro-series-ultra-sonic-humidifier-600-pint
    # TODO: Max. air volume flow [m³/h]	900
    https://uk.trotec.com/products-services/machinery-homecomfort/humidification/evaporation-humidifiers/b-400/

    Examples:
    >>> max_unit = 10  # Maximum humidification output in l/h
    >>> humidifier = SimpleEvaporativeHumidifier(max_unit)
    >>> humidifier.signal_to_actuation(20)
    2.0
    """

    def __init__(self, max_unit, power_per_unit=50, *args, **kwargs):
        super().__init__(
            max_unit,
            power_per_unit,
            *args,
            **kwargs,
        )


class SimpleCO2Generator(Actuator):
    """CO2 Generator model

    Simple CO2Generator model that takes a signal percentage and returns the co2 generation.

    https://www.hotboxworld.com/product/co2-generator
    https://www.eugardencenter.com/en/hotbox-co2-generator.html

    Examples:
    >>> max_unit = 10  # Maximum co2 generation in kg/h
    >>> co2generator = SimpleCO2Generator(max_unit)
    >>> co2generator.signal_to_actuation(20)
    2.0
    """

    def __init__(self, max_unit, power_per_unit=4400, *args, **kwargs):
        super().__init__(max_unit, power_per_unit, *args, **kwargs)
