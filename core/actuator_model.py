from abc import ABC

import casadi as ca


class Actuator(ABC):
    """Actuator model interface

    Interface for actuator models

    Args:
        max_unit: Maximum reachable actuation [unit]
        power_per_unit: Power per Actuation Unit [W/unit]
        energy_cost: Cost of Energy [EUR/kWh]
        co2_per_energy: Carbon Intensity [gCO₂eq/kWh]  https://app.electricitymaps.com/zone/SK
        co2_cost: Social cost of CO2 [EUR/gCO₂eq]  https://www.theguardian.com/environment/article/2024/may/17/economic-damage-climate-change-report
        dt: Duration [s]
    """

    def __init__(
        self,
        max_unit: float,
        power_per_unit: float = 1.0,
        energy_cost: float = 0.0612,
        co2_per_energy: float = 200.0,
        co2_cost: float = 0.001,
        efficiency: float = 0.8,
        dt: float = 1.0,
    ):
        self.max_unit = max_unit  # Maximum value of actuation
        self.power_per_unit = power_per_unit
        self.energy_cost = energy_cost
        self.co2_per_energy = co2_per_energy
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
            signal = max(0.0, min(100.0, signal))

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
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.GenericExpressionCommon:
        return (
            self.energy_cost
            * self.signal_to_power(signal)
            / 1000
            * self.dt
            / 3600
        )

    def signal_to_co2(
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.GenericExpressionCommon:
        return (
            self.co2_per_energy
            * self.signal_to_power(signal)
            / 1000
            * self.dt
            / 3600
        )

    def signal_to_co2_eur(
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.GenericExpressionCommon:
        return self.co2_cost * self.signal_to_co2(signal)


class SimpleHeater(Actuator):
    """Heater model

    Simple Heater model that takes a signal percentage and returns the heating power

    Examples:
    >>> max_act = 1000  # Maximum heating power in watts
    >>> heater = SimpleHeater(max_act)
    >>> heater.signal_to_actuation(50)
    500.0
    """

    def __init__(self, max_act, power_per_unit=1):
        super().__init__(
            max_act,
            power_per_unit,
        )


class SimpleVentilation(Actuator):
    """Ventilation model

    Simple Ventilation model that takes a signal percentage and returns the airflow

    Examples:
    >>> max_flow = 1000  # Maximum airflow in m^3/s
    >>> ventilation = SimpleVentilation(max_flow)
    >>> ventilation.signal_to_actuation(50)
    500.0
    """

    def __init__(self, max_act, power_per_unit=5):
        super().__init__(
            max_act,
            power_per_unit,
        )
