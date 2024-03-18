from abc import ABC

import casadi as ca


class Actuator(ABC):
    """Actuator model interface

    Interface for actuator models

    Args:
        max_unit: Maximum reachable actuation [unit]
        power_per_unit: Power per Actuation Unit [kWh/unit]
        price_per_power: Price of Power [EUR/kWh]
        co2_per_power: Carbon Intensity [gCO₂eq/kWh]
    """

    def __init__(
        self,
        max_unit: float,
        power_per_unit: float = 1000.0,
        price_per_power: float = 0.0612,
        co2_per_power: float = 250.0,
    ):
        self.max_unit = max_unit  # Maximum value of actuation
        self.power_per_unit = power_per_unit
        self.price_per_power = price_per_power
        self.co2_per_power = co2_per_power

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
        return self.signal_to_actuation(signal) * self.power_per_unit

    def signal_to_price(
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.GenericExpressionCommon:
        return self.signal_to_power(signal) * self.price_per_power

    def signal_to_co2(
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.GenericExpressionCommon:
        return self.signal_to_power(signal) * self.co2_per_power


class SimpleHeater(Actuator):
    """Heater model

    Simple Heater model that takes a signal percentage and returns the heating power

    Examples:
    >>> max_act = 1000  # Maximum heating power in watts
    >>> heater = SimpleHeater(max_act)
    >>> heater.transform_one(50)
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
    >>> ventilation.transform_one(50)
    500.0
    """

    def __init__(self, max_act, power_per_unit=5000):
        super().__init__(
            max_act,
            power_per_unit,
        )
