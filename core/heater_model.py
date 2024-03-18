from abc import ABC, abstractmethod

import casadi as ca


class Heater(ABC):
    """Heater model interface

    Interface for heater models
    """

    def __init__(self, max_power):
        self.max_power = max_power  # Maximum heating power in watts

    @abstractmethod
    def transform_one(self, signal: float) -> float:
        pass


class SimpleHeater(Heater):
    """Heater model

    Simple Heater model that takes a signal percentage and returns the heating power

    Examples:
    >>> max_power = 1000  # Maximum heating power in watts
    >>> heater = SimpleHeater(max_power)
    >>> heater.transform_one(50)
    500.0
    """

    def __init__(self, max_power):
        self.max_power = max_power  # Maximum heating power in watts

    def transform_one(
        self, signal: float | ca.GenericExpressionCommon
    ) -> float | ca.MX:
        # Ensure signal is within the range of 0 to 100
        if isinstance(signal, ca.GenericExpressionCommon):
            signal = ca.if_else(signal < 0, 0, signal)
            signal = ca.if_else(signal > 100, 100, signal)
        else:
            signal = max(0.0, min(100.0, signal))

        # Calculate heating power based on the signal
        return (signal / 100.0) * self.max_power
