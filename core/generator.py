"""Generator functions for the simulation."""
from typing import Protocol, Union

import numpy as np


class GenProt(Protocol):
    def __name__(self) -> str:
        ...

    def __call__(
        self,
        t: int,
        *,
        u_min: Union[list[float], float] = ...,
        u_max: Union[list[float], float] = ...,
    ) -> Union[list[float], float]:
        ...


def gen_step(
    t: int,
    u_min: Union[list[float], float],
    u_max: Union[list[float], float],
    **_,
) -> Union[list[float], float]:
    u = u_min
    if t >= 10:
        u = u_max
    return u


def gen_u_daily_sine(
    t: int,
    sim_time: int,
    t_s: int,
    u_min: list,
    u_max: list,
    **_,
) -> list[float]:
    """Generate the sine waves for the inputs.

    Args:
        t: time step
        sim_time: total simulation time for normalization

    Returns:
       tuple of us with len equal to len(u_min)
    """
    t_ = t / (sim_time - 1)  # Normalize t to the range [0, 1]
    num_sines = 2 * sim_time * t_s / 86400

    u: list[float] = []
    for u_min_, u_max_ in zip(u_min, u_max):
        # Calculate the values of each sine wave at time t
        u_: float = u_min_ + (u_max_ - u_min_) / 2 * (
            1 + np.sin(num_sines * np.pi * t_)
        )
        u.append(u_)
    return u


def gen_u_control(
    t: int,
    x_ode_prev: list,
    x_ref: list,
    u_min: float,
    u_max: float,
    **_,
):
    k = np.array([1.0, 1.0, 1.0])
    u = -k @ (np.array(x_ode_prev) - np.array(x_ref))
    u = max(min(u, u_max), u_min)
    return u


def gen_ref_step(
    t: int,
    x_ode_prev: list[float],
    **_,
):
    x_ref = x_ode_prev
    if t >= 50:
        x_ref = [50.0, 50.0]
    return x_ref


def validate_gen(gen: GenProt, **kwargs) -> None:
    try:
        _ = gen(**kwargs)
    except TypeError as e:
        print(
            f"Make sure that {gen.__name__} uses following arguments: "
            f"'{kwargs.keys()}', implements kwargs and returns a list"
        )
        raise e
