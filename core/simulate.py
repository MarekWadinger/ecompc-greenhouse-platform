"""Simulation of the system"""
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

TS = 1


def gen_sine_u(
        t: int,
        sim_time: int,
        u_min: list,
        u_max: list,
        ) -> list[float]:
    """Generate the sine waves for the inputs.

    Args:
        t: time step
        sim_time: total simulation time for normalization

    Returns:
       tuple of us with len equal to len(u_min)
    """
    t_ = t / (sim_time - 1)  # Normalize t to the range [0, 1]

    u: list[float] = []
    for u_min_, u_max_ in zip(u_min, u_max):
        # Calculate the values of each sine wave at time t
        u_: float = (u_min_ + (u_max_ - u_min_) / 2
                     * (1 + np.sin(2 * np.pi * t_)))
        u.append(u_)
    return u


def simulate_open_loop(
        system: Callable,
        sim_time: int = 300
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulation of the system in the open-loop.

    Args:
        system (Callable): _description_
        sim_time (optional): Simulation time in seconds [s]

    Returns:
        Simulation results for time, outputs and inputs
    """
    t_out = []
    y_out = []
    u_out = []

    u_min = [5, 0, 400]
    u_max = [25, 100, 800]
    # Initial mass of structural and non-structural dry weight
    x_ode_prev = (10, 10)

    for t in range(sim_time):
        tspan = [t, t+TS]
        u = gen_sine_u(t, sim_time, u_min, u_max)

        x_ode = solve_ivp(system, tspan, x_ode_prev, args=(u,), method="RK45")
        x_ode_prev = x_ode.y[:, -1]

        y_out.append(x_ode)
        t_out.append(t)
        u_out.append(u)
    y_out = np.array(y_out)
    t_out = np.array(t_out)
    u_out = np.array(u_out)

    return t_out, y_out, u_out


def simulate_closed_loop(
        system: Callable,
        sim_time: int = 300
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulation of the system in the closed-loop.

    Args:
        system (Callable): _description_
        sim_time (optional): Simulation time in seconds [s]

    Returns:
        Simulation results for time, outputs and inputs
    """
    t_out = []
    y_ref = []
    y_out = []
    u_out = []

    u_max = 10
    u_min = 0

    x_ode_prev = (10, 10)
    x_ref = x_ode_prev

    for t in range(sim_time):
        tspan = [t, t+TS]
        if t == 50:
            x_ref = (50, 50)

        k = np.array([1., 1., 1.])
        u = -k @ (np.array(x_ode_prev) - np.array(x_ref))
        u = max(min(u, u_max), u_min)
        x_ode = solve_ivp(system, tspan, x_ode_prev, args=(u,), method="RK45")
        x_ode_prev = x_ode.y[:, -1]

        y_out.append(x_ode)
        y_ref.append(x_ref)
        t_out.append(t)
        u_out.append(u)
    y_ref = np.array(y_ref)
    y_out = np.array(y_out)
    t_out = np.array(t_out)
    u_out = np.array(u_out)

    return t_out, y_ref, y_out, u_out
