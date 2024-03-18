# import os
import sys
from functools import partial

# import time
from pathlib import Path

import numpy as np
from casadi import Opti, mtimes, vertcat, vertsplit

sys.path.insert(1, str(Path().resolve()))
## load initial conditions & climate data
from GES_Example import climate, z  # noqa: E402

from core.greenhouse_model import model  # noqa: E402

greenhouse_model = partial(model, climate=climate)


def f(k, x, u):
    return vertcat(*greenhouse_model(k, vertsplit(x), vertsplit(u)))


if __name__ == "__main__":
    ## optimization problem
    opti = Opti()  # Optimization problem

    N = 50  # number of control intervals
    sec_in_day = 86400
    # x_ref = [25 + T_k, 7.5869e-4]  #TODO reference state
    x_ref = [z_ * 2.0 for z_ in z]
    # ---- decision variables ---------
    n_states = 14
    X = opti.variable(n_states, N + 1)  # state trajectory
    U = opti.variable(2, N)  # control trajectory (Q_heating, R_a)
    Ts = 60  # TODO choose appropriate step size
    dt = Ts
    Q = np.eye(n_states)
    R = np.eye(2)
    u_min = [0.0, 0.0]
    u_max = [100.0, 100.0]
    x_min = [0.0] * n_states
    x_max = [10000.0] * n_states

    # Get x0 to feasible initial state - hopefully...
    x0 = np.array(z)
    u0 = np.array([0.0, 0.0])
    for k in range(N):
        k1 = greenhouse_model(k, x0, u0)
        k2 = greenhouse_model(k, x0 + dt / 2 * k1, u0)
        k3 = greenhouse_model(k, x0 + dt / 2 * k2, u0)
        k4 = greenhouse_model(k, x0 + dt * k3, u0)
        x_next = x0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x0 = x_next

    # Implement Runge-Kutta 4 integrator manually 😱
    for k in range(N):
        k1 = f(k, X[:, k], U[:, k])
        k2 = f(k, X[:, k] + dt / 2 * k1, U[:, k])
        k3 = f(k, X[:, k] + dt / 2 * k2, U[:, k])
        k4 = f(k, X[:, k] + dt * k3, U[:, k])
        x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        for state in range(X.shape[0]):
            opti.subject_to(X[state, k + 1] == x_next[state])

    for i in range(U.shape[0]):
        opti.subject_to(U[i, :] >= u_min[i])
        opti.subject_to(U[i, :] <= u_max[i])
    for i in range(X.shape[0]):
        opti.subject_to(X[i, :] >= x_min[i])
        opti.subject_to(X[i, :] <= x_max[i])

    for i in range(X.shape[0]):
        opti.set_initial(X[i, :], x0[i])
        opti.subject_to(X[i, 0] == x0[i])

    opti.minimize(
        mtimes((X[:, 0] - x_ref).T, mtimes(Q, (X[:, 0] - x_ref)))
        + mtimes(U[:, 0].T, mtimes(R, U[:, 0]))
    )

    opti.solver("ipopt")  # set numerical backend
    sol = opti.solve()  # actual solve
