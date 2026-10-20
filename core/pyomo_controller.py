from typing import List, Literal, Union

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints

from core.actuators import Actuator
from core.greenhouse_model import GreenHouse, x_init
from core.lettuce_model import DRY_TO_WET_RATIO


# Helper functions for actuator cost calculations
def signal_to_eur_value(
    actuator: Actuator, signal_value: float, energy_cost: float = None
) -> float:
    """Calculate the EUR cost for a single signal value.

    Args:
        actuator: Actuator object
        signal_value: Control signal value
        energy_cost: Energy cost (optional)

    Returns:
        Cost in EUR
    """
    # Base conversion factors from the Actuator class
    power_kw = actuator.signal_to_power(signal_value)

    if energy_cost is None:
        energy_cost = actuator.energy_cost

    cost = power_kw * energy_cost * actuator.dt / 3600
    return cost


def signal_to_co2_eur_value(actuator: Actuator, signal_value: float) -> float:
    """Calculate the CO2 cost in EUR for a single signal value.

    Args:
        actuator: Actuator object
        signal_value: Control signal value

    Returns:
        CO2 cost in EUR
    """
    power_kw = actuator.signal_to_power(signal_value)
    co2_kg = power_kw * actuator.co2_intensity * actuator.dt / 3600
    co2_eur = co2_kg * actuator.co2_cost
    return co2_eur


class PyomoGreenHouseModel:
    """PyOmo implementation of the greenhouse model."""

    def __init__(
        self,
        gh_model: GreenHouse,
        climate_vars: List[str],
        model_type: Literal["continuous", "discrete"] = "discrete",
        lettuce_price: float = 0.0054,  # EUR/g
    ):
        """Initialize the PyOmo greenhouse model.

        Args:
            gh_model: Greenhouse model object
            climate_vars: List of climate variable names
            model_type: Type of model (continuous or discrete)
            lettuce_price: Price of lettuce in EUR/g
        """
        self.gh = gh_model
        self.dt = self.gh.dt
        self.lettuce_price = lettuce_price
        self.model_type = model_type
        self.climate_vars = climate_vars
        self.n_x = len(x_init)
        self.n_u = sum(
            1 for active in gh_model.active_actuators.values() if active
        )
        self.active_actuators = [
            act for act, active in gh_model.active_actuators.items() if active
        ]

    def runge_kutta_step(
        self, t: float, x: np.ndarray, u: np.ndarray, tvp: tuple
    ) -> np.ndarray:
        """Performs a single Runge-Kutta 4th order integration step.

        Args:
            t: Current time.
            x: Current state vector.
            u: Control input vector.
            tvp: Tuple of time-varying parameters.

        Returns:
            The next state vector after integration.
        """
        k1 = self.gh.model(t, x, u, tvp)
        k2 = self.gh.model(t, x + self.dt / 2 * k1, u, tvp)
        k3 = self.gh.model(t, x + self.dt / 2 * k2, u, tvp)
        k4 = self.gh.model(t, x + self.dt * k3, u, tvp)
        return x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def backward_euler_step(
        self, t: float, x: np.ndarray, u: np.ndarray, tvp: tuple
    ) -> np.ndarray:
        """Performs a single backward Euler integration step.

        Args:
            t: Current time.
            x: Current state vector.
            u: Control input vector.
            tvp: Tuple of time-varying parameters.

        Returns:
            The next state vector after integration.
        """
        f = self.gh.model(t + self.dt, x, u, tvp)
        return x + self.dt * f

    def init_states(self, x, u, tvp, steps=300):
        """Initialize states by simulating the model forward.

        Args:
            x: Initial state vector
            u: Control input vector
            tvp: Time-varying parameters
            steps: Number of steps to simulate

        Returns:
            Initialized state vector
        """
        x_next = x
        for t in range(steps):
            x_next = self.runge_kutta_step(t, x_next, u, tvp)
        return np.array(x_next).flatten()

    def analyze_profit_and_costs(self, X, U, energy_cost=None):
        """Analyze profit and costs from simulation results.

        Args:
            X: State trajectory
            U: Control trajectory
            energy_cost: Energy cost data

        Returns:
            DataFrame with profit and cost analysis
        """
        profit = pd.Series(
            self.lettuce_price * X[-2:].sum() / DRY_TO_WET_RATIO * self.gh.A_c,
            index=["Lettuce profit "],
        )

        costs = pd.Series(
            index=[f"Energy ({i})" for i in U.columns]
            + [f"CO2 ({i})" for i in U.columns]
        )
        for act in self.active_actuators:
            actuator: Actuator = getattr(self.gh, act.lower().replace(" ", ""))
            costs[f"Energy ({act})"] = -actuator.signal_to_eur(
                U[act], energy_cost
            ).sum()
            costs[f"CO2 ({act})"] = -actuator.signal_to_co2_eur(U[act]).sum()

        profit_costs = pd.concat([profit, costs]).rename("EUR")
        profit_costs["Total"] = profit_costs.sum()
        return profit_costs


class PyomoEconomicMPC:
    """PyOmo implementation of Economic MPC for greenhouse control."""

    def __init__(
        self,
        model: PyomoGreenHouseModel,
        climate: pd.DataFrame,
        N: int = 60,  # number of control intervals
        x_weight_init=x_init[-2:],
        u_min: Union[float, List[float]] = 0.0,
        u_max: Union[float, List[float]] = 100.0,
        co2_we_care: bool = True,
    ):
        """Initialize the PyOmo Economic MPC.

        Args:
            model: PyOmo greenhouse model
            climate: Climate data DataFrame
            N: Prediction horizon length
            x_weight_init: Initial weights for lettuce components
            u_min: Minimum control input(s)
            u_max: Maximum control input(s)
            co2_we_care: Whether to consider CO2 emissions in the objective
        """
        self.model = model
        self.climate = climate
        self.N = N
        self.dt = model.dt
        self.co2_we_care = co2_we_care
        self.lettuce_price = model.lettuce_price
        self.cultivated_area = model.gh.A_c

        # Initial conditions
        x_init_ = x_init.copy()
        x_init_[-2:] = x_weight_init
        self.x0 = x_init_

        # Control bounds
        if isinstance(u_min, list):
            assert len(u_min) == model.n_u
            self.u_min = u_min
        else:
            self.u_min = [u_min] * model.n_u

        if isinstance(u_max, list):
            assert len(u_max) == model.n_u
            self.u_max = u_max
        else:
            self.u_max = [u_max] * model.n_u

        # For warm starting
        self.prev_solution = None

    def _create_pyomo_model(self, t_now, x0=None):
        """Create the PyOmo optimization model.

        Args:
            t_now: Current time
            x0: Current state (if None, use self.x0)

        Returns:
            PyOmo ConcreteModel
        """
        if x0 is None:
            x0 = self.x0

        # Calculate time index
        t_ = int(t_now // self.dt)

        # Create concrete model
        m = pyo.ConcreteModel()

        # Sets
        m.T = pyo.RangeSet(0, self.N)  # Time steps
        m.X = pyo.RangeSet(0, self.model.n_x - 1)  # States
        m.U = pyo.RangeSet(0, self.model.n_u - 1)  # Controls

        # Variables
        m.x = pyo.Var(m.T, m.X, domain=pyo.NonNegativeReals)  # States
        m.u = pyo.Var(m.T, m.U, domain=pyo.NonNegativeReals)  # Controls

        # Parameters
        m.x0_param = pyo.Param(
            m.X, initialize={i: x0[i] for i in range(len(x0))}
        )
        m.climate_params = {}

        for k in range(self.N + 1):
            for i, key in enumerate(self.climate.columns):
                param_name = f"tvp_{key}_{k}"
                time_idx = min(t_ + k, len(self.climate) - 1)
                m.climate_params[param_name] = self.climate[key].iloc[time_idx]

        # Initial conditions
        def init_constraint_rule(m, i):
            return m.x[0, i] == m.x0_param[i]

        m.init_constraint = pyo.Constraint(m.X, rule=init_constraint_rule)

        # Control bounds
        def u_lb_constraint_rule(m, t, i):
            return m.u[t, i] >= self.u_min[i]

        def u_ub_constraint_rule(m, t, i):
            return m.u[t, i] <= self.u_max[i]

        m.u_lb_constraint = pyo.Constraint(m.T, m.U, rule=u_lb_constraint_rule)
        m.u_ub_constraint = pyo.Constraint(m.T, m.U, rule=u_ub_constraint_rule)

        # Precompute the next state for each time step since we need to use the model
        # This is a workaround for the limitation of using Pyomo variables in the model
        m.next_states = {}

        for t in range(self.N):
            # Define initial value expressions
            for i in range(self.model.n_x):
                # First time step uses initial state
                if t == 0:
                    m.x[t, i] = m.x0_param[i]

            # Define state dynamics constraints for each state variable
            for i in range(self.model.n_x):
                # Current time for this step
                current_time = t_now + t * self.dt

                # Get climate parameters for this timestep
                tvp_vals = []
                for key in self.climate.columns:
                    param_name = f"tvp_{key}_{t}"
                    tvp_vals.append(m.climate_params[param_name])

                # Add dynamics constraint
                def state_dynamics_rule(
                    m, t, i, current_time=current_time, tvp_vals=tvp_vals
                ):
                    return (
                        m.x[t + 1, i]
                        == m.x[t, i]
                        + self.dt
                        * self.model.gh.model(
                            current_time,
                            [m.x[t, j] for j in range(self.model.n_x)],
                            [m.u[t, j] for j in range(self.model.n_u)],
                            tuple(tvp_vals),
                        )[i]
                    )

                # Using a unique constraint name for each state and time step
                setattr(
                    m,
                    f"state_dynamics_{t}_{i}",
                    pyo.Constraint(expr=state_dynamics_rule(m, t, i)),
                )

        # Objective function - simplified version that avoids direct use of Pyomo variables
        def objective_rule(m):
            obj = 0

            # Lettuce growth profit
            for t in range(self.N):
                lettuce_growth = (
                    self.lettuce_price
                    * (
                        m.x[t + 1, -2]
                        + m.x[t + 1, -1]
                        - m.x[t, -2]
                        - m.x[t, -1]
                    )
                    / DRY_TO_WET_RATIO
                    * self.cultivated_area
                )
                obj -= lettuce_growth

            # Energy and CO2 costs
            for t in range(self.N):
                for i, act in enumerate(self.model.active_actuators):
                    actuator = getattr(
                        self.model.gh, act.lower().replace(" ", "")
                    )

                    # Get energy cost for this timestep
                    energy_cost_param = m.climate_params.get(
                        f"tvp_energy_cost_{t}", actuator.energy_cost
                    )

                    # Calculate costs
                    power_expr = m.u[t, i] * (
                        actuator.max_power / 100.0
                    )  # Linear approximation
                    energy_cost = (
                        power_expr * energy_cost_param * self.dt / 3600
                    )
                    obj += energy_cost

                    # CO2 cost
                    if self.co2_we_care:
                        co2_kg = (
                            power_expr
                            * actuator.co2_intensity
                            * self.dt
                            / 3600
                        )
                        co2_cost = co2_kg * actuator.co2_cost
                        obj += co2_cost

                    # Rate penalty
                    if t > 0:
                        rate_penalty = (1 / (self.dt * 1000)) * (
                            m.u[t, i] - m.u[t - 1, i]
                        ) ** 2
                        obj += rate_penalty

            return obj

        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        return m

    def make_step(self, x0=None):
        """Make one MPC step.

        Args:
            x0: Current state vector (if None, use self.x0)

        Returns:
            Optimal control input for the current step
        """
        if x0 is not None:
            self.x0 = x0

        # Current time is 0 for the first step
        t_now = 0

        # Create the optimization model
        m = self._create_pyomo_model(t_now, self.x0)

        # Initialize solver
        solver = pyo.SolverFactory("ipopt")
        solver.options["max_iter"] = 100
        solver.options["tol"] = 0.01
        solver.options["max_cpu_time"] = 0.5
        solver.options["warm_start_init_point"] = "yes"
        solver.options["warm_start_entire_iterate"] = "yes"
        solver.options["mu_allow_fast_monotone_decrease"] = "yes"
        solver.options["fast_step_computation"] = "yes"
        solver.options["print_level"] = 0

        # Apply warm start if available
        if self.prev_solution is not None:
            for t in range(min(self.N, len(self.prev_solution))):
                for i in range(self.model.n_u):
                    if t + 1 < self.N:
                        # Use variable.fix() instead of set_value
                        m.u[t, i].value = self.prev_solution[t + 1][i]
                    else:
                        m.u[t, i].value = self.prev_solution[-1][i]

        # Solve the optimization problem
        try:
            results = solver.solve(m, tee=False)

            # Check solver status
            if (
                results.solver.status == pyo.SolverStatus.ok
                and results.solver.termination_condition
                == pyo.TerminationCondition.optimal
            ):
                # Extract solution
                u_optimal = np.zeros(self.model.n_u)
                for i in range(self.model.n_u):
                    u_optimal[i] = pyo.value(m.u[0, i])

                # Store solution for warm start
                self.prev_solution = []
                for t in range(self.N):
                    u_t = [pyo.value(m.u[t, i]) for i in range(self.model.n_u)]
                    self.prev_solution.append(u_t)

                return u_optimal
            else:
                # Log infeasible constraints
                log_infeasible_constraints(m)
                raise RuntimeError(
                    f"Solver failed: {results.solver.status}, "
                    f"{results.solver.termination_condition}"
                )
        except Exception as e:
            # If we encounter an error during solving, try a simpler approach
            print(f"Error during optimization: {e}")
            print("Using fallback control strategy...")

            # Simple fallback strategy: use mid-range control values
            u_optimal = np.array([50.0] * self.model.n_u)
            return u_optimal


class PyomoGreenhouseSimulator:
    """PyOmo-based simulator for greenhouse system."""

    def __init__(
        self,
        model: PyomoGreenHouseModel,
        climate: pd.DataFrame,
        x_weight_init=x_init[-2:],
    ):
        """Initialize the PyOmo greenhouse simulator.

        Args:
            model: PyOmo greenhouse model
            climate: Climate data DataFrame
            x_weight_init: Initial weights for lettuce components
        """
        self.model = model
        self.climate = climate

        # Convert to numpy array for faster access
        self.climate_values = np.array(
            [climate[col].values for col in climate.columns]
        )
        self.climate_keys = list(climate.columns)
        self.climate_len = len(climate)

        # Initial conditions
        x_init_ = x_init.copy()
        x_init_[-2:] = x_weight_init
        self.x0 = x_init_

    def make_step(self, u):
        """Simulate one step of the system dynamics.

        Args:
            u: Control input

        Returns:
            Next state
        """
        # Get climate data for current timestep
        t_ = 0  # Assuming t_now = 0 for simplicity
        current_climate = self.climate_values[:, min(t_, self.climate_len - 1)]
        tvp = tuple(current_climate)

        # Simulate using selected integration method
        if self.model.model_type == "continuous":
            x_next = self.model.runge_kutta_step(0, self.x0, u, tvp)
        else:
            x_next = self.model.backward_euler_step(0, self.x0, u, tvp)

        self.x0 = x_next
        return x_next
