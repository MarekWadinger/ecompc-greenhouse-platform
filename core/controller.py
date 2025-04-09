from typing import Literal

import numpy as np
import pandas as pd
from casadi import SX, vertcat
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.simulator import Simulator

from core.actuators import Actuator
from core.greenhouse_model import GreenHouse, x_init
from core.lettuce_model import DRY_TO_WET_RATIO


class GreenHouseModel(Model):  # Create a model instance
    def __init__(
        self,
        gh_model: GreenHouse,
        climate_vars,
        model_type: Literal["continuous", "discrete"] = "continuous",
        lettuce_price=0.0054,  # EUR/g
    ):
        self.gh = gh_model
        self.dt = self.gh.dt
        self.lettuce_price = lettuce_price

        super().__init__(model_type)

        # Define state and control variables
        t = self.set_variable(var_type="_tvp", var_name="t")
        x = self.set_variable(
            var_type="_x", var_name="x", shape=(len(x_init), 1)
        )

        # TODO: review whether it is needed to store in locals maybe we can use self.u directly?
        actuators = []
        for act, active in gh_model.active_actuators.items():
            if active:
                locals()[act] = self.set_variable(var_type="_u", var_name=act)
            else:
                locals()[act] = 0.0
            actuators.append(locals()[act])

        self.u_ = vertcat(*actuators)
        tvp = {
            name: self.set_variable(var_type="_tvp", var_name=name)
            for name in climate_vars
        }

        if model_type == "discrete":
            # x_next = self.runge_kutta_step(t, x, self.u_, tuple(tvp.values()))
            x_next = self.backward_euler_step(
                t, x, self.u_, tuple(tvp.values())
            )
            # x_next = self.bdf2_step(
            #     t + self.dt, x, x_next, self.u_, tuple(tvp.values())
            # )
        else:
            x_next = gh_model.model(t, x, self.u_, tuple(tvp.values()))

        self.set_rhs("x", x_next)

        self.setup()

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

        Examples:
            >>> controller = Controller(...)
            >>> t = 0.0
            >>> x = np.zeros(10)  # Example state vector
            >>> u = np.array([50.0, 30.0])  # Example control inputs
            >>> tvp = (20.0, 400.0)  # Example time-varying parameters
            >>> controller.runge_kutta_step(t, x, u, tvp)

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

        Examples:
            >>> controller = Controller(...)
            >>> t = 0.0
            >>> x = np.zeros(10)  # Example state vector
            >>> u = np.array([50.0, 30.0])  # Example control inputs
            >>> tvp = (20.0, 400.0)  # Example time-varying parameters
            >>> controller.backward_euler_step(t, x, u, tvp)
        """
        f = self.gh.model(t + self.dt, x, u, tvp)
        return x + self.dt * f

    def bdf2_step(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        tvp: tuple,
    ) -> np.ndarray:
        """Performs a single BDF2 (Backward Differentiation Formula) integration step.

        Args:
            t: Current time.
            x: Current state vector.
            u: Control input vector.
            tvp: Tuple of time-varying parameters.

        Returns:
            The state vector after BDF2 integration.

        Examples:
            >>> controller = Controller(...)
            >>> t = 0.0
            >>> x = np.zeros(10)  # Example state vector
            >>> u = np.array([50.0, 30.0])  # Example control inputs
            >>> tvp = (20.0, 400.0)  # Example time-varying parameters
            >>> controller.bdf2_step(t, x, u, tvp)
        """
        return (
            4 / 3 * x
            - 1 / 3 * x
            + 2 / 3 * self.backward_euler_step(t, x, u, tvp)
        )

    def analyze_profit_and_costs(self, X, U, X0=None, energy_cost=None):
        profit = pd.Series(
            self.lettuce_price * X[-2:].sum() / DRY_TO_WET_RATIO * self.gh.A_c,
            index=["Lettuce profit "],
        )

        costs = pd.Series(
            index=[f"Energy ({i})" for i in U.columns]
            + [f"CO2 ({i})" for i in U.columns]
        )
        for act in [
            act for act, active in self.gh.active_actuators.items() if active
        ]:
            actuator: Actuator = getattr(self.gh, act.lower().replace(" ", ""))
            costs[f"Energy ({act})"] = -actuator.signal_to_eur(
                U[act], energy_cost
            ).sum()  # type: ignore
            costs[f"CO2 ({act})"] = -actuator.signal_to_co2_eur(U[act]).sum()  # type: ignore
        if X0 is not None:
            costs["Seedlings"] = (
                -self.lettuce_price
                * X0[-2:].sum()
                / DRY_TO_WET_RATIO
                * self.gh.A_c
            )  # type: ignore

        profit_costs = pd.concat([profit, costs]).rename("EUR")
        profit_costs["Total"] = profit_costs.sum()
        return profit_costs

    def init_states(self, x, u, tvp, steps=300):
        x_next = x
        for t in range(steps):
            x_next = self.runge_kutta_step(t, x_next, u, tvp)
        return np.array(x_next).flatten()


class EconomicMPC(MPC):
    def __init__(
        self,
        model: GreenHouseModel,
        climate,
        N=60,  # number of control intervals
        x_weight_init=x_init[-2:],
        u_min: float | list[float] = 0.0,
        u_max: float | list[float] = 100.0,
        co2_we_care: bool = True,
    ):
        # Define optimization variables
        self.x = model.x["x"]
        self.u = model.u
        self.N = N  # Store horizon length
        self.model = model  # Store model reference

        self.climate = climate

        self.lettuce_price = model.lettuce_price
        self.cultivated_area = model.gh.A_c

        if isinstance(u_min, list):
            assert len(u_min) == model.n_u
        if isinstance(u_max, list):
            assert len(u_max) == model.n_u

        # Create an MPC instance
        super().__init__(model)
        x_init_ = x_init
        x_init_[-2:] = x_weight_init
        self.x0 = x_init_

        # Set parameters
        setup_mpc = {
            "n_horizon": N,
            "t_step": model.dt,
            "supress_ipopt_output": True,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 1,
            "collocation_ni": 1,
            "nl_cons_single_slack": True,
            "store_full_solution": True,  # Important for warm starting
            "nlpsol_opts": {
                "ipopt": {  # https://coin-or.github.io/Ipopt/OPTIONS.html
                    "max_iter": 100,  # TODO: 100 is not enough
                    "tol": 0.01,  # obj. function is in EUR, 0.01 cent tol should be enough for given Ts
                    "max_cpu_time": 0.5,  # seconds
                    # "linear_solver": "MA57",  # https://licences.stfc.ac.uk/product/coin-hsl
                    "warm_start_init_point": "yes",
                    # "warm_start_same_structure": "yes",  # Speeds things up, but does not seem to optimize anything
                    "warm_start_entire_iterate": "yes",
                    "mu_allow_fast_monotone_decrease": "yes",
                    "fast_step_computation": "yes",
                    "print_level": 0,
                    "output_file": "/dev/null",
                    # "print_user_options": "yes",
                    # "print_options_documentation": "yes",
                    # "print_frequency_iter": 10,
                    "hessian_approximation": "limited-memory",
                    "limited_memory_max_history": 3,  # Minimal history
                    # "acceptable_tol": 1,  # More relaxed acceptable tolerance
                    # "acceptable_iter": 3,  # Accept solution faster
                    "nlp_scaling_method": "none",  # Skip scaling for speed
                    "fixed_variable_treatment": "make_parameter",  # Fa
                }
            },
        }
        self.set_param(**setup_mpc)
        lterm = -(
            self.lettuce_price
            * (self.x[-2] + self.x[-1] - self.x0["x"][-2] - self.x0["x"][-1])
            / DRY_TO_WET_RATIO
            * self.cultivated_area
        )
        for i, act in enumerate([
            act for act, active in model.gh.active_actuators.items() if active
        ]):
            actuator: Actuator = getattr(
                model.gh, act.lower().replace(" ", "")
            )
            lterm += actuator.signal_to_eur(
                model.u[act], model.tvp["Electricity price [EUR/kWh]"]
            )
            if co2_we_care:
                lterm += actuator.signal_to_co2_eur(model.u[act])
            self.set_rterm(**{act: (1 / (model.dt * 1000))})  # type: ignore
            # Define path constraints
            self.bounds["lower", "_u", act] = (
                u_min[i] if isinstance(u_min, list) else u_min
            )
            self.bounds["upper", "_u", act] = (
                u_max[i] if isinstance(u_max, list) else u_max
            )

        # Define objective
        self.set_objective(mterm=SX(0.0), lterm=lterm)

        self.bounds["lower", "_x", "x"] = [0.0] * model.n_x
        # # Small violation due to numerical instability
        # self.set_nl_cons(
        #     "x", -model.x["x"], ub=0, soft_constraint=True, penalty_term_cons=10000
        # )

        # self.set_nl_cons("u", self.u_prev["u"] - model.u["u"], ub=0)

        # Get the template
        tvp_mpc_template = self.get_tvp_template()

        # Define the function (indexing is much simpler ...)
        def tvp_mpc_fun(t_now):
            if isinstance(t_now, np.ndarray):
                t_now = t_now[0]
            t_ = int(t_now // model.dt)
            # N is the horizon with given ts
            for k in range(N + 1):
                tvp_mpc_template["_tvp", k, "t"] = t_now + k * model.dt
                for key in self.climate.columns:
                    tvp_mpc_template["_tvp", k, key] = self.climate[key].iloc[
                        t_ + k
                    ]
            return tvp_mpc_template

        # Set the tvp_fun:
        self.set_tvp_fun(tvp_mpc_fun)

        self.setup()
        self.set_initial_guess()

    def make_step(self, x0):
        """Make one MPC step with warm-starting.

        Args:
            x0: Current state vector.

        Returns:
            Optimal control input for the current step.
        """
        for k in range(0, self._settings.n_horizon - 1):
            self.opt_x_num["_u", k, :] = self.opt_x_num["_u", k + 1, :]
        return super().make_step(x0)


class GreenhouseSimulator(Simulator):
    def __init__(
        self,
        model,
        climate,
        x_weight_init=x_init[-2:],
    ):
        super().__init__(model)

        self.climate = climate
        # Convert to numpy array directly for fastest access
        self.climate_values = np.array([
            climate[col].values for col in climate.columns
        ])
        self.climate_keys = list(climate.columns)
        self.climate_len = len(climate)

        params_simulator = {
            "t_step": model.dt,
            "abstol": 1e-2,
            "reltol": 1e-2,
        }

        if model.model_type == "continuous":
            # Configure simulator for stiff systems
            params_simulator.update({
                "integration_tool": "cvodes",
                "integration_opts": {
                    "max_num_steps": 100,  # Increase max steps for stiff systems
                    "sensitivity_method": "simultaneous",
                    "linear_multistep_method": "bdf",  # Better for stiff systems
                    "nonlinear_solver_iteration": "newton",  # More robust for stiff systems
                    "newton_scheme": "direct",  # Direct solver for better convergence
                    "max_step_size": 100,  # Limit step size for stability
                    "min_step_size": 1,  # Prevent too small steps
                    "scale_abstol": True,  # Scale tolerances for better performance
                    "always_recalculate_jacobian": False,  # Reuse Jacobian when possible
                    "use_preconditioner": True,  # Improve convergence
                    "max_krylov": 10,  # Maximum Krylov dimension
                    "step0": 1e-4,  # Initial step size
                },
            })

        self.set_param(**params_simulator)

        # Get the template
        tvp_sim_template = self.get_tvp_template()

        # Simple, fast function with minimal overhead
        def tvp_sim_fun(t_now):
            if isinstance(t_now, np.ndarray):
                t_now = t_now[0]

            # Fast index calculation with bounds check
            t_ = min(int(t_now // model.dt), self.climate_len - 1)

            # Set time once
            tvp_sim_template["t"] = t_now

            # Use pre-slice for faster access
            current_climate = self.climate_values[:, t_]

            # Fast assignment using direct array access
            for i, key in enumerate(self.climate_keys):
                tvp_sim_template[key] = current_climate[i]

            return tvp_sim_template

        # Set the tvp_fun:
        self.set_tvp_fun(tvp_sim_fun)

        self.setup()

        x_init_ = x_init
        x_init_[-2:] = x_weight_init
        self.x0 = x_init_
        self.set_initial_guess()
