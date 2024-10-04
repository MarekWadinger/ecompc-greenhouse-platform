import numpy as np
from casadi import SX, vertcat, vertsplit
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.simulator import Simulator

from core.greenhouse_model import GreenHouse, x_init
from core.lettuce_model import DRY_TO_WET_RATIO


class GreenHouseModel(Model):  # Create a model instance
    def __init__(
        self,
        gh_model: GreenHouse,
        climate_vars,
    ):
        self.gh = gh_model
        self.dt = self.gh.dt

        super().__init__("discrete")

        # Define state and control variables
        t = self.set_variable(var_type="_tvp", var_name="t")
        x = self.set_variable(
            var_type="_x", var_name="x", shape=(len(x_init), 1)
        )

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

        # Define the model equations
        def gh_model_(t, x, u, tvp):
            return vertcat(
                *gh_model.model(
                    t, vertsplit(x), vertsplit(u), climate=tuple(tvp.values())
                )
            )

        k1 = gh_model_(t, x, self.u_, tvp)
        k2 = gh_model_(t, x + self.dt / 2 * k1, self.u_, tvp)
        k3 = gh_model_(t, x + self.dt / 2 * k2, self.u_, tvp)
        k4 = gh_model_(t, x + self.dt * k3, self.u_, tvp)
        x_next = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # x_next = backward_euler_step(f, t, x, self.u_, tvp, dt)
        # x_next = bdf2_step(f, t + dt, x, x_next, self.u_, tvp, dt)

        self.set_rhs("x", x_next)

        self.setup()


class EconomicMPC(MPC):
    def __init__(
        self,
        model: GreenHouseModel,
        climate,
        lettuce_price=0.0054,  # EUR/g
        N=60,  # number of control intervals
        x_weight_init=x_init[-2:],
        u_min: float | list[float] = 0.0,
        u_max: float | list[float] = 100.0,
    ):
        # Define optimization variables
        self.x = model.x["x"]
        self.u = model.u

        self.climate = climate

        self.lettuce_price = lettuce_price
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
            "state_discretization": "discrete",
            "nlpsol_opts": {
                "ipopt": {  # https://coin-or.github.io/Ipopt/OPTIONS.html
                    # "max_iter": 3,
                    # "linear_solver": "MA57",  # https://licences.stfc.ac.uk/product/coin-hsl
                    "warm_start_init_point": "yes",
                    "mu_allow_fast_monotone_decrease": "yes",
                    "fast_step_computation": "yes",
                    "print_level": 0,
                    # "output_file": "ipopt.out",
                    # "print_user_options": "yes",
                    # "print_options_documentation": "yes",
                    "print_frequency_iter": 10,
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
        for act in [
            act for act, active in model.gh.active_actuators.items() if active
        ]:
            actuator = getattr(model.gh, act.lower().replace(" ", ""))
            lterm += actuator.signal_to_eur(model.u[act])
            lterm += actuator.signal_to_co2_eur(model.u[act])
            self.set_rterm(**{act: (1 / (model.dt * 1000))})  # type: ignore
            # Define path constraints
            self.bounds["lower", "_u", act] = u_min
            self.bounds["upper", "_u", act] = u_max

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


class GreenhouseSimulator(Simulator):
    def __init__(
        self,
        model,
        climate,
        x_weight_init=x_init[-2:],
    ):
        super().__init__(model)

        self.climate = climate

        params_simulator = {"t_step": model.dt}

        self.set_param(**params_simulator)

        # Get the template
        tvp_sim_template = self.get_tvp_template()

        # Define the function (indexing is much simpler ...)
        def tvp_sim_fun(t_now):
            if isinstance(t_now, np.ndarray):
                t_now = t_now[0]
            t_ = int(t_now // model.dt)
            tvp_sim_template["t"] = t_now
            for key in climate.columns:
                tvp_sim_template[key] = climate[key].iloc[t_]
            return tvp_sim_template

        # Set the tvp_fun:
        self.set_tvp_fun(tvp_sim_fun)

        self.setup()

        x_init_ = x_init
        x_init_[-2:] = x_weight_init
        self.x0 = x_init_
        self.set_initial_guess()
