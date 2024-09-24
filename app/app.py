import os
import sys
from contextlib import contextmanager
from functools import partial
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from stqdm import stqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.controller import EconomicMPC, GreenHouseModel, GreenhouseSimulator
from core.greenhouse_model import model as gh_model
from core.openmeteo_query import OpenMeteo
from examples.GES_Example import z


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# --- Functions ---
def update_selection_X():
    st.session_state.selected_X = st.session_state.multiselect_X
    st.session_state.m = len(st.session_state.selected_X)
    st.session_state.m
    if st.session_state.m > 0:
        st.session_state.disable_params = False
    else:
        st.session_state.disable_params = True


def update_selection_U():
    st.session_state.selected_U = st.session_state.multiselect_U
    st.session_state.l = len(st.session_state.selected_U)


def export_fig(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="pdf")
    buf.seek(0)
    return buf.getvalue()


@st.cache_data
def concat_results(X, scores_dmd, scores_dmd_diff):
    df = pd.concat(
        [
            X,
            pd.Series(scores_dmd.real, index=X.index, name="DMD"),
            pd.Series(scores_dmd_diff.real, index=X.index, name="DMD (diff)"),
        ],
        axis=1,
    )
    return df


@st.cache_data
def plot(
    _timestamps,
    y_nexts,
    u0s,
    ums,
):
    y_nexts_ = np.array(y_nexts).reshape(-1, 2)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Lettuce Dry Weight (g)", "Actuation [%]"),
    )

    # Plot Lettuce Dry Weight
    fig.add_trace(
        go.Scatter(
            x=_timestamps,
            y=y_nexts_[:, -2],
            mode="lines",
            name="structural",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=_timestamps,
            y=y_nexts_[:, -1],
            mode="lines",
            name="nonstructural",
        ),
        row=1,
        col=1,
    )

    # Plot Actuation
    labels = ["fan", "heater"]
    colors = ["#1f77b4", "#ff7f0e"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=np.array(u0s).reshape(-1, 2)[:, i],
                mode="lines",
                name=label,
                line=dict(color=color),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=np.array(ums).reshape(-1, 4)[:, i],
                mode="lines",
                name=f"{label} min",
                line=dict(color=color, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=np.array(ums).reshape(-1, 4)[:, i + 2],
                mode="lines",
                name=f"{label} max",
                line=dict(color=color, dash="dash"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=600, width=800, title_text="Greenhouse Control")
    fig.update_yaxes(title_text="Lettuce Dry Weight (g)", row=1, col=1)
    fig.update_yaxes(title_text="Actuation [%]", row=2, col=1)

    return fig


@st.cache_data
def export_df(df):
    return df.to_csv().encode("utf-8")


# --- Page Config ---

st.set_page_config(
    page_title="Green House Control",
    layout="wide",
    initial_sidebar_state="expanded",
)


# # --- Initialize session ---
def set_gh_form_submit():
    st.session_state["gh_form_submitted"] = True


if "gh_form_submitted" not in st.session_state:
    st.session_state["gh_form_submitted"] = False


def set_params_form_submit():
    st.session_state["params_form_submitted"] = True


if "params_form_submitted" not in st.session_state:
    st.session_state["params_form_submitted"] = False

# === Sidebar ===
st.sidebar.title("Greenhouse Location and Panel Orientation")

with st.sidebar.form(key="gh_form", border=False):
    latitude = st.slider(
        "Latitude of the location in degrees",
        -90.0,
        90.0,
        value=52.52,
    )
    longitude = st.slider(
        "Longitude of the location in degrees",
        -90.0,
        90.0,
        value=13.41,
        key="slider_ref_size",
    )
    tilt = st.multiselect(
        "Tilt of the solar panel in degrees",
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        default=[90, 40],
    )
    azimuth = st.multiselect(
        "Azimuth of the solar panel in degrees",
        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        default=["NE", "SE", "SW", "NW"],
    )

    tilt_, azimuth_ = zip(*[(t, a) for a in azimuth for t in tilt])
    tilt = list(tilt_)
    azimuth = list(azimuth_)

    submit_gh = st.form_submit_button("Validate", on_click=set_gh_form_submit)

if st.session_state.gh_form_submitted:
    st.sidebar.title("Control Parameters")

    with st.sidebar.form(key="params_form", border=False):
        sim_steps = st.slider(
            "Simulation steps (1/dt)",
            0,
            60 * 24,
            value=60,
            step=1,
        )
        lettuce_price = st.number_input(
            "Lettuce price (EUR/kg)",
            min_value=0.0,
            value=5.4,
            step=0.01,
            format="%.2f",
        )
        cultivated_area = st.number_input(
            "Greenhouse area (m^2)",
            min_value=0.0,
            value=187.5,
            step=0.5,
            format="%.1f",
        )
        N = st.number_input(
            "Number of control intervals",
            min_value=1,
            value=30,
            step=1,
        )
        dt = st.number_input(
            "Sampling time (seconds)",
            min_value=1,
            max_value=120,
            value=120,
            step=1,
        )
        x_ref_ = st.text_input(
            "Reference state (comma-separated values)",
            value="50.0, 5.0",
        )
        x_ref = np.array([float(x) for x in x_ref_.split(",")])
        u_min_ = st.text_input(
            "Minimum control input (comma-separated values)",
            value="0.0, 0.0",
        )
        u_min = [float(u) for u in u_min_.split(",")]
        u_max_ = st.text_input(
            "Maximum control input (comma-separated values)",
            value="100.0, 100.0",
        )
        u_max = [float(u) for u in u_max_.split(",")]

        submit_params = st.form_submit_button(
            "Run", on_click=set_params_form_submit
        )

# === Main ===
st.title("Economic MPC for Greenhouse Climate Control")

# === Enable after submitting parameters ===
if st.session_state.gh_form_submitted:
    # Initialize runtime
    openmeteo = OpenMeteo(
        latitude=latitude,  # Latitude of the location in degrees
        longitude=longitude,  # Longitude of the location in degrees
        tilt=tilt,  # Tilt angle of the surface in degrees
        azimuth=azimuth,  # Azimuth angle of the surface in degrees (South facing)
        frequency="minutely_15",  # Frequency of the data
    )

    start_date = pd.Timestamp.now() - pd.Timedelta(days=16)
    climdat = openmeteo.get_weather_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )

    climate = climdat.asfreq("1s").interpolate(method="time")
    start_date = pd.Timestamp(climate.index[0])
    runtime_info = st.info("Plotting forecast ...")

    forecast_plot = st.empty()
    forecast_plot.plotly_chart(climdat.plot(backend="plotly"))
    runtime_info.success("Skadoosh ...")

if (
    st.session_state.gh_form_submitted
    and st.session_state.params_form_submitted
):
    runtime_info.info("Preparing simulation ...")

    greenhouse_model = partial(gh_model, climate=climate.values)

    model = GreenHouseModel(climate_vars=climate.columns)

    mpc = EconomicMPC(
        model,
        climate,
        lettuce_price / 1000,
        cultivated_area,
        N,
        dt,
        x_ref,
        u_min,
        u_max,
    )

    simulator = GreenhouseSimulator(model, climate, dt)

    runtime_info.info("Simulating ...")

    # Find feasible initial state
    x0 = z
    u0 = np.array([0.0, 0.0])
    for k in range(N):
        k1 = greenhouse_model(k, x0, u0)
        k2 = greenhouse_model(k, x0 + dt / 2 * k1, u0)
        k3 = greenhouse_model(k, x0 + dt / 2 * k2, u0)
        k4 = greenhouse_model(k, x0 + dt * k3, u0)
        x_next = x0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x0 = x_next

    mpc.x0 = x0
    mpc.u0 = u0
    simulator.x0 = x0
    mpc.set_initial_guess()
    simulator.set_initial_guess()

    # Run the MPC simulation
    u0s = []
    y_nexts = []
    x0s = []
    ums = []
    for step in stqdm(range(sim_steps)):
        if step * dt + N + 1 > len(climate):
            if climate.index[-1] < pd.Timestamp.now():
                runtime_info.info("Fetching new forecast")
                start_date = start_date + pd.Timedelta(seconds=step * dt)
                climdat = openmeteo.get_weather_data(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=(start_date + pd.Timedelta(days=8)).strftime(
                        "%Y-%m-%d"
                    ),
                )

                climate = climdat.asfreq("1s").interpolate(method="time")
                mpc.climate = climate
                simulator.climate = climate
                forecast_plot.plotly_chart(climdat.plot(backend="plotly"))
                runtime_info.info("Simulating ...")
            else:
                break

        ums.append([u_min, u_max])
        with suppress_stdout():
            u0 = mpc.make_step(x0)
            u0s.append(u0)
            y_next = simulator.make_step(u0)
            y_nexts.append(y_next[-2:])
            x0 = y_next
            if np.isnan(x0).any():
                raise ValueError("x0 contains NaN values.")
            x0s.append(x0)

    runtime_info.info("Plotting results ...")
    timestamps = pd.date_range(
        start=start_date, periods=sim_steps, freq=pd.Timedelta(seconds=dt)
    )
    st.plotly_chart(plot(timestamps, y_nexts, u0s, ums))

    runtime_info.success(
        f"Congrats, your greenhouse generated profit of {np.sqrt(-np.mean(np.array(mpc.solver_stats["iterations"]["obj"]))):.2f} EUR! 🤑"
    )
