import os
import sys
from contextlib import contextmanager
from functools import partial
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm
from streamlit_theme import st_theme

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.controller import EconomicMPC, GreenHouseModel, GreenhouseSimulator
from core.greenhouse_model import GreenHouse, x_init
from core.openmeteo_query import OpenMeteo, get_city_geocoding
from core.plot import plotly_greenhouse, plotly_response


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
def plotly_greenhouse_(length, width, height, roof_tilt, azimuth):
    return plotly_greenhouse(length, width, height, roof_tilt, azimuth)


@st.cache_data
def plot_(
    _timestamps,
    y_nexts,
    u0s,
    ums,
):
    return plotly_response(_timestamps, y_nexts, u0s, ums)


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


def set_gh_shape_form_submit():
    st.session_state["gh_shape_form_submitted"] = True


if "gh_shape_form_submitted" not in st.session_state:
    st.session_state["gh_shape_form_submitted"] = False


def set_params_form_submit():
    st.session_state["params_form_submitted"] = True


if "params_form_submitted" not in st.session_state:
    st.session_state["params_form_submitted"] = False

#  === Sidebar ===
theme: dict | None = st_theme()
if theme is not None and theme.get("base", "light") == "dark":
    st.sidebar.image(
        "/Users/mw/pyprojects/dynamic_opt_growth_model/app/qr-white_transparent.png"
    )
else:
    st.sidebar.image(
        "/Users/mw/pyprojects/dynamic_opt_growth_model/app/qr-black_transparent.png"
    )
st.sidebar.title("Greenhouse Shape and Orientation")

with st.sidebar.form(key="gh_shape_form", border=False):
    # Input for length and width
    length = st.number_input(
        "Length (meters)",
        min_value=0.0,
        value=25.0,
        step=0.1,
        format="%.1f",
    )

    width = st.number_input(
        "Width (meters)",
        min_value=0.0,
        value=10.0,
        step=0.1,
        format="%.1f",
    )

    height = st.number_input(
        "Wall Height (meters)",
        min_value=1.0,
        value=4.0,
        step=0.1,
        format="%.1f",
    )

    st.markdown("### Orientation and Roof Tilt")

    # Compass azimuth selection (slider from 0° to 360°)
    azimuth_face = st.slider(
        "Azimuth (degrees - greenhouse width faces)",
        min_value=0,
        max_value=360,
        value=90,  # Default to South
        format="%d°",
        help="Direction your greenhouse faces: North = 0°, South = 180°",
    )

    # Tilt for roof (0° to 90°)
    roof_tilt = st.slider(
        "Roof Tilt (degrees)",
        min_value=0,
        max_value=45,
        value=30,
        step=1,
        format="%d°",
        help="Tilt angle of the roof: 0° = flat, 90° = vertical",
    )
    wall_tilt = 90

    submit_gh_shape = st.form_submit_button(
        "Validate", on_click=set_gh_shape_form_submit
    )


if st.session_state.gh_shape_form_submitted:
    st.sidebar.title("Greenhouse Location")

    with st.sidebar.form(key="gh_form", border=False):
        city_ = st.text_input("City", "Bratislava")
        city, country, latitude, longitude, altitude = get_city_geocoding(
            city_
        )
        # latitude = st.slider(
        #     "Latitude of the location in degrees",
        #     -90.0,
        #     90.0,
        #     value=52.52,
        # )
        # longitude = st.slider(
        #     "Longitude of the location in degrees",
        #     -90.0,
        #     90.0,
        #     value=13.41,
        #     key="slider_ref_size",
        # )

        submit_gh = st.form_submit_button(
            "Validate", on_click=set_gh_form_submit
        )

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
        N = st.number_input(
            "Number of control intervals",
            min_value=1,
            value=30,
            step=1,
        )
        dt = st.number_input(
            "Sampling time (seconds)",
            min_value=1,
            max_value=180,
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
runtime_info = st.empty()
# === Enable after submitting parameters ===
if st.session_state.gh_shape_form_submitted:
    st.header("Greenhouse Visualization")
    fig_gh = plotly_greenhouse_(length, width, height, roof_tilt, azimuth_face)
    st.plotly_chart(fig_gh)


if (
    st.session_state.gh_shape_form_submitted
    and st.session_state.gh_form_submitted
):
    st.header("Weather Forecast for {city} ({country})")
    tilt = [90, 90, 90, 90, 89, 89, roof_tilt, roof_tilt]
    azimuth: list[int | str] = [
        azimuth_face,  # Front
        azimuth_face + 180,  # Back
        azimuth_face + 90,  # Right
        azimuth_face + 270,  # Left
    ] * 2
    # Initialize runtime
    openmeteo = OpenMeteo(
        latitude=latitude,  # Latitude of the location in degrees
        longitude=longitude,  # Longitude of the location in degrees
        tilt=tilt,  # Tilt angle of the surface in degrees
        azimuth=azimuth,  # Azimuth angle of the surface in degrees (South facing)
        frequency="minutely_15",  # Frequency of the data
    )

    start_date = pd.Timestamp.now() - pd.Timedelta(days=1)
    climate = (
        openmeteo.get_weather_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        .asfreq(f"{dt}s")
        .interpolate(method="time")
    )
    start_date = pd.Timestamp(climate.index[0])
    runtime_info.info("Plotting forecast ...")

    weather_plot = st.empty()
    weather_plot.plotly_chart(
        climate.resample("1H").median().plot(backend="plotly")
    )
    runtime_info.success("Skadoosh ...")

if (
    st.session_state.gh_shape_form_submitted
    and st.session_state.gh_form_submitted
    and st.session_state.params_form_submitted
):
    st.header("Simulation Results")
    runtime_info.info("Preparing simulation ...")

    gh_model = GreenHouse(
        length,
        width,
        height,
        roof_tilt,
        latitude=latitude,
        longitude=longitude,
        dt=dt,
    )
    greenhouse_model = partial(gh_model.model, climate=climate.values)

    model = GreenHouseModel(gh_model, climate_vars=climate.columns)

    mpc = EconomicMPC(
        model,
        climate,
        lettuce_price / 1000,
        N,
        dt,
        x_ref,
        u_min,
        u_max,
    )

    simulator = GreenhouseSimulator(model, climate, dt)

    runtime_info.info("Simulating ...")

    # Find feasible initial state
    x0 = x_init
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
                climate = (
                    openmeteo.get_weather_data(
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=(start_date + pd.Timedelta(days=8)).strftime(
                            "%Y-%m-%d"
                        ),
                    )
                    .asfreq(f"{dt}s")
                    .interpolate(method="time")
                )

                mpc.climate = climate
                simulator.climate = climate
                weather_plot.plotly_chart(
                    climate.resample("1H").median().plot(backend="plotly")
                )
                runtime_info.info("Simulating ...")

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
    forecast_plot = st.empty()
    forecast_plot.plotly_chart(plotly_response(timestamps, x0s, u0s, ums))

    runtime_info.success(
        f"Congrats, your greenhouse generated profit of {np.sqrt(-np.array(mpc.solver_stats['iterations']['obj'][-1])):.2f} EUR! 🤑"
    )
