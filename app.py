import os
import sys
from contextlib import contextmanager
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm
from streamlit_theme import st_theme

from core.api_queries import (
    ElectricityMap,
    Entsoe,
    OpenMeteo,
    get_city_geocoding,
    get_weather_and_energy_data,
)
from core.controller import EconomicMPC, GreenHouseModel, GreenhouseSimulator
from core.greenhouse_model import GreenHouse, x_init_dict
from core.lettuce_model import DRY_TO_WET_RATIO, RATIO_SDW_NSDW
from core.plot import plotly_greenhouse, plotly_response, plotly_weather

# --- Page Config ---

st.set_page_config(
    page_title="Green House Control",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# --- Constants ---
sim_steps_max = 60 * 24
N_max = 60
Ts_default = 300

TTL = 5 * 60  # Cache for 5 minutes


# --- Functions ---


# # --- Initialize session ---
def set_shape_form_submit():
    st.session_state["shape_form_submitted"] = True
    st.session_state["location_form_submitted"] = False
    st.session_state["params_form_submitted"] = False


if "shape_form_submitted" not in st.session_state:
    st.session_state["shape_form_submitted"] = False


def set_location_form_submit():
    st.session_state["location_form_submitted"] = True
    st.session_state["params_form_submitted"] = False


if "location_form_submitted" not in st.session_state:
    st.session_state["location_form_submitted"] = False


def set_params_form_submit():
    st.session_state["params_form_submitted"] = True


if "params_form_submitted" not in st.session_state:
    st.session_state["params_form_submitted"] = False

#  === Sidebar ===
with st.sidebar:
    theme: dict | None = st_theme()
    if theme is not None and theme.get("base", "light") == "dark":
        st.image("app/qr-white_transparent.png")
    else:
        st.image("app/qr-black_transparent.png")

    st.title("Greenhouse Designer")
    st.markdown("Design new greenhouse or create digital twin of your own.")

    st.header("Shape")
    with st.expander(
        "Customize",
        icon="🏠",
        expanded=not st.session_state.shape_form_submitted,
    ):
        with st.form(key="shape_form", border=False):
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

            st.header("Orientation")

            # Compass azimuth selection (slider from 0° to 360°)
            azimuth_face = st.slider(
                "Azimuth (degrees - greenhouse width faces)",
                min_value=0,
                max_value=360,
                value=90,  # Default to South
                format="%d°",
                help="Direction your greenhouse faces: North = 0°, South = 180°",
            )

            submit_gh_shape = st.form_submit_button(
                "Build Greenhouse", on_click=set_shape_form_submit
            )

    if st.session_state.shape_form_submitted:
        st.header(
            "Location",
            help="Weather forecast and geolocation data provided by [Open-Meteo](https://open-meteo.com).",
        )
        st.markdown("Fetch weather forecast for your location.")
        with st.expander(
            "Customize",
            icon="📍",
            expanded=not st.session_state.location_form_submitted,
        ):
            with st.form(key="location_form", border=False):
                city_ = st.text_input("City", "Bratislava")

                grid = st.columns([1.0, 1.0])
                start_date_ = grid[0].date_input(
                    "Date",
                    min_value=pd.Timestamp(year=1940, month=1, day=1),
                    max_value=pd.Timestamp.now() + pd.Timedelta(days=7),
                )
                start_time = grid[1].time_input(
                    "Time",
                )

                submit_gh = st.form_submit_button(
                    "Fetch Forecast", on_click=set_location_form_submit
                )
                (
                    city,
                    country,
                    country_code,
                    tz,
                    latitude,
                    longitude,
                    altitude,
                ) = get_city_geocoding(city_)
                try:
                    co2_intensity = ElectricityMap().get_co2_intensity(
                        country_code,
                    )
                    co2_source = "Current"
                except ValueError:
                    co2_intensity = 200.0
                    co2_source = "Default"
        st.markdown(
            f"{co2_source} carbon intensity: **{co2_intensity} gCO₂/kWh**",
            help="Carbon intensity data provided by [ELECTRICITY MAPS](https://electricitymap.org)",
        )

    if st.session_state.location_form_submitted:
        gh_model = GreenHouse(
            length,
            width,
            height,
            roof_tilt,
            latitude=latitude,
            longitude=longitude,
            dt=Ts_default,
            **{"co2_intensity": co2_intensity},
        )

        st.header(
            "Climate Controls",
            help="Optimally scaled actuators for your greenhouse. But you're in control.",
        )
        with st.expander(
            "Customize",
            icon="🌡️",
            expanded=False,
        ):
            max_vent = st.slider(
                "Max. ventilation power (m³/s)",
                min_value=0.0,
                max_value=gh_model.fan.max_unit * 2,
                value=gh_model.fan.max_unit,
                step=1.0,
                format="%.0f",
            )
            max_heat = st.slider(
                "Max. heating power (W)",
                min_value=0.0,
                max_value=gh_model.heater.max_unit * 2,
                value=gh_model.heater.max_unit,
                step=1.0,
                format="%.0f",
            )
            max_hum = st.slider(
                "Max. humidifier power (l/h)",
                min_value=0.0,
                max_value=gh_model.humidifier.max_unit * 2,
                value=gh_model.humidifier.max_unit,
                step=1.0,
                format="%.0f",
            )
            max_co2 = st.slider(
                "Max. CO₂ generation (kg/h)",
                min_value=0.0,
                max_value=gh_model.co2generator.max_unit * 2,
                value=gh_model.co2generator.max_unit,
                step=1.0,
                format="%.0f",
            )

        st.title(
            "eMPC Design",
            help="Economic Model Predictive Control (eMPC) is a control strategy that optimizes the control inputs to minimize the cost of operation. In this case, we are optimizing the climate control of a greenhouse to maximize the profit from lettuce production.",
        )
        st.markdown(
            "Change parameters of optimal controller and watch your crop growing."
        )

        with st.expander(
            "Customize",
            icon="🌱",
            expanded=not st.session_state.params_form_submitted,
        ):
            with st.form(key="params_form", border=False):
                lettuce_price = st.number_input(
                    "Lettuce price (EUR/kg)",
                    min_value=0.0,
                    value=5.4,
                    step=0.01,
                    format="%.2f",
                    help="Changes in price affect profit margins, influencing production decisions and the feasibility of growing lettuce.",
                )
                x_lettuce_wet_init = st.number_input(
                    "Planted seeds weight (g/m²)",
                    min_value=5,
                    max_value=500,
                    value=500,
                    step=1,
                    help=(
                        "This parameter impacts the potential yield; more seeds can lead to higher biomass but also requires more resources.\n"
                        "1 seedling ~ 5g.\n"
                        "For mature lettuce and seedling, we assume that 10 % is dry weight.\n"
                        "Ratio of structural to non-structural dry weight is "
                        "assumed to be 3:7."
                    ),
                )
                Ts = st.slider(
                    "Sampling time (s)",
                    min_value=0,
                    max_value=300,
                    value=Ts_default,
                    step=10,
                    help="A shorter sampling time allows for more responsive control but increases computational load. A longer sampling time may lead to slower responses to changes.",
                )
                Ts = max(1, Ts)
                sim_steps = st.slider(
                    "Simulation steps (samples)",
                    min_value=0,
                    max_value=sim_steps_max,
                    value=290,
                    step=10,
                    help="More simulation steps provide a more detailed understanding of the future growth but require more computational resources.",
                )
                N = st.slider(
                    "Prediction horizon (samples)",
                    min_value=1,
                    max_value=N_max,
                    value=3,
                    step=1,
                    help="A longer prediction horizon enables better long-term planning but significantly increase complexity and uncertainty in predictions.",
                )
                x_lettuce_dry_init = (
                    x_lettuce_wet_init * DRY_TO_WET_RATIO
                )  # g/m²

                x_sn_init = x_lettuce_dry_init * RATIO_SDW_NSDW
                us = st.slider(
                    "Control input range (%)",
                    value=[0.0, 100.0],
                    help="Adjusting this range impacts how much control you have over the climate control.",
                )
                u_min, u_max = us

                submit_params = st.form_submit_button(
                    "Start Growing!", on_click=set_params_form_submit
                )

# === Main ===
st.title("Economic MPC for Greenhouse Climate Control")
runtime_info = st.empty()
# === Enable after submitting parameters ===
if st.session_state.shape_form_submitted:
    st.header("Greenhouse Visualization")
    fig_gh = plotly_greenhouse(length, width, height, roof_tilt, azimuth_face)
    st.plotly_chart(fig_gh)


if (
    st.session_state.shape_form_submitted
    and st.session_state.location_form_submitted
):
    st.header(f"Weather Forecast for {city} ({country})")
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
        altitude=altitude,
        tilt=tilt,  # Tilt angle of the surface in degrees
        azimuth=azimuth,  # Azimuth angle of the surface in degrees (South facing)
        frequency="minutely_15",  # Frequency of the data
    )
    start_date = pd.Timestamp.combine(
        start_date_,  # type: ignore
        start_time,
    )
    end_date = start_date + pd.Timedelta(
        days=min(15, (sim_steps_max + N_max) * Ts // (3600 * 24))
    )
    climate = get_weather_and_energy_data(
        openmeteo,
        Entsoe(),
        start_date,
        end_date,
        tz,
        country_code,
        Ts,
    )
    runtime_info.info("Plotting forecast ...")

    weather_plot = st.empty()
    weather_plot.plotly_chart(plotly_weather(climate))
    runtime_info.success(
        "Forecast fetched from [Open-Meteo](https://open-meteo.com)."
    )

if (
    st.session_state.shape_form_submitted
    and st.session_state.location_form_submitted
    and st.session_state.params_form_submitted
):
    st.header("Simulation Results")
    runtime_info.info("Preparing simulation ...")

    # Overwrite by user specification
    gh_model.fan.max_unit = max_vent
    gh_model.heater.max_unit = max_heat
    gh_model.humidifier.max_unit = max_hum
    gh_model.co2generator.max_unit = max_co2
    gh_model.dt = Ts
    greenhouse_model = partial(gh_model.model, climate=climate.values)

    model = GreenHouseModel(
        gh_model,
        climate_vars=climate.columns,
        lettuce_price=lettuce_price / 1000,
    )

    mpc = EconomicMPC(
        model,
        climate,
        N,
        x_sn_init,
        u_min,
        u_max,
    )

    simulator = GreenhouseSimulator(model, climate, x_sn_init)

    runtime_info.info("Simulating ...")

    # Find feasible initial state for given climate
    x0 = np.array([*x_init_dict.values()])
    x0[-2:] = x_sn_init
    u0 = np.array([50.0] * len(gh_model.active_actuators))
    for k in range(N):
        k1 = greenhouse_model(k, x0, u0)
        k2 = greenhouse_model(k, x0 + Ts / 2 * k1, u0)
        k3 = greenhouse_model(k, x0 + Ts / 2 * k2, u0)
        k4 = greenhouse_model(k, x0 + Ts * k3, u0)
        x_next = x0 + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x0 = x_next

    mpc.x0 = x0
    mpc.u0 = np.array([50.0] * model.n_u)
    simulator.x0 = x0
    mpc.set_initial_guess()
    simulator.set_initial_guess()

    # Run the MPC simulation
    u0s = pd.DataFrame(
        columns=[
            act
            for act, active in gh_model.active_actuators.items()
            if active == 1
        ],
        index=range(sim_steps),
    )
    x0s = pd.DataFrame(columns=[*x_init_dict.keys()], index=range(sim_steps))
    for step in stqdm(range(sim_steps)):
        if step * Ts + N + 1 > len(climate):
            if step + N == len(climate):
                runtime_info.info("Fetching new forecast")
                start_date = start_date + pd.Timedelta(seconds=step * Ts)
                climate = get_weather_and_energy_data(
                    OpenMeteo(
                        latitude=latitude,
                        longitude=longitude,
                        altitude=altitude,
                        tilt=tilt,
                        azimuth=azimuth,
                        frequency="minutely_15",
                    ),
                    Entsoe(),
                    start_date,
                    start_date + pd.Timedelta(days=1),
                    tz,
                    country_code,
                    Ts,
                )

                mpc.climate = climate
                simulator.climate = climate
                weather_plot.plotly_chart(
                    climate.resample("1h").median().plot(backend="plotly")
                )
                runtime_info.info("Simulating ...")

        with suppress_stdout():
            u0 = mpc.make_step(x0)
            u0s.iloc[step] = u0.flatten()
            x0 = simulator.make_step(u0)
            if np.isnan(x0).any():
                runtime_info.error("x0 contains NaN values.")
                break
            x0s.iloc[step] = x0.flatten()

    runtime_info.info("Plotting results ...")
    timestamps = pd.date_range(
        start=start_date, periods=sim_steps, freq=pd.Timedelta(seconds=Ts)
    )
    forecast_plot = st.empty()
    forecast_plot.plotly_chart(
        plotly_response(timestamps, x0s, u0s, [u_min], [u_max])
    )

    # Export results to table
    profit_costs = model.analyze_profit_and_costs(
        x0.flatten()[-2:] - x_sn_init,
        u0s,
        climate["energy_cost"].values[: len(u0s)],
    )

    st.table(profit_costs.to_frame().style.format("{:.2f}"))

    if profit_costs["Total"] < 0:
        runtime_info.error(
            f"Unfortunately, your greenhouse generated a loss of {profit_costs['Total']:.2f} EUR. 😢"
        )
    else:
        runtime_info.success(
            f"Congrats, your greenhouse generated profit of {profit_costs['Total']:.2f} EUR! 🤑"
        )
