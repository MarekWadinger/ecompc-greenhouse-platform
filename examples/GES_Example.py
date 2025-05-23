#######################################################################################################################################
## GES: Greenhouse Energy Simulation

# This python3 code simulates heat, mass and CO$_2$ exchange in an unheated, ventilated single-zone greenhouse for a simple test case.
# The heat transfer processes simulated include convection, conduction and radiation, together with plant transpiration for simulation
# of heat and mass exchange due to evaporation of water from the leaf surface.
# Simple models of photosynthesis and crop growth for tomatoes are included in order to simulate CO_2 exchange.
# The model is based on the GDGCM (Pieters, J. and Deltour, J., 'The Gembloux Dynamic Greenhouse Climate Model - GDGCM', 2000)
# and on the thesis by Vanthoor (Vanthoor, B.H.,'A model-based greenhouse design method', PhD Thesis, Wageningen University, 2011).

# The original code was written in MATLAB and can be found on GitHub at https://github.com/EECi/GES.

# The files required include:

# greenhouse_model.py
#   contains generic functions for convection, radiation and conduction calculations, climate data interpolation and calculation of
#   relative humidity
#   contains parameter values for fundamental constants, greenhouse construction and operation and plant geometry and growth.
# SampleWeather.csv
#   Hourly input weather data, in the format 'Hour, Ambient Temperature ($^o$C), Sky Temperature ($^o$C), Windspeed (m/s),
#   Relative Humidity (%), Direct Solar Radiation (order NE wall, NE Roof, SE Wall, SE Roof, SW Wall, SW Roof, NW Wall, NW Roof)
#   (W/m$^2$), Diffuse Solar Radiation (order as for Direct)(W/m$^2$)

#######################################################################################################################################

import os
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(1, str(Path().resolve()))
from core.api_queries import OpenMeteo  # noqa: E402
from core.greenhouse_model import (  # noqa: E402
    GreenHouse,
    M_c,
    R,
    T_k,
    atm,
    x_init,
)

results_dir = "examples/results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

gh_model = GreenHouse()

openmeteo = OpenMeteo(
    latitude=52.52,  # Latitude of the location in degrees
    longitude=13.41,  # Longitude of the location in degrees
    altitude=None,
    tilt=[
        90,
        40,
        90,
        40,
        90,
        40,
        90,
        40,
    ],  # Tilt angle of the surface in degrees
    azimuth=[
        "NE",
        "NE",
        "SE",
        "SE",
        "SW",
        "SW",
        "NW",
        "NW",
    ],  # Azimuth angle of the surface in degrees (South facing)
    frequency="hourly",
)
forecast: int = 3

climdat = openmeteo.get_weather_data(forecast)

climate = climdat.asfreq("1s").interpolate(method="time")

if __name__ == "__main__":
    ## Simulate over time

    tic = time.time()

    sim_days = 2  # Number of days of simulation
    tf = 86400 * sim_days  # Time in seconds
    t = [0, tf]
    tval = np.linspace(0, tf, tf + 1)

    # Use solve_ivp with 'BDF' stiff solver to solve the ODEs
    perc_vent = 100.0
    perc_heater = 100.0
    params = [(perc_vent, perc_heater)]

    # TODO: FIX: for some reason, the simulation requires longer weather forecast than the actual simulation time
    output = solve_ivp(
        partial(gh_model.model, climate=climate.values),
        t,
        x_init,
        method="BDF",
        t_eval=tval,
        rtol=1e-5,
        args=params,
    )

    # Time simulation and print time taken
    toc = time.time()

    xt = toc - tic

    print("Runtime(s) = ", f"{xt:.3}")

    ## Plot results

    Tout_i = np.transpose(output.y[1, :] - T_k)  # [°C]
    Ccout = np.transpose(output.y[-3, :])

    ## Temperatures

    time = output.t / (3600 * 24)  # Time in days
    dpi = 1200

    ## Weather data
    fig, ax = plt.subplots()
    cond = climdat.columns.str.contains("poa_direct|poa_diffuse")
    clim_raw = climdat.loc[:, ~cond]
    clim_stat = climdat.loc[:, cond]
    aggregations = ["min", "max"]
    clim_stat = (
        clim_stat.T.groupby(clim_stat.columns.str.split(":").str[0])
        .agg(aggregations)  # type: ignore
        .T
    )
    clim_stat = clim_stat.unstack()
    clim_stat.columns = [
        "_".join(col).strip() for col in clim_stat.columns.values
    ]
    clim_raw.plot(ax=ax)
    clim_stat.plot(ax=ax)
    fig.savefig("examples/results/weather.png", format="png", dpi=dpi)

    ## Internal air
    fig, ax = plt.subplots()
    ax.plot(time, Tout_i, color="b", label="Internal air")
    ax.set_title("Internal air and vegetation temperatures")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Temperature ($^o$C)")
    fig.savefig("examples/results/temp.png", format="png", dpi=dpi)

    ## CO_2
    Ccout_ppm = Ccout * R * (Tout_i + T_k) / (M_c * atm) * 1.0e6
    fig, ax = plt.subplots()
    ax.plot(time, Ccout_ppm, color="b")
    ax.set_title("CO$_2$")
    ax.set_xlabel("Day")
    ax.set_ylabel("CO$_2$ (ppm)")
    fig.savefig("examples/results/co2.png", format="png", dpi=dpi)

    ## Salaattia
    dx_sdw_dt = np.transpose(output.y[-2, :])

    dx_nsdw_dt = np.transpose(output.y[-1, :])

    fig, ax = plt.subplots()
    ax.plot(time, dx_sdw_dt, color="b", label="Structural Dry Weight")
    ax.plot(time, dx_nsdw_dt, color="g", label="Non-Structural Dry Weight")
    ax.set_title("Plant Dry Weight")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Rate [g.m-2]")
    ax.legend()
    fig.savefig("examples/results/salat.png", format="png", dpi=dpi)
