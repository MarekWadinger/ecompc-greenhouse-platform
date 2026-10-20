#!/usr/bin/env python
"""
Greenhouse Simulation Runner using PyOmo-based MPC Controller.

This script runs a simulation of a greenhouse control system using
economic MPC formulated with PyOmo.
"""

import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.greenhouse_model import GreenHouse, x_init
from core.pyomo_controller import (
    PyomoEconomicMPC,
    PyomoGreenHouseModel,
    PyomoGreenhouseSimulator,
)

# Define x_init_dict directly if not available from import
x_init_dict = {
    "C_ww": 0,
    "C_w": 0,
    "C_g": 0,
    "C_z": 0,
    "C_d": 0,
    "C_f": 0,
    "C_s": 0,
    "X_wsSt": 0,
    "X_wsSu": 0,
}


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def main():
    """Run the main greenhouse simulation with PyOmo controller."""
    # Time step
    dt = 120  # seconds

    # Initialize greenhouse model
    gh_model = GreenHouse(dt=dt)

    # Parameters
    cultivated_area = gh_model.A_c
    lettuce_price = 0.0054  # EUR/g
    N = 30  # number of control intervals

    # Paths
    filepath = "examples/"
    start_date = pd.Timestamp(2024, 10, 11, 14, 11)
    end_date = start_date + pd.Timedelta(days=15)
    date_tag = f"{start_date.strftime('%Y-%m-%d')}_{(start_date + pd.Timedelta(days=15)).strftime('%Y-%m-%d')}-{dt}s"

    # Load or generate climate data
    filename = f"{filepath}data/weather-{date_tag}.csv"

    if os.path.exists(filename):
        print(f"Loading climate data from {filename}")
        climate = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        print("Generating climate data...")
        # Create synthetic data for testing
        try:
            # Try importing the OpenMeteo module
            from core.api_queries import (
                Entsoe,
                OpenMeteo,
                get_city_geocoding,
            )

            has_tools_io = True
        except ImportError:
            has_tools_io = False

        if has_tools_io:
            try:
                openmeteo = OpenMeteo(
                    latitude=52.52,  # Latitude of the location in degrees
                    longitude=13.41,  # Longitude of the location in degrees
                    altitude=157,
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
                    frequency="minutely_15",  # Frequency of the data
                )

                (
                    city,
                    country,
                    country_code,
                    tz,
                    latitude,
                    longitude,
                    altitude,
                ) = get_city_geocoding("Bratislava")
                climate = (
                    openmeteo.get_weather_data(
                        start_date=start_date,
                        end_date=end_date,
                    )
                    .tz_localize(tz, ambiguous=True)
                    .asfreq(f"{dt}s")
                    .interpolate(method="time")
                )
                entsoe = Entsoe()
                energy_cost = entsoe.get_electricity_price(
                    country_code=country_code,
                    start_date=start_date.tz_localize(tz),
                    end_date=end_date.tz_localize(tz),
                    tz=tz,
                )
                climate["energy_cost"] = (
                    pd.concat([pd.Series(index=climate.index), energy_cost])
                    .sort_index()
                    .interpolate(method="time", limit_direction="both")
                )
            except Exception as e:
                print(f"Error using OpenMeteo: {e}")
                has_tools_io = False

        if not has_tools_io:
            # Create synthetic data for testing
            print("Creating synthetic climate data")
            hours = int((end_date - start_date).total_seconds() / 3600)
            timestamps = pd.date_range(
                start=start_date, periods=hours * 3600 // dt, freq=f"{dt}s"
            )

            # Create synthetic climate data with typical variations
            climate = pd.DataFrame(index=timestamps)

            # Convert timestamps to list of hours for calculations
            hours_of_day = [ts.hour for ts in timestamps]

            # Temperature: varies between 10-30°C with daily cycle
            climate["T_out"] = 20 + 10 * np.sin(
                np.pi * (np.array(hours_of_day) - 12) / 12
            )

            # Solar radiation: varies between 0-1000 W/m2 with daily cycle
            climate["R_sun"] = 500 * np.maximum(
                0, np.sin(np.pi * (np.array(hours_of_day) - 6) / 12)
            )

            # Add other necessary climate variables
            climate["CO2_out"] = 400  # Constant CO2 concentration
            climate["RH_out"] = 60 + 20 * np.sin(
                np.pi * np.array(hours_of_day) / 12
            )  # Relative humidity
            climate["v_wind"] = (
                2 + np.random.rand(len(climate)) * 3
            )  # Wind speed

            # Energy cost - varies by time of day
            climate["energy_cost"] = 0.1 + 0.05 * np.sin(
                np.pi * (np.array(hours_of_day) - 12) / 12
            )

        # Save climate data
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        climate.to_csv(filename)
        print(f"Climate data saved to {filename}")

    # Control bounds
    x0 = x_init.copy()
    u_min = [0.0] * len(gh_model.active_actuators)
    u_max = [100.0] * len(gh_model.active_actuators)

    # === Model ===================================================================
    print("Initializing greenhouse model...")
    # Convert climate.columns to list for the model
    climate_vars = list(climate.columns)
    model = PyomoGreenHouseModel(gh_model, climate_vars=climate_vars)

    # Find feasible initial state
    x0 = x_init.copy()
    u0 = np.array([100.0] * len(gh_model.active_actuators))
    tvp0 = tuple(climate.values[0])
    x0 = model.init_states(x0, u0, tvp0)

    # === Control =================================================================
    print("Setting up MPC controller...")
    mpc = PyomoEconomicMPC(
        model,
        climate,
        N,
        u_min=u_min,
        u_max=u_max,
    )

    # === Simulator ===============================================================
    print("Setting up simulator...")
    simulator = PyomoGreenhouseSimulator(model, climate)

    # === Runtime =================================================================
    # Start with a smaller simulation to test
    sim_steps = min(
        len(climate), 10
    )  # Limiting to 10 steps for initial testing
    print(f"Running simulation for {sim_steps} steps...")

    mpc.x0 = x0
    simulator.x0 = x0

    # Run the MPC simulation
    U = pd.DataFrame(
        columns=[
            act
            for act, active in gh_model.active_actuators.items()
            if active == 1
        ],
        index=range(sim_steps),
    )
    X = pd.DataFrame(columns=[*x_init_dict.keys()], index=range(sim_steps))

    for step in tqdm(range(sim_steps)):
        if step * dt + N + 1 > len(climate):
            if step + N == len(climate):
                break

        try:
            # Don't suppress stdout during debugging
            # with suppress_stdout():
            u0 = mpc.make_step(x0)
            U.iloc[step] = u0
            x0 = simulator.make_step(u0)
            X.iloc[step] = x0

            if np.isnan(x0).any():
                print(
                    "Warning: NaN values detected in state, stopping simulation"
                )
                break
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break

    # Generate timestamps
    timestamps = pd.date_range(
        start=start_date, periods=len(X), freq=pd.Timedelta(seconds=dt)
    )

    # Add timestamps to dataframes
    X.index = timestamps
    U.index = timestamps

    # Calculate profit and costs
    print("Calculating profit and costs...")
    profit_costs = model.analyze_profit_and_costs(X, U, climate["energy_cost"])
    print(profit_costs)

    # Save results
    results_dir = f"{filepath}results/"
    os.makedirs(results_dir, exist_ok=True)

    X.to_csv(f"{results_dir}states_pyomo_{date_tag}.csv")
    U.to_csv(f"{results_dir}controls_pyomo_{date_tag}.csv")
    profit_costs.to_csv(f"{results_dir}profit_costs_pyomo_{date_tag}.csv")

    print(f"Results saved to {results_dir}")

    print("Simulation complete!")


if __name__ == "__main__":
    main()
