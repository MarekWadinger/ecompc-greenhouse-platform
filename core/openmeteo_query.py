import warnings
from datetime import datetime
from typing import Literal, Sequence

import pandas as pd
import pvlib
import requests_cache
from retry_requests import retry


def azimuth_to_deg(azimuth: str):
    directions = {
        "N": 0,
        "NE": -45,
        "E": -90,
        "SE": -135,
        "S": 180,
        "SW": 135,
        "W": 90,
        "NW": 45,
    }

    return directions.get(azimuth.upper(), None)


def get_openmeteo(
    latitude: float,
    longitude: float,
    frequency: Literal["hourly", "minutely_15", "current"] = "current",
    forecast: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    if frequency != "current":
        if forecast is not None and (
            start_date is not None or end_date is not None
        ):
            raise ValueError(
                "Either forecast or start_date and end_date must be provided"
            )

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        frequency: [
            "temperature_2m",
            "apparent_temperature",
            "wind_speed_10m",
            "relative_humidity_2m",
            # "shortwave_radiation",
            # "diffuse_radiation",
            # "direct_normal_irradiance",
            # "global_tilted_irradiance",
        ],
        "forecast_days": forecast,
        "start_date": start_date,
        "end_date": end_date,
    }
    responses = retry_session.get(url, params=params)

    if responses.ok:
        r = responses.json()
        # Convert the response to a pandas DataFrame
        data = r.pop(frequency)
        if "interval" in data:
            del data["interval"]
        if isinstance(data["time"], list):
            idx = data.pop("time")
        else:
            idx = [data.pop("time")]
        df = pd.DataFrame(data, index=idx)

        # Add units to column names
        suffix_dict = r[f"{frequency}_units"]
        for col in df.columns:
            if col in suffix_dict:
                df.rename(
                    columns={col: f"{col} [{suffix_dict[col]}]"}, inplace=True
                )
        return df, r
    else:
        raise ValueError(
            f"Failed to fetch data from Open-Meteo API: {responses.text}"
        )


def get_irr_at_tilt_and_azimuth(
    latitude: float,
    longitude: float,
    altitude: int,
    tilt: float | int | Sequence[float | int],
    azimuth: float | int | str | Sequence[float | int | str],
    frequency: Literal["hourly", "minutely_15", "current"] = "current",
    forecast: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    dni: float | None = None,
    ghi: float | None = None,
    dhi: float | None = None,
    clearsky_model: str = "ineichen",
):
    """
    Compute radiation at a given tilt and azimuth angle from Global Horizontal Irradiance (GHI).

    Parameters:
    ghi: Global Horizontal Irradiance in W/m^2.
    latitude: Latitude of the location in degrees.
    longitude: Longitude of the location in degrees.
    altitude: Altitude of the location in meters above sea level.
    tilt: Tilt angle of the surface in degrees.
    azimuth: Azimuth angle of the surface in degrees.
    date_time: Date and time at which to compute the radiation.
    dni: Direct Normal Irradiance in W/m^2.
    ghi: Global Horizontal Irradiance in W/m^2.
    dhi: Diffuse Horizontal Irradiance in W/m^2.
    clearsky_model: The clear sky model to be used. Default is 'ineichen'.

    Returns:
    Radiation on the tilted surface in W/m^2.
    """
    if frequency != "current":
        if forecast is not None and (
            start_date is not None or end_date is not None
        ):
            raise ValueError(
                "Either forecast or start_date and end_date must be provided"
            )
    elif (
        forecast is not None
        and start_date is not None
        and end_date is not None
    ):
        warnings.warn(
            "When frequency is 'current' set forecast, start_date, and end_date to None for efficiency."
        )

    now = datetime.utcnow()
    frequency_mapping = {"hourly": "H", "minutely_15": "15T", "current": "H"}
    freq = frequency_mapping[frequency]
    if frequency == "current":
        date_time = pd.date_range(start=now, end=now, freq=freq)
    elif forecast is not None:
        now = pd.Timestamp.now().normalize()
        end = now + pd.Timedelta(f"{forecast}D")
        date_time = pd.date_range(
            start=now, end=end, freq=freq, inclusive="left"
        )
    else:
        # To align with the Open-Meteo API
        if end_date is not None:
            end_date_ = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        date_time = pd.date_range(
            start=start_date, end=end_date_, freq=freq, inclusive="left"
        )

    # Define location
    location = pvlib.location.Location(
        latitude=latitude, longitude=longitude, altitude=altitude
    )

    # Solar position
    solar_position = location.get_solarposition(date_time)

    # Clear sky model
    if dni is None and ghi is None and dhi is None:
        clearsky = location.get_clearsky(date_time, model=clearsky_model)
        dni_ = clearsky["dni"]
        ghi_ = clearsky["ghi"]
        dhi_ = clearsky["dhi"]
    else:
        dni_ = dni
        ghi_ = ghi
        dhi_ = dhi

    # Tilted surface radiation
    if not isinstance(tilt, Sequence):
        tilt = [tilt]
    if not isinstance(azimuth, Sequence):
        azimuth = [azimuth]
    df_rad = pd.DataFrame()
    for t, a in zip(tilt, azimuth):
        if not isinstance(a, (int, float)):
            a = azimuth_to_deg(a)
        radiation = pvlib.irradiance.get_total_irradiance(
            surface_tilt=t,
            surface_azimuth=a,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=dni_,
            ghi=ghi_,
            dhi=dhi_,
        )
        if isinstance(radiation, (pd.Series, pd.DataFrame)):
            radiation = radiation.drop(
                columns=["poa_global", "poa_sky_diffuse", "poa_ground_diffuse"]
            )
            radiation.loc[:, ~radiation.columns.str.contains("poa_global")]
            df_rad = pd.concat(
                [df_rad, radiation.add_suffix(f" [W/m²]:t{t}a{a}")],
                axis=1,
            )
    return df_rad


def sort_diffuse_last(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    cols_sort = []
    cols_diffuse = []
    for c in cols:
        if "diffuse" in c:
            cols_diffuse.append(c)
        else:
            cols_sort.append(c)
    cols_sort.extend(cols_diffuse)
    df = df[cols_sort]
    return df


def get_weather_data(
    latitude: float,
    longitude: float,
    tilt: float | int | Sequence[float | int],
    azimuth: float | int | str | Sequence[float | int | str],
    frequency: Literal["hourly", "minutely_15", "current"] = "current",
    forecast: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    df, meta = get_openmeteo(
        latitude, longitude, frequency, forecast, start_date, end_date
    )

    df_rad = get_irr_at_tilt_and_azimuth(
        latitude,
        longitude,
        meta["elevation"],
        tilt,
        azimuth,
        frequency,
        forecast,
        start_date,
        end_date,
    )

    df = pd.concat(
        [df.reset_index(drop=True), df_rad.reset_index()], axis=1
    ).set_index("index")

    df = df.dropna()
    df = sort_diffuse_last(df)
    return df


if __name__ == "__main__":
    # Example usage
    latitude = 52.52  # Latitude of the location in degrees
    longitude = 13.41  # Longitude of the location in degrees
    tilt = [
        90,
        40,
        90,
        40,
        90,
        40,
        90,
        40,
    ]  # Tilt angle of the surface in degrees
    azimuth = [
        "NE",
        "NE",
        "SE",
        "SE",
        "SW",
        "SW",
        "NW",
        "NW",
    ]  # Azimuth angle of the surface in degrees (South facing)
    frequency: Literal["hourly", "minutely_15", "current"] = "hourly"
    forecast: int = 2
    start_date = None
    end_date = None

    df = get_weather_data(
        latitude,
        longitude,
        tilt,
        azimuth,
        frequency,
        forecast,
        start_date,
        end_date,
    )
