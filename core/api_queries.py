import os
import warnings
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import pvlib
import requests
import requests_cache
from dotenv import load_dotenv
from retry_requests import retry

load_dotenv()


def get_co2_intensity(
    country_code: str, api_key: str | None = None, default: float = 200.0
) -> float:
    """This endpoint retrieves the last 24 hours of carbon intensity (in gCO2eq/kWh) of an area. It can either be queried by zone identifier or by geolocation. The resolution is 60 minutes."""

    if api_key is None:
        api_key = os.getenv("ELECTRICITYMAP_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key must be provided or set in the .env file"
            )

    url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={country_code}"
    headers = {"auth-token": api_key}
    response = requests.get(url, headers=headers)
    if response.ok:
        data = response.json()
        return data["carbonIntensity"]
    else:
        warnings.warn(
            f"Failed to fetch CO2 intensity data. Using {default} gCO₂eq/kWh. Details:\n{response.text}"
        )
        return default


class ElectricityMap:
    def __init__(
        self,
        api_key: str | None = None,
        default: float = 200.0,
    ):
        if api_key is None:
            api_key = os.getenv("ELECTRICITYMAP_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided or set in the .env file"
                )
        self.default = default

        cache_session = requests_cache.CachedSession(
            ".elmapsapi.cache", expire_after=3600
        )
        self.retry_session = retry(
            cache_session, retries=5, backoff_factor=0.2
        )

        headers = {"auth-token": api_key}
        self.retry_session.headers.update(headers)

    def get_co2_intensity(
        self,
        country_code: str,
    ) -> float:
        """This endpoint retrieves the last 24 hours of carbon intensity (in gCO2eq/kWh) of an area. It can either be queried by zone identifier or by geolocation. The resolution is 60 minutes."""

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        params = {
            "zone": country_code,
        }
        response = self.retry_session.get(
            "https://api.electricitymap.org/v3/carbon-intensity/latest",
            params=params,
        )

        if response.ok:
            data = response.json()
            return data["carbonIntensity"]
        else:
            warnings.warn(
                f"Failed to fetch CO2 intensity data. Using {self.default} gCO₂eq/kWh. Details:\n{response.text}"
            )
            return self.default

    def get_co2_intensity_history(
        self,
        country_code: str,
    ) -> pd.DataFrame:
        """This endpoint retrieves the last 24 hours of carbon intensity (in gCO2eq/kWh) of an area. It can either be queried by zone identifier or by geolocation. The resolution is 60 minutes."""

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        params = {
            "zone": country_code,
        }
        response = self.retry_session.get(
            "https://api.electricitymap.org/v3/carbon-intensity/history",
            params=params,
        )

        if response.ok:
            data: dict = response.json()
            return pd.DataFrame.from_records(
                data["history"],
                index="datetime",
                columns=["datetime", "carbonIntensity"],
            )
        else:
            warnings.warn(
                f"Failed to fetch CO2 intensity data. Using {self.default} gCO₂eq/kWh. Details:\n{response.text}"
            )
            return pd.DataFrame(
                {
                    "datetime": pd.date_range(
                        start=pd.Timestamp.now().normalize()
                        - pd.Timedelta(days=1),
                        periods=24,
                        freq="H",
                    ),
                    "carbonIntensity": [self.default] * 24,
                }
            ).set_index("datetime")


class Entsoe:
    """This endpoint retrieves the last 24 hours of electricity price (in €/MWh) of an area. It can be queried by zone identifier. The resolution is 60 minutes.

    https://newtransparency.entsoe.eu/
    """

    def __init__(
        self,
        api_key: str | None = None,
        default: float = 0.0612,  # [EUR/kWh]
    ):
        if api_key is None:
            api_key = os.getenv("ENTSOE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided or set in the .env file"
                )
        from entsoe import EntsoePandasClient

        self.client = EntsoePandasClient(api_key=api_key)
        self.default = default

    def get_electricity_price(
        self,
        country_code: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.Series:
        from entsoe.exceptions import NoMatchingDataError

        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date, tz="UTC")
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date, tz="UTC")
        try:
            df = self.client.query_day_ahead_prices(
                country_code,
                start=start_date,
                end=end_date,
                resolution="60min",
            )
            # Since the API provides only day ahead prices extrapolate daily pattern
            end_idx = pd.Timestamp(df.index[-1])
            if end_idx < end_date:
                last_day = df.loc[
                    df.index >= (end_idx - pd.Timedelta(hours=23))
                ]
                extrapolated_index = pd.date_range(
                    end_idx + pd.Timedelta(hours=1),
                    end_date.tz_convert(end_idx.tz),
                    freq="h",
                )
                extrapolated_series = pd.Series(
                    (
                        last_day.tolist()
                        * (len(extrapolated_index) // len(last_day) + 1)
                    )[: len(extrapolated_index)],
                    index=extrapolated_index,
                )
                df = pd.concat([df, extrapolated_series])

            start_idx = pd.Timestamp(df.index[0])
            if start_idx > start_date:
                first_day = df.loc[
                    df.index < (start_idx + pd.Timedelta(hours=24))
                ]
                prepended_index = pd.date_range(
                    start_date.tz_convert(start_idx.tz),
                    start_idx - pd.Timedelta(hours=1),
                    freq="h",
                )
                prepended_series = pd.Series(
                    (
                        first_day.tolist()
                        * (len(prepended_index) // len(first_day) + 1)
                    )[: len(prepended_index)],
                    index=prepended_index,
                )
                df = pd.concat([prepended_series, df])
        except NoMatchingDataError:
            warnings.warn("No matching data found. Using default price.")
            df = pd.Series(
                index=pd.date_range(start_date, end_date, freq="h"),
                data=self.default,
            )

        return df


def get_city_geocoding(
    city: str,
) -> tuple[str, str, str, str, float, float, int]:
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    response = requests.get(url)
    if response.ok:
        data = response.json()["results"][0]
        return (
            data["name"],
            data["country"],
            data["country_code"],
            data["timezone"],
            data["latitude"],
            data["longitude"],
            data["elevation"],
        )
    else:
        raise ValueError(f"Failed to fetch city data: {response.text}")


class OpenMeteo:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        altitude: int | None,
        tilt: int | list[int],
        azimuth: int | str | list[int | str],
        frequency: Literal["hourly", "minutely_15", "current"] = "current",
    ):
        self.latitude = latitude
        self.longitude = longitude
        if altitude is None:
            # Get altitude based on latitude and longitude
            altitude_url = f"https://api.open-meteo.com/v1/elevation?latitude={self.latitude}&longitude={self.longitude}"
            altitude_response = requests.get(altitude_url)
            if altitude_response.ok:
                altitude_data = altitude_response.json()
                self.altitude: int = altitude_data["results"][0]["elevation"][
                    0
                ]
            else:
                raise ValueError(
                    f"Failed to fetch altitude data: {altitude_response.text}"
                )
        else:
            self.altitude = altitude

        self.tilt = tilt
        self.azimuth = azimuth
        self.frequency = frequency

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession(
            ".openmeteo.cache", expire_after=3600
        )
        self.retry_session = retry(
            cache_session, retries=5, backoff_factor=0.2
        )

    def azimuth_to_deg(self, azimuth: str):
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
        self,
        forecast: int | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ):
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%dT%H:%M")
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%dT%H:%M")

        if self.frequency != "current":
            if forecast is not None and (
                start_date is not None or end_date is not None
            ):
                raise ValueError(
                    "Either forecast or start_date and end_date must be provided"
                )

        # Determine the appropriate API URL based on the end_date
        now = datetime.now()
        if end_date is not None and pd.to_datetime(end_date) < now:
            if pd.to_datetime(end_date) < datetime(2022, 1, 1):
                self.frequency = "hourly"
                url = "https://archive-api.open-meteo.com/v1/archive"
            else:
                self.frequency = "minutely_15"
                url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        else:
            self.frequency = "minutely_15"
            url = "https://api.open-meteo.com/v1/forecast"

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            self.frequency: [
                "temperature_2m",
                "apparent_temperature",
                "wind_speed_10m",
                "relative_humidity_2m",
            ],
            "forecast_days": forecast,
            f"start_{self.frequency.rstrip('ly')}": start_date,
            f"end_{self.frequency.rstrip('ly')}": end_date,
        }
        responses = self.retry_session.get(url, params=params)

        if responses.ok:
            r = responses.json()
            # Convert the response to a pandas DataFrame
            data = r.pop(self.frequency)
            if "interval" in data:
                del data["interval"]
            if isinstance(data["time"], list):
                idx = data.pop("time")
            else:
                idx = [data.pop("time")]
            df = pd.DataFrame(data, index=idx)
            df.index = pd.to_datetime(df.index)

            # Add units to column names
            suffix_dict = r[f"{self.frequency}_units"]
            for col in df.columns:
                if col in suffix_dict:
                    df.rename(
                        columns={col: f"{col} [{suffix_dict[col]}]"},
                        inplace=True,
                    )
            return df
        else:
            raise ValueError(
                f"Failed to fetch data from Open-Meteo API: {responses.text}"
            )

    def get_irr_at_tilt_and_azimuth(
        self,
        forecast: int | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
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
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%d")

        if self.frequency != "current":
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

        now = datetime.now(timezone.utc)
        frequency_mapping = {
            "hourly": "h",
            "minutely_15": "15min",
            "current": "h",
        }
        freq = frequency_mapping[self.frequency]
        if self.frequency == "current":
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
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
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
        if not isinstance(self.tilt, list):
            tilt = [self.tilt]
        else:
            tilt = self.tilt
        if not isinstance(self.azimuth, list):
            azimuth = [self.azimuth]
        else:
            azimuth = self.azimuth
        df_rad = pd.DataFrame()
        df_rad["elevation"] = solar_position["elevation"]
        for t, a in zip(tilt, azimuth):
            if not isinstance(a, (int, float)):
                a = self.azimuth_to_deg(a)
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
                    columns=[
                        "poa_global",
                        "poa_sky_diffuse",
                        "poa_ground_diffuse",
                    ]
                )
                radiation.loc[:, ~radiation.columns.str.contains("poa_global")]
                df_rad = pd.concat(
                    [df_rad, radiation.add_suffix(f" [W/m²]:t{t}a{a}")],
                    axis=1,
                )
        return df_rad

    def sort_diffuse_last(self, df: pd.DataFrame) -> pd.DataFrame:
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
        self,
        forecast: int | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ):
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%dT%H:%M")
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%dT%H:%M")

        df = self.get_openmeteo(forecast, start_date, end_date)
        df_rad = self.get_irr_at_tilt_and_azimuth(
            forecast,
            start_date,
            end_date,
        )

        df = pd.concat([df, df_rad], axis=1)

        df = df.dropna()
        df = self.sort_diffuse_last(df)
        return df


if __name__ == "__main__":
    # Example usage
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
        frequency="hourly",
    )
    forecast: int = 2
    start_date = None
    end_date = None

    df = openmeteo.get_weather_data(
        forecast,
        start_date,
        end_date,
    )
