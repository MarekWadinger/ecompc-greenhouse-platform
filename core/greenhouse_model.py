import sys
from pathlib import Path
from typing import Union

import casadi as ca
import numpy as np

sys.path.insert(1, str(Path().resolve()))
from core.actuators import (
    SimpleCO2Generator,
    SimpleEvaporativeHumidifier,
    SimpleFan,
    SimpleHeater,
)
from core.lettuce_model import (
    C_LAR,
    DRY_TO_WET_RATIO,
    RATIO_SDW_NSDW,
    get_f_resp,
    lettuce_growth_model,
)

# CONSTANTS
Nz = 1.0

# Constants
sigm = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]
T_k = 273.15  # zero celsius [K]
g = 9.81  # acceleration due to gravity [m/s^2]
atm = 1.013e5  # standard atmospheric pressure [Pa]
N_A = 6.02214e23  # Avogadro's number
M_a = 0.029  # molar mass of dry air [kg/mol]
lam = 0.025  # thermal conductivity of air [W/m/K]
c_i = 1003.2  # heat capacity of humid air [J/kg/K]
H_fg = 2437000.0  # latent heat of condensation of water [J/kg]
Le = 0.819  # Lewis number [-]
R = 8.314  # gas constant [J/mol/K]
M_w = 0.018  # molar mass of water [kg/mol]
M_c = 0.044  # molar mass of CO2 [kg/mol]
M_carb = 0.03  # molar mass of CH2O [kg/mol]
nu = 15.1e-6  # kinematic viscosity [m^2/s]
rho_i = 1.2  # density of dry air [kg/m^3]
rho_w = 1000.0  # density of water [kg/m^3]
a_obs = 0.05  # fraction of solar radiation hitting obstructions [-]

# Air characteristics
IAS = 0.5  # internal air speed [m/s]

# Cover
# Glass
eps_ce = 0.85  # far-IR emissivity, outer surface [-]
eps_ci = 0.85  # far-IR emissivity, inner surface [-]
tau_c = 0.0  # far-IR transmissivity (0.0) [-]
rho_ci = 0.15  # far-IR reflectivity, inner surface (0.1) [-]
alph_c = 0.04  # solar absorptivity, taking 'perpendicular' values [-]
tau_c_NIR = 0.85  # near-IR transmissivity of cover (0.84) [-]
tau_c_VIS = 0.85  # visible transmissivity of cover [-]
d_c = 1.5  # characteristic length of cover [m]
cd_c = 8736.0  # cover heat capacity per unit area [J/m^2/K]

# Floor
lam_s = [
    1.7,
    0.85,
    0.85,
    0.85,
    0.85,
]  # thermal conductivity of soil layers [W/mK] Concrete, Soil, Clay
c_s = [
    880.0,
    1081.0,
    1081.0,
    1081.0,
    1081.0,
]  # specific heat of soil layers [J/kgK]
l_s = [0.02, 0.05, 0.1, 0.25, 1.0]  # thickness of soil layers [m]
rhod_s = [
    2300.0,
    1500.0,
    1600.0,
    1600.0,
    1600.0,
]  # density of soil layers [kg/m^3]
rho_s = 0.85  # far-IR reflectance of floor [-]
eps_s = 0.95  # far-IR emmittance of floor [-]
rhoS_s = 0.5  # solar reflectance of floor [-]
alphS_s = 0.5  # solar absorptance of floor [-]
d_f = 0.5  # characteristic floor length [m]
T_ss = 10.0 + T_k  # deep soil temperature [K]

# Vegetation
c_v = 4180.0  # heat capacity of vegetation [J/kgK]
k_l = 0.94  # long-wave extinction coefficient [-]
rho_v = 0.22  # far-IR reflectivity of vegetation [-]
eps_v = 0.95  # far-IR emissivity of vegetation [-]
rhoS_v = 0.35  # solar reflectance of vegetation [-]
d_v = 0.1  # characteristic leaf length [m]
p_v = 0.75  # cultivated fraction of floor
msd_v = 1.326  # surface density [kg/m^2]

# Tray/mat
d_p = 1.0  # characteristic dimension of tray (width)
d_m = 0.1  # characteristic dimension of mat (width)
lam_p = 0.2  # thermal conductivity of plastic tray [W/mK]
c_m = 45050.0  # specific heat of mat assumed 25% saturated [J/m^2K]
c_p = 10020.0  # specific heat of tray [J/m^2K]
l_m = 0.03  # thickness of mat [m]
l_p = 0.005  # thickness of tray [m]
rho_m = 0.05  # far-IR reflectivity of mat [-]
rho_p = 0.05  # far-IR reflectivity of tray
eps_m = 0.95  # far-IR emissivity of mat [-]
eps_p = 0.95  # far-IR emissivity of tray

# Photosynthesis model - Vanthoor
heat_phot = 3.6368e-19  # conversion rate from incident energy to number of photons [num{photons}/J]


# Infiltration
c = 0.35  # terrain factor, see Awbi, Chapter 3, Table 3.2
a = 0.25  # terrain factor, see Awbi, Chapter 3, Table 3.2
Cp = 0.62  # static pressure coefficient - for wind perpendicular to gap
Cd = 0.61  # sharp edge orifice, see Awbi
crack_length = 1.0  # typical estimate
crack_width = 0.001  # typical estimate
crack_area = crack_length * crack_width
crack_length_total = 350.0

# View Factors
F_f_c = 1 - p_v  # Floor to cover
F_f_p = p_v  # Floor to tray
F_v_c = 0.5  # Vegetation to cover
F_v_m = 0.5  # Vegetation to mat
F_v_p = 0.0  # Vegetation to tray
F_m_p = 0.0  # Mat to tray
F_p_v = 0.0  # Tray to vegetation
F_p_m = 0.0  # Tray to mat
F_p_f = 1.0  # Tray to floor


# Crop Growth Model
SLA = C_LAR  # specific leaf area index [m^2{leaf}/g{CH2O}]
LAI_max = 5.0  # the maximum allowed leaf area index [m^2{leaf}/m^2{floor}]

## Initial conditions
# Temperatures
T_c_0 = 20.0 + T_k  # Cover temperature [K]
T_i_0 = 12.0 + T_k  # Internal air temperature [K]
T_v_0 = 12.0 + T_k  # Vegetation temperature [K]
T_m_0 = 12.0 + T_k  # Growing medium temperature [K]
T_p_0 = 12.0 + T_k  # Tray temperature [K]
T_f_0 = 12.0 + T_k  # Floor temperature [K]
T_s1_0 = 12.0 + T_k  # Temperature of soil layer 1 [K]

C_w_0 = 0.0085  # Density of water vapour [kg/m^3]
C_c_0 = 7.5869e-4  # CO_2 density

# The proportion should be somewhere between 40:60 - 30:70 for lettuce
x_lettuce_dry_init = 500 * DRY_TO_WET_RATIO  # g/m^2

# Structural and non-structural dry weight of the plant [g/m^2]
x_sdw, x_nsdw = x_lettuce_dry_init * RATIO_SDW_NSDW

x_init = np.array(
    [
        T_c_0,
        T_i_0,
        T_v_0,
        T_m_0,
        T_p_0,
        T_f_0,
        T_s1_0,
        C_w_0,
        C_c_0,
        x_sdw,
        x_nsdw,
    ],
    dtype=float,
)
x_init_dict = {
    "Cover temperature": T_c_0,
    "Internal air temperature": T_i_0,
    "Vegetation temperature": T_v_0,
    "Growing medium temperature": T_m_0,
    "Tray temperature": T_p_0,
    "Floor temperature": T_f_0,
    "Temperature of soil layer 1": T_s1_0,
    "Water vapor concentration": C_w_0,
    "CO2 concentration": C_c_0,
    "Structural dry weight": x_sdw,
    "Non-structural dry weight": x_nsdw,
}


# FUNCTIONS
# # Dynamic Behavior Models
## GSE model
def lamorturb(Gr, Re):
    free = Gr < 1e5
    Nu_G = 0.5 * free * Gr**0.25 + 0.13 * (1 - free) * Gr**0.33

    forced = Re < 2e4
    Nu_R = 0.6 * forced * Re**0.5 + 0.032 * (1 - forced) * Re**0.8

    x = Nu_G > Nu_R

    Nu = x * Nu_G + (1 - x) * Nu_R

    Sh = x * Nu * Le**0.25 + (1 - x) * Nu * Le**0.33

    return (Nu, Sh)


def convection(
    d, A, T1, T2: Union[float, ca.GenericExpressionCommon], ias, rho, c, C
):
    g = 9.81
    nu = 15.1e-6
    lam = 0.025
    if isinstance(T2, ca.GenericExpressionCommon):
        abs_diff_T = ca.fabs(T1 - T2)
    else:
        abs_diff_T = abs(T1 - T2)
    Gr = (g * d**3) / (T1 * nu**2) * abs_diff_T
    Re = ias * d / nu
    (Nu, Sh) = lamorturb(Gr, Re)

    QV_1_2 = A * Nu * lam * (T1 - T2) / d
    QP_1_2 = A * H_fg / (rho * c) * Sh / Le * lam / d * (C - sat_conc(T2))

    return (QV_1_2, QP_1_2, Nu)


def radiation(eps_1, eps_2, rho_1, rho_2, F_1_2, F_2_1, A_1, T_1, T_2):
    sigm = 5.67e-8

    k = eps_1 * eps_2 / (1 - rho_1 * rho_2 * F_1_2 * F_2_1)
    QR_1_2 = k * sigm * A_1 * F_1_2 * (T_1**4 - T_2**4)

    return QR_1_2


def conduction(A, lam, thickness, T1, T2):
    QD_12 = (A * lam / thickness) * (T1 - T2)

    return QD_12


def sat_conc(T):
    TC = T - T_k
    spec_hum = np.exp(11.56 - 4030 / (TC + 235))
    air_dens = -0.0046 * TC + 1.2978
    a = spec_hum * air_dens

    return a


class GreenHouse:
    def __init__(
        self,
        length: float = 25.0,
        width: float = 10.0,
        height: float = 4.0,
        roof_tilt: float = 30.0,
        max_vent: float | None = None,
        max_heat: float | None = None,
        max_humid: float | None = None,  # Maximum humidification in l/h
        max_co2: float | None = None,
        latitude: float = 53.193583,  #  latitude of greenhouse
        longitude: float = 5.799383,  # longitude of greenhouse
        dt=60,  # sampling time in seconds
        **act_kwargs,
    ) -> None:
        self.length = length
        self.width = width
        self.height = height
        self.roof_tilt = roof_tilt
        self.latitude = latitude
        self.longitude = longitude
        self.dt = dt

        # Geometry
        roof_height = width / 2 * np.tan(np.radians(roof_tilt))
        roof_width = np.sqrt(width**2 + roof_height**2)
        self.A_f = length * width  # greenhouse floor area [m^2]
        wall_surface = [width * height] * 2 + [length * height] * 2
        roof_surface = [roof_width * roof_height] * 2 + [
            roof_width * length
        ] * 2

        self.surface_area = np.array(
            wall_surface + roof_surface
        )  # surface areas [m^2]
        self.area = np.sum(self.surface_area)
        self.area_roof = np.sum(roof_surface)
        self.volume = (
            length * width * height + roof_width * roof_height * length
        )  # greenhouse volume [m^3]

        # Air characteristics
        ACH = 20  # air changes per hour
        R_a_max = self.volume * ACH / 3600.0  # fan air change rate [m^3/s]
        T_sp_vent = 25.0 + T_k  # setpoint temperature for fan [K]

        if max_vent is not None:
            R_a_max = max_vent
        self.fan = SimpleFan(R_a_max, dt=self.dt, **act_kwargs)

        # Heater
        # Q_heater_max is computed as the mass of air we want to heat per second
        #  considering the heat capacity of air and the temperature difference
        # TODO: Scaling this value by 100 makes heater overly powerful and may kill the plant
        # TODO: Scaling this value by 100 makes the plant grow much faster
        if max_heat is not None:
            Q_heater_max = max_heat
        else:
            Q_heater_max = rho_i * (T_sp_vent - T_k) * R_a_max * c_i
        self.heater = SimpleHeater(Q_heater_max, dt=self.dt, **act_kwargs)

        # Humidifier
        if max_humid is not None:
            V_dot_max = max_humid  # [l/h]
        else:
            # https://www.tis-gdv.de/tis_e/misc/klima-htm/
            # RH_range = 40 % - 80 % at 20C [- / h]
            AH_40 = 6.9  # [g/m^3]
            AH_80 = 13.8  # [g/m^3]
            max_AH_increase = AH_80 - AH_40  # [g/m^3/h]
            V_dot_max = self.volume * max_AH_increase / rho_w  # [l/h]
        self.humidifier = SimpleEvaporativeHumidifier(
            V_dot_max, dt=self.dt, **act_kwargs
        )

        if max_co2 is not None:
            co2_gen_max = max_co2
        else:
            # https://www.hotboxworld.com/product/co2-
            Cco2_per_hour = 0.009  # [kg/m^3/h]
            co2_gen_max = (
                Cco2_per_hour * self.volume
            )  # Maximum CO2 generation in kg/h
        self.co2generator = SimpleCO2Generator(
            co2_gen_max, dt=self.dt, **act_kwargs
        )

        # Tray/mat
        self.A_c = p_v * self.A_f  # Area of cultivated floor [m^2]
        self.A_p = self.A_c  # Area of plants [m^2]
        self.A_m = self.A_c  # Area of mat for conduction to tray [m^2]
        self.A_f_c = self.A_p / self.A_f  # Fraction of floor covered by plants
        self.A_f_nc = (
            self.A_f - self.A_p
        ) / self.A_f  # Fraction of floor not covered by plants

        # View Factors
        self.F_c_f = self.A_f / self.area_roof * F_f_c  # Cover to floor

    @property
    def active_actuators(self) -> dict[str, bool]:
        """Return the list of active actuators."""
        return {
            "Fan": self.fan.max_unit > 0,
            "Heater": self.heater.max_unit > 0,
            "Humidifier": self.humidifier.max_unit > 0,
            "CO2 Generator": self.co2generator.max_unit > 0,
        }

    def model(
        self,
        t,
        x: tuple | np.ndarray,
        u: tuple | np.ndarray,
        climate: tuple | np.ndarray | None = None,
    ) -> np.ndarray:
        """Greenhouse model.

        This function models the greenhouse system with

        Args:
            t: Elapsed time [s]
            z: System states
            u: System inputs
            c: System parameters
            climate: climate information. Must be sampled at dt and have appropriate length.

        Returns:
            np.ndarray: System state derivatives
        """
        if climate is None:
            raise ValueError("Climate information must be provided.")
        dz_dt, __ = self._model(t, x, u, climate)
        return dz_dt

    def _model(
        self,
        t,
        z: tuple | np.ndarray,
        u: tuple | np.ndarray,
        climate: tuple | np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        T_c = z[0]  # Cover temperature [K]
        T_i = z[1]  # Internal air temperature [K]
        T_v = z[2]  # Vegetation temperature [K]
        T_m = z[3]  # Growing medium temperature [K]
        T_p = z[4]  # Tray temperature [K]
        T_f = z[5]  # Floor temperature [K]
        T_s1 = z[6]  # Temperature of soil layer 1 [K]
        C_w = z[7]  # Water vapor concentration [kg/m^3]
        C_c = z[8]  # CO2 concentration [kg/m^3]
        x_sdw = z[9]  # Structural dry weight of the plant [kg/m^2]
        x_nsdw = z[10]  # Non-structural dry weight of the plant [kg/m^2]
        perc_vent = u[0]
        perc_heater = u[1]
        perc_humid = u[2]
        perc_co2 = u[3]

        R_a = self.fan.signal_to_actuation(perc_vent)
        Q_heater = self.heater.signal_to_actuation(perc_heater)
        V_dot = (
            self.humidifier.signal_to_actuation(perc_humid) * rho_w
        )  # [g/h]
        # mass of CO2 pumped in per hour [kg/h]
        added_CO2 = self.co2generator.signal_to_actuation(perc_co2)

        ias = IAS + R_a / (self.width * self.height)

        # External weather and dependent internal parameter values
        if isinstance(climate, np.ndarray):
            climate = tuple(climate[int(t) // self.dt, :])
        T_ext = climate[0] + T_k  # External air temperature (K)
        T_sk = climate[1] + T_k  # External sky temperature (K)
        wind_speed = climate[2]  # External wind speed (m/s)
        RH_e = climate[3] / 100  # External relative humidity
        Cw_ext = RH_e * sat_conc(T_ext)  # External air moisture content
        p_w = C_w * R * T_i / M_w  # Partial pressure of water [Pa]
        rho_i = ((atm - p_w) * M_a + p_w * M_w) / (
            R * T_i
        )  # Internal density of air [kg/m^3]
        # Let's assume all the bimass are leafs
        LAI = ca.fmin(SLA * x_sdw, LAI_max)  # Leaf area index
        C_ce = (
            4.0e-4 * M_c * atm / (R * T_ext)
        )  # External carbon dioxide concentration [kg/m^3]
        C_c_ppm = (
            C_c * R * T_i / (M_c * atm) * 1.0e6
        )  # External carbon dioxide concentration [ppm]
        h = 6.626e-34  # Planck's constant in Joule*Hz^{-1}

        ## Lights
        # _ = 0  # No additional lighting included
        # _ = 0  # No ambient lighting included

        ## Convection
        # Convection external air -> cover

        (QV_e_c, _, _) = convection(
            d_c, self.area, T_ext, T_c, wind_speed, rho_i, c_i, C_w
        )

        # Convection internal air -> cover
        (QV_i_c, QP_i_c, _) = convection(
            d_c, self.area, T_i, T_c, ias, rho_i, c_i, C_w
        )
        QP_i_c = ca.fmax(
            QP_i_c, 0
        )  # assumed no evaporation from the cover, only condensation

        # Convection internal air -> floor

        (QV_i_f, QP_i_f, _) = convection(
            d_f, self.A_f, T_i, T_f, ias, rho_i, c_i, C_w
        )
        QP_i_f = ca.fmax(
            QP_i_f, 0
        )  # assumed no evaporation from the floor, only condensation

        # Convection internal air -> vegetation
        A_v_exp = LAI * self.A_p
        (QV_i_v, _, Nu_i_v) = convection(
            d_v, A_v_exp, T_i, T_v, ias, rho_i, c_i, C_w
        )
        HV = Nu_i_v * lam / d_v

        # Convection internal air -> mat
        A_m_wool = 0.75 * self.A_m  # Area of mat exposed
        A_m_water = 0.25 * self.A_m  # assumed 25% saturated

        (QV_i_m, QP_i_m, _) = convection(
            d_m, A_m_wool, T_i, T_m, ias, rho_i, c_i, C_w
        )

        QP_i_m = A_m_water / A_m_wool * QP_i_m  # Factored down

        # Convection internal air -> tray
        (QV_i_p, QP_i_p, _) = convection(
            d_p, self.A_c, T_i, T_p, ias, rho_i, c_i, C_w
        )
        # QP_i_p = 0  # Assumed no condensation/evaporation from tray

        ## Far-IR Radiation

        A_vvf = ca.fmin(LAI * p_v * self.A_f, p_v * self.A_f)
        F_c_v = ca.fmin(
            (1 - self.F_c_f) * LAI, (1 - self.F_c_f)
        )  # Cover to vegetation
        F_c_m = ca.fmax((1 - self.F_c_f) * (1 - LAI), 0)  # Cover to mat
        F_m_c = ca.fmax((1 - LAI), 0.0)  # Mat to cover
        F_m_v = 1 - F_m_c  # Mat to vegetation

        # Cover to sky
        QR_c_sk = radiation(eps_ce, 1, 0, 0, 1, 0, self.area, T_c, T_sk)

        # Radiation cover to floor
        QR_c_f = radiation(
            eps_ci,
            eps_s,
            rho_ci,
            rho_s,
            self.F_c_f,
            F_f_c,
            self.area_roof,
            T_c,
            T_f,
        )

        # Radiation cover to vegetation
        QR_c_v = radiation(
            eps_ci,
            eps_v,
            rho_ci,
            rho_v,
            F_c_v,
            F_v_c,
            self.area_roof,
            T_c,
            T_v,
        )

        # Radiation cover to mat
        QR_c_m = radiation(
            eps_ci,
            eps_m,
            rho_ci,
            rho_m,
            F_c_m,
            F_m_c,
            self.area_roof,
            T_c,
            T_m,
        )

        # Radiation vegetation to cover
        QR_v_c = radiation(
            eps_v, eps_ci, rho_v, rho_ci, F_v_c, F_c_v, A_vvf, T_v, T_c
        )

        # Radiation vegetation to mat
        QR_v_m = radiation(
            eps_v, eps_m, rho_v, rho_m, F_v_m, F_m_v, A_vvf, T_v, T_m
        )

        # Radiation vegetation to tray
        QR_v_p = radiation(
            eps_v, eps_p, rho_v, rho_p, F_v_p, F_p_v, A_vvf, T_v, T_p
        )

        # Radiation mat to cover
        QR_m_c = radiation(
            eps_m, eps_ci, rho_m, rho_ci, F_m_c, F_c_m, self.A_m, T_m, T_c
        )

        # Radiation mat to vegetation
        QR_m_v = radiation(
            eps_m, eps_v, rho_m, rho_v, F_m_v, F_v_m, self.A_m, T_m, T_v
        )

        # Radiation mat to tray
        QR_m_p = radiation(
            eps_m, eps_p, rho_m, rho_p, F_m_p, F_p_m, self.A_m, T_m, T_p
        )

        # Radiation tray to vegetation
        QR_p_v = radiation(
            eps_p, eps_v, rho_p, rho_v, F_p_v, F_v_p, self.A_c, T_p, T_v
        )

        # Radiation tray to mat
        QR_p_m = radiation(
            eps_p, eps_m, rho_p, rho_m, F_p_m, F_m_p, self.A_c, T_p, T_m
        )

        # Radiation tray to floor
        QR_p_f = radiation(
            eps_p, eps_s, rho_p, rho_s, F_p_f, F_f_p, self.A_c, T_p, T_f
        )

        # Radiation floor to cover
        QR_f_c = radiation(
            eps_s, eps_ci, rho_s, rho_ci, F_f_c, self.F_c_f, self.A_f, T_f, T_c
        )

        # Radiation floor to tray
        QR_f_p = radiation(
            eps_s, eps_p, rho_s, rho_p, F_f_p, F_p_f, self.A_f, T_f, T_p
        )

        ## Conduction
        # Conduction through floor
        QD_sf1 = conduction(self.A_f, lam_s[0], l_s[0], T_f, T_s1)
        QD_s12 = conduction(self.A_f, lam_s[1], l_s[1], T_s1, T_ss)

        # Conduction mat to tray
        QD_m_p = conduction(self.A_m, lam_p, l_m, T_m, T_p)

        ## Ventilation
        # Leakage (equations for orifice flow from Awbi, Ventilation of Buildings, Chapter 3)
        wind_speed_H = (
            wind_speed * c * self.height**a
        )  # Wind speed at height H
        wind_pressure = (
            Cp * 0.5 * rho_i * wind_speed_H**2
        )  # Equals DeltaP for wind pressure [Pa]
        stack_pressure_diff = (
            rho_i * g * self.height * (T_i - T_ext) / T_i
        )  # DeltaP for stack pressure [Pa]

        Qw = (
            Cd * crack_area * (2 * wind_pressure / rho_i) ** 0.5
        )  # Flow rate due to wind pressure
        Qs = (
            Cd * crack_area * (2 * ca.fabs(stack_pressure_diff) / rho_i) ** 0.5
        )  # Flow rate due to stack pressure
        Qt = (Qw**2 + Qs**2) ** 0.5  # Total flow rate

        total_air_flow = Qt * crack_length_total / crack_length
        R_a_min = total_air_flow / self.volume

        # Ventilation account for disturbance
        R_a = R_a_min + R_a / self.volume

        QV_i_e = (
            R_a * self.volume * rho_i * c_i * (T_i - T_ext)
        )  # Internal air to outside air [J/s]

        MW_i_e = R_a * (
            C_w - Cw_ext
        )  # Internal moisture to outside air [kg/m^3/s]

        ##      Solar radiation
        # We first define the solar elevation angle that determines that absorption of solar radiation. Notation: r is direct radiation, f is diffuse radiation, whilst VIS and NIR stand for visible and near infra-red respectively.

        angle = climate[4]

        # Radiation from artificial lighting
        QS_al_NIR = 0.0  # no artificial lighting
        QS_al_VIS = 0.0

        # Solar radiation incident on the cover
        QS_tot_rNIR = 0.5 * self.surface_area @ climate[5:13]  # Direct
        QS_tot_rVIS = 0.5 * self.surface_area @ climate[5:13]
        QS_tot_fNIR = 0.5 * self.surface_area @ climate[13:21]  # Diffuse
        QS_tot_fVIS = 0.5 * self.surface_area @ climate[13:21]

        # Transmitted solar radiation
        QS_int_rNIR = tau_c_NIR * QS_tot_rNIR  # J/s total inside greenhouse
        QS_int_rVIS = tau_c_VIS * QS_tot_rVIS
        QS_int_fNIR = tau_c_NIR * QS_tot_fNIR
        QS_int_fVIS = tau_c_VIS * QS_tot_fVIS

        # Solar radiation absorbed by the cover and the obstructions
        QS_c = alph_c * (
            QS_tot_rNIR + QS_tot_rVIS + QS_tot_fNIR + QS_tot_fVIS
        )  # J/s
        QS_i = a_obs * (QS_int_rNIR + QS_int_rVIS + QS_int_fNIR + QS_int_fVIS)

        # Solar radiation absorbed by the vegetation
        # Area = A_v i.e. planted area
        # factor QS by A_v/self.A_f

        k_fNIR = 0.27  # Near-IR diffuse extinction coefficient [-]
        a_v_fNIR = 0.65 - 0.65 * np.exp(
            -k_fNIR * LAI
        )  # Near-IR diffuse absorption coefficient [-]

        k_fVIS = 0.85  # Visible diffuse extinction coefficient [-]
        a_v_fVIS = 0.95 - 0.9 * np.exp(
            -k_fVIS * LAI
        )  # Visible diffuse absorption coefficient [-]

        k_rNIR = 0.25 + 0.38 * np.exp(
            -0.12 * angle
        )  # Near-IR direct extinction coefficient [-]
        a_v_rNIR = (
            0.67
            - 0.06 * np.exp(-0.08 * angle)
            - (0.68 - 0.5 * np.exp(-0.11 * angle)) * np.exp(-k_rNIR * LAI)
        )  # Near-IR direct absorption coefficient [-]

        k_rVIS = 0.88 + 2.6 * np.exp(
            -0.18 * angle
        )  # Visible direct extinction coefficient [-]
        a_v_rVIS = 0.94 - 0.95 * np.exp(
            -k_rVIS * LAI
        )  # Visible direct absorption coefficient [-]

        QS_v_rNIR = (
            (QS_int_rNIR * (1 - a_obs) + QS_al_NIR)
            * a_v_rNIR
            * self.A_p
            / self.A_f
        )
        QS_v_fNIR = (QS_int_fNIR * (1 - a_obs)) * a_v_fNIR * self.A_f_c
        QS_v_NIR = (
            QS_v_rNIR + QS_v_fNIR
        )  # factor as planted area not entire floor

        QS_v_rVIS = (
            (QS_int_rVIS * (1 - a_obs) + QS_al_VIS)
            * a_v_rVIS
            * self.A_p
            / self.A_f
        )
        QS_v_fVIS = (QS_int_fVIS * (1 - a_obs)) * a_v_fVIS * self.A_f_c
        QS_v_VIS = QS_v_rVIS + QS_v_fVIS  # Used for photosynthesis calc

        # Solar radiation absorbed by the mat
        a_m_fNIR = 0.05 + 0.91 * np.exp(
            -0.5 * LAI
        )  # Near-IR diffuse absorption coefficient [-]
        a_m_rNIR = (
            0.05
            + 0.06 * np.exp(-0.08 * angle)
            + (0.92 - 0.53 * np.exp(-0.18 * angle))
            * np.exp(-(0.48 + 0.54 * np.exp(-0.13 * angle)) * LAI)
        )  # Near-IR direct absorption coefficient [-]

        QS_m_rNIR = (
            (QS_int_rNIR * (1 - a_obs) + QS_al_NIR)
            * a_m_rNIR
            * self.A_p
            / self.A_f
        )
        QS_m_fNIR = QS_int_fNIR * (1 - a_obs) * a_m_fNIR * self.A_f_c  # W
        QS_m_NIR = QS_m_rNIR + QS_m_fNIR

        # Solar radiation absorbed by the floor
        # factor by (self.A_f-self.A_v)/self.A_f

        QS_s_rNIR = QS_int_rNIR * (1 - a_obs) * alphS_s * self.A_f_nc
        QS_s_fNIR = QS_int_fNIR * (1 - a_obs) * alphS_s * self.A_f_nc
        QS_s_NIR = QS_s_rNIR + QS_s_fNIR

        ## Transpiration
        QS_int = (
            (QS_int_rNIR + QS_int_rVIS + QS_int_fNIR + QS_int_fVIS)
            * (1 - a_obs)
            * self.A_p
            / self.A_f
        )  # J/s

        #  Vapour pressure deficit at leaf surface
        xa = C_w / rho_i  # [-]
        xv = sat_conc(T_v) / rho_i  # [-]
        vpd = atm * (xv / (xv + 0.622) - xa / (xa + 0.622))  # [Pa]

        # Stomatal resistance according to Stanghellini
        x = np.exp(-0.24 * LAI)  # [-]
        a_v_short = (
            0.83
            * (1 - 0.70 * x)
            * (1 + 0.58 * x**2)
            * (0.88 - x**2 + 0.12 * x ** (8 / 3))
        )  # [-]Absorption for shortwave radiation
        I_s_bar = (
            QS_int * a_v_short / (2 * LAI)
        )  # [J/s] Mean radiation interacting with leaf surface

        Heavy_CO2 = I_s_bar > 0.0
        r_i_CO2 = 1 + Heavy_CO2 * 6.1e-7 * (C_c_ppm - 200) ** 2
        Heavy_vpd = vpd / 1000 < 0.8
        r_i_vpd = (
            Heavy_vpd * (1 + 4.3 * (vpd / 1000) ** 2) + (1 - Heavy_vpd) * 3.8
        )
        r_st = (
            82
            * ((QS_int + 4.3) / (QS_int + 0.54))
            * (1 + 0.023 * (T_v - T_k - 24.5) ** 2)
            * r_i_CO2
            * r_i_vpd
        )  # [s/m]

        hL_v_i = (
            2
            * LAI
            * H_fg
            / (rho_i * c_i)
            * (Le ** (2 / 3) / HV + r_st / (rho_i * c_i)) ** (-1)
        )

        QT_St = self.A_p * hL_v_i * (sat_conc(T_v) - C_w)  # J/s

        QT_v_i = ca.fmax(QT_St, 0)

        ## Dehumidification
        MW_cc_i = V_dot / 1000 / 3600 / self.volume  # [kg/m^3/s]

        # CO2 exchange with outside
        MC_i_e = R_a * (C_c - C_ce)  # [kg/m^3/s]

        MC_cc_i = (
            added_CO2 / 3600.0 / Nz / self.volume
        )  # [kg/h / - / 3600 / m^3] -> [kg m^{-3} s^{-1}]

        ## Photosynthesis model - Vanthoor

        # Consider photosynthetically active radiation to be visible radiation

        I_VIS = QS_v_VIS  # J/s incident on planted area

        PAR = I_VIS / heat_phot / N_A / self.A_p  # [mol{photons}.s−1.m−2]

        # The number of moles of photosynthetically active photons per unit area of planted floor [mol{phot}/m^2/s]
        # J/s/(J/photon)/(photons/mol)/m^2 cf Vanthoor 2.3mumol(photons)/J

        # Maintenance respiration
        f_resp = get_f_resp(x_sdw, T_i - T_k)

        # Temperature components
        dT_c_dt = (1 / (self.area * cd_c)) * (
            QV_i_c
            + QP_i_c
            - QR_c_f
            - QR_c_v
            - QR_c_m
            + QV_e_c
            - QR_c_sk
            + QS_c
        )

        dT_i_dt = (1 / (self.volume * rho_i * c_i)) * (
            -QV_i_m
            - QV_i_v
            - QV_i_f
            - QV_i_c
            - QV_i_e
            - QV_i_p
            + QS_i
            + Q_heater
        )

        dT_v_dt = (1 / (c_v * self.A_p * msd_v)) * (
            QV_i_v - QR_v_c - QR_v_m - QR_v_p + QS_v_NIR - QT_v_i
        )

        dT_m_dt = (1 / (self.A_m * c_m)) * (
            QV_i_m + QP_i_m - QR_m_v - QR_m_c - QR_m_p - QD_m_p + QS_m_NIR
        )
        dT_p_dt = (1 / (self.A_c * c_p)) * (
            QD_m_p + QV_i_p + QP_i_p - QR_p_f - QR_p_v - QR_p_m
        )
        dT_f_dt = (1 / (rhod_s[0] * self.A_f * c_s[0] * l_s[0])) * (
            QV_i_f + QP_i_f - QR_f_c - QR_f_p - QD_sf1 + QS_s_NIR
        )
        dT_s1_dt = (1 / (rhod_s[1] * c_s[1] * l_s[1] * self.A_f)) * (
            QD_sf1 - QD_s12
        )

        # Water vapour
        dC_w_dt = (
            (1 / (self.volume * H_fg))
            * (QT_v_i - QP_i_c - QP_i_f - QP_i_m - QP_i_p)
            - MW_i_e
            + MW_cc_i
        )

        # Carbon Dioxide
        dC_c_dt = (
            MC_cc_i  # [kg/m^3/s]
            - MC_i_e  # [kg/m^3/s]
            + (M_c / M_carb)  # [-]
            * (self.A_p / self.volume)  # [m^2 / m^3]
            * (f_resp)
            / 1000  # [g m^{-2}] / 1000 -> [g m^{-2} s^{-1}]
        )

        # Salaatia growth
        cLight = 3.0e8  # Speed of light [m/s]
        lambda_nm = 550  # Wavelength [nm]
        lambda_m = lambda_nm * 1e-9
        E = h * cLight / lambda_m  # [J/num{photons}]
        u_par = PAR * E * N_A  # [W/m^2]
        u_co2 = C_c_ppm  # [ppm] >> external C_c_ppm
        dx_sdw_dt, dx_nsdw_dt = lettuce_growth_model(
            t, (x_sdw, x_nsdw), (T_i - T_k, u_par, u_co2)
        )

        return (
            np.array(
                [
                    dT_c_dt,
                    dT_i_dt,
                    dT_v_dt,
                    dT_m_dt,
                    dT_p_dt,
                    dT_f_dt,
                    dT_s1_dt,
                    dC_w_dt,
                    dC_c_dt,
                    dx_sdw_dt,
                    dx_nsdw_dt,
                ]
            ),
            locals(),
        )
