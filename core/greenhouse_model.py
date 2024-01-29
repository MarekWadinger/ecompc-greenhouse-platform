import numpy as np
from simpleParameters import *
from math import exp


# CONSTANTS
# # Lettuce Growth Model Parameters

# [g g^{-1}] ratio of molecular weight of CH_2O to CO_2
C_CH2O = 30 / 44
# [g g^{-1}] Penning de Vries et al. (1974)
C_YF = 0.8
# [s^{-1}] Van Holsteijn (1981)
C_GR_MAX = 5e-6
# [-] (0.5 - 1.) Thornley & Hurd (1974); (1.) Sweeney et al. (1981)
C_GAMMA = 1.1981
# [-] Sweeney et al. (1981)
C_Q10_GR = 1.6
# [s^{-1}] Van Keulen et al. (1982)
C_RESP_SHT = 3.47e-7
# [s^{-1}] Van Keulen et al. (1982)
C_RESP_RT = 1.16e-7
# [-] Van Keulen et al. (1982)
C_Q10_RESP = 2.0
# [-] Lorenz & Wiebe, 1980; Sweeney et al., 1981
C_TAU = 0.15
# [-] (0.9 and 0.3 for planophile and erectophile) Goudriaan & Monteith, 1990
C_K = 0.9
# [g^{-1} m^{-2}] Lorenz & Wiebe (1980)
C_LAR = 75e-3
# [g m^{-3}] temperature 15 C and pressure 101.3 kPa
C_OMEGA = 1.83e-3
# [ppm] CO_2 compensation point at 20 C (Goudriaan et al. 1985)
C_GGAMMA = 40
# [-] CO_2 compensation point sensitivity to temp (Goudriaan et al.1985)
C_Q10_GGAMMA = 2
# [g J^{-1}] light use efficiency (Goudriaan et al. 1985)
C_EPSILON = 17e-6
# [m s^{-1}] boundary layer conductance (Stanghellini et al.)
G_BND = 0.00071987
# [m s^{-1}] stomatal resistance (Stanghellini et al. 1987)
G_STM = 0.005
# [m s^{-1} C^{-2}] carboxilation parameter
C_CAR1 = -1.32e-5
# [m s^{-1} C^{-1}] carboxilation parameter
C_CAR2 = 5.94e-4
# [m s^{-1}] carboxilation parameter
C_CAR3 = -2.64e-3

# FUNCTIONS
# # Dynamic Behavior Models
def lettuce_growth_model(_: int, x: tuple[float, float], u: tuple[float, float, float]) -> tuple[float, float]:
    dx_sdw_dt, dx_nsdw_dt, info = _lettuce_growth_model(_, x, u)
    # print(f"Time: {_}, x_sdw: {x[0]}, x_nsdw: {x[1]}, dx_sdw_dt: {dx_sdw_dt}, dx_nsdw_dt: {dx_nsdw_dt}")
    return dx_sdw_dt, dx_nsdw_dt


def _lettuce_growth_model(
    _: int, x: tuple[float, float], u: tuple[float, float, float]
) -> tuple[float, float, dict]:
    """Overall dynamic growth model with states."""
    x_sdw, x_nsdw = x
    u_T, u_par, u_co2 = u

    r_gr = get_r_gr(x_sdw, x_nsdw, u_T)

    gamma = get_ggamma(u_T)
    epsilon = get_epsilon(u_co2, gamma)
    g_co2 = get_g_co2(get_g_car(u_T))
    f_phot_max = get_f_phot_max(u_par, u_co2, epsilon, g_co2, gamma)
    f_phot = get_f_phot(x_sdw, f_phot_max)

    f_resp = get_f_resp(x_sdw, u_T)

    dx_sdw_dt = predict_x_sdw(x_sdw, r_gr)
    dx_nsdw_dt = predict_x_nsdw(x_sdw, r_gr, f_phot, f_resp)
    return dx_sdw_dt, dx_nsdw_dt, locals()


def predict_x_sdw(
    x_sdw: float,
    r_gr: float,
) -> float:
    """Dynamic model of structural dry weight.

    Nonlinear differential equation governing dynamic behavior of structural
    dry weight.

    Args:
        x_sdw: structural dry weight [g m^{-2}]
        r_gr (float): specific growth rate [s^{-1}]

    Returns:
        dx_sdw/dt - structural dry weight increment [g m^{-2} s^{-1}]

    References:
        Sweeney et al., 1981
    """
    # params_string = (
    #     f"predict_x_sdw: "
    #     f"x_sdw (Structural Dry Weight) = {x_sdw} [g m^-2], "
    #     f"r_gr (Specific Growth Rate) = {r_gr} [s^-1], "
    # )
    # print(params_string)
    return r_gr * x_sdw


def predict_x_nsdw(
    x_sdw: float,
    r_gr: float,
    f_phot: float,
    f_resp: float,
    c_ch2o: float = C_CH2O,
    c_yf: float = C_YF,
) -> float:
    """Dynamic model of non-structural dry weight.

    Nonlinear differential equation governing dynamic behavior of
    non-structural dry weight.

    Args:
        x_sdw: structural dry weight [g m^{-2}]
        r_gr: specific growth rate [s^{-1}]
        c_ch2o: conversion of assimilated CO_2 to CH_2O [g g^{-1}]
        c_yf: yield factor [g g^{-1}]
        f_phot: gross canopy photosynthesis [g m^{-2} s^{-1}]
        f_resp: maintenance respiration [g m^{-2} s^{-1}]

    Returns:
        dx_nsdw/dt - non-structural dry weight increment
        [g m^{-2} s^{-1}]

    References:
        Thornley & Hurd, 1974
    """
    # params_string = (
    #     f"predict_x_nsdw: "
    #     f"x_sdw (Structural Dry Weight) = {x_sdw} [g m^-2], "
    #     f"r_gr (Specific Growth Rate) = {r_gr} [s^-1], "
    #     f"f_phot (Gross Canopy Photosynthesis) = {f_phot} [g m^-2 s^-1], "
    #     f"f_resp (Maintenance Respiration) = {f_resp} [g m^-2 s^-1], "
    #     f"c_ch2o (Conversion of CO2 to CH2O) = {c_ch2o} [g g^-1], "
    #     f"c_yf (Yield Factor) = {c_yf} [g g^-1]",
    #     # c_ch2o * f_phot - r_gr * x_sdw - f_resp - ((1 - c_yf) / c_yf) * r_gr * x_sdw
    # )
    # print(params_string)
    return (
        c_ch2o * f_phot
        - r_gr * x_sdw
        - f_resp
        - ((1 - c_yf) / c_yf) * r_gr * x_sdw
    )


def get_r_gr(
    x_sdw: float,
    x_nsdw: float,
    u_T: float,
    c_gr_max: float = C_GR_MAX,
    c_gamma: float = C_GAMMA,
    c_Q10_gr: float = C_Q10_GR,
) -> float:
    """Specific growth rate.

    Args:
        x_sdw: structural dry weight [g m^{-2}]
        u_T: canopy temperature [C]
        x_nsdw: non-structural dry weight [g m^{-2}]
        c_gr_max: saturated growth rate at 20 C [s^{-1}]
        c_gamma: growth rate coefficient [-]
        c_Q10_gr: growth rate sensitivity to temperature [-]

    Returns:
        r_gr - specific growth rate [s^{-1}]

    References:
        Thornley & Hurd (1974)
    """
    # params_string = (
    #     f"x_sdw (Structural Dry Weight) = {x_sdw} [g m^-2], "
    #     f"x_nsdw (Non-Structural Dry Weight) = {x_nsdw} [g m^-2], "
    #     f"u_T (Canopy Temperature) = {u_T} [C], "
    #     f"c_gr_max (Saturated Growth Rate at 20 C) = {c_gr_max} [s^-1], "
    #     f"c_gamma (Growth Rate Coefficient) = {c_gamma}, "
    #     f"c_Q10_gr (Growth Rate Sensitivity to Temperature) = {c_Q10_gr}"
    # )
    # print(params_string)
    return (
        c_gr_max
        * (x_nsdw / (c_gamma * x_sdw + x_nsdw))
        * c_Q10_gr ** ((u_T - 20) / 10)
    )


def get_f_resp(
    x_sdw: float,
    u_T: float,
    c_resp_sht: float = C_RESP_SHT,
    c_tau: float = C_TAU,
    c_resp_rt: float = C_RESP_RT,
    c_Q10_resp: float = C_Q10_RESP,
) -> float:
    """Maintenance respiration rate.

    Args:
        x_sdw: structural dry weight [g m^{-2}]
        u_T: canopy temperature [C]
        c_resp_sht: shoot maintenance respiration rate at 25 C [s^{-1}]
        c_tau: root dry mass ratio [-]
        c_resp_rt: root maintenance respiration rate at 25 C [s^{-1}]
        c_Q10_resp: maintenance respiration sensitivity to temperature [-]

    Returns:
        f_resp - maintenance respiration rate [g m^{-2} s^{-1}]
    """
    # params_string = (
    #     f"get_f_resp: "
    #     f"x_sdw (Structural Dry Weight) = {x_sdw} [g m^-2], "
    #     f"u_T (Canopy Temperature) = {u_T} [C], "
    #     f"c_resp_sht (Shoot Maintenance Respiration Rate at 25 C) = {c_resp_sht} [s^-1], "
    #     f"c_tau (Root Dry Mass Ratio) = {c_tau}, "
    #     f"c_resp_rt (Root Maintenance Respiration Rate at 25 C) = {c_resp_rt} [s^-1], "
    #     f"c_Q10_resp (Maintenance Respiration Sensitivity to Temperature) = {c_Q10_resp}",
    #     (c_resp_sht * (1 - c_tau) * x_sdw + c_resp_rt * c_tau * x_sdw) * c_Q10_resp ** ((u_T - 25) / 10)
    # )
    # print(params_string)
    return (
        c_resp_sht * (1 - c_tau) * x_sdw + c_resp_rt * c_tau * x_sdw
    ) * c_Q10_resp ** ((u_T - 25) / 10)


def get_f_phot(
    x_sdw: float,
    f_phot_max: float,
    c_K: float = C_K,
    c_lar: float = C_LAR,
    c_tau: float = C_TAU,
) -> float:
    """Groos canopy photosynthesis.

    Args:
        x_sdw: structural dry weight [g m^{-2}]
        c_K: extinction coefficient [-]
        c_lar: structural leaf area ratio [g^{-1} m^{-2}]
        c_tau: root dry mass ratio [-]
        f_phot_max: gross CO_2 assimilation rate [g m^{-2} s^{-1}]

    Returns:
        f_phot - gross canopy photosynthesis [g m^{-2} s^{-1}]

    References:
        Goudriaan & Van Laar (1978) and Goudriaan & Monteith (1990)
    """
    return (1 - exp(-c_K * c_lar * (1 - c_tau) * x_sdw)) * f_phot_max


def get_f_phot_max(
    u_par: float,
    u_co2: float,
    epsilon: float,
    g_co2: float,
    gamma: float,
    c_omega: float = C_OMEGA,
) -> float:
    """Response of canopy photosynthesis.

    Args:
        u_par: incident photosynthetically active radiation [W m^{-2}]
        u_co2: CO_2 concentration [ppm]
        epsilon: light use efficiency [g J^{-1}]
        g_co2: canopy conductance to CO_2 diffusion [m s^{-1}]
        c_omega: density of CO_2 in air [g m^{-3}]
        gamma: CO_2 compensation point [g m^{-3}]

    Returns:
        f_phot_max - gross CO_2 assimilation rate [g m^{-2} s^{-1}]

    References:
        Acock et al. (1978)
    """
    return (epsilon * u_par * g_co2 * c_omega * (u_co2 - gamma)) / (
        epsilon * u_par + g_co2 * c_omega * (u_co2 - gamma)
    )


def get_ggamma(
    u_T: float,
    c_Gamma: float = C_GGAMMA,
    c_Q10_Gamma: float = C_Q10_GGAMMA,
) -> float:
    """CO_2 compensation point.

    Args:
        u_T: canopy temperature [C]
        c_Gamma (optional): CO_2 compensation points at 20 C [ppm]
        c_Q10_Gamma (optional): CO_2 compensation point sensitivity [-]

    Returns:
        Gamma - CO2 compensation point [ppm]

    References:
        Goudriaan et al. (1985)
    """
    return c_Gamma * c_Q10_Gamma ** ((u_T - 20) / 10)


def get_epsilon(
    u_co2: float,
    gamma: float,
    c_epsilon: float = C_EPSILON,
) -> float:
    """Light use efficency.

    Args:
        u_co2: CO_2 concentration [ppm]
        gamma: CO_2 compensation point [ppm]
        c_epsilon (optional): light use efficiency at high CO_2 [g J^{-1}]

    Returns:
        epsilon - light use efficiency [g J^{-1}]
    """
    return c_epsilon * ((u_co2 - gamma) / (u_co2 + 2 * gamma))


def get_g_co2(
    g_car: float,
    g_bdn: float = G_BND,
    g_stm: float = G_STM,
) -> float:
    """Canopy conductance to CO_2 diffusion.

    Args:
        g_car: carboxilation conductance [m s^{-1}]
        g_bdn (optional): boundary layer conductance [m s^{-1}]
        g_stm (optional): stomatal resistance [m s^{-1}]

    Returns:
        g_co2 - canopy conductance to CO_2 diffusion.
    """
    return 1 / ((1 / g_bdn) + (1 / g_stm) + (1 / g_car))


def get_g_car(
    u_T: float,
    c_car1: float = C_CAR1,
    c_car2: float = C_CAR2,
    c_car3: float = C_CAR3,
):
    """Carboxilation conductance.

    Args:
        u_T : canopy temperature [C]
        c_car1 (optional): carboxilation parameter [m s^{-1} C^{-2}]
        c_car2 (optional): carboxilation parameter [m s^{-1} C^{-1}]
        c_car3 (optional): carboxilation parameter [m s^{-1}]

    Returns:
       g_car - carboxilation conductance [m s^{-1}]
    """
    return c_car1 * u_T**2 + c_car2 * u_T + c_car3

## GSE model
def lamorturb(Gr, Re):
    free = Gr < 1e5
    Nu_G = 0.5 * free * Gr ** 0.25 + 0.13 * (1 - free) * Gr ** 0.33

    forced = Re < 2e4
    Nu_R = 0.6 * forced * Re ** 0.5 + 0.032 * (1 - forced) * Re ** 0.8

    x = Nu_G > Nu_R

    Nu = x * Nu_G + (1 - x) * Nu_R

    Sh = x * Nu * Le ** 0.25 + (1 - x) * Nu * Le ** 0.33

    return (Nu, Sh)


def convection(d, A, T1, T2, ias, rho, c, C):
    g = 9.81
    nu = 15.1e-6
    lam = 0.025

    Gr = (g * d ** 3) / (T1 * nu ** 2) * abs(T1 - T2)
    Re = ias * d / nu
    (Nu, Sh) = lamorturb(Gr, Re)

    QV_1_2 = A * Nu * lam * (T1 - T2) / d
    QP_1_2 = A * H_fg / (rho * c) * Sh / Le * lam / d * (C - sat_conc(T2))
    # QP_1_2 = 0

    return (QV_1_2, QP_1_2, Nu)


def radiation(eps_1, eps_2, rho_1, rho_2, F_1_2, F_2_1, A_1, T_1, T_2):
    sigm = 5.67e-8

    k = eps_1 * eps_2 / (1 - rho_1 * rho_2 * F_1_2 * F_2_1)
    QR_1_2 = k * sigm * A_1 * F_1_2 * (T_1 ** 4 - T_2 ** 4)

    return (QR_1_2)


def conduction(A, lam, l, T1, T2):
    QD_12 = (A * lam / l) * (T1 - T2)

    return (QD_12)


def T_ext(t):
    # Weather data

    climate = np.genfromtxt('climate.txt', delimiter=',')

    deltaT = 600
    n = int(np.ceil(t / deltaT))
    T_e = climate[n, 0] + T_k

    return (T_e)


def sat_conc(T):
    TC = T - T_k
    spec_hum = np.exp(11.56 - 4030 / (TC + 235))
    air_dens = -0.0046 * TC + 1.2978
    a = spec_hum * air_dens

    return a


def Cw_ext(t):
    # Weather data

    climate = np.genfromtxt('climate.txt', delimiter=',')

    deltaT = 600
    n = int(np.ceil(t / deltaT))
    RH_e = climate[n, 1] / 100;

    Cw_e = RH_e * sat_conc(T_ext(t))

    return (Cw_e)


def day(t):
    ## Day
    day_new = np.ceil(t / 86400)
    return (day_new)


def model(t, z, climate, daynum):
    # Values being calculated
    T_c = z[0]
    T_i = z[1]
    T_v = z[2]
    T_m = z[3]
    T_p = z[4]
    T_f = z[5]
    T_s1 = z[6]
    T_s2 = z[7]
    T_s3 = z[8]
    T_s4 = z[9]
    T_vmean = z[10]
    T_vsum = z[11]
    C_w = z[12]
    C_c = z[13]
    C_buf = z[14]
    C_fruit = z[15]
    C_leaf = z[16]
    C_stem = z[17]
    R_fruit = z[18]
    R_leaf = z[19]
    R_stem = z[20]
    x_sdw = z[21]
    x_nsdw = z[22]

    # External weather and dependent internal parameter values
    n = int(np.ceil(t / deltaT))  # count
    T_ext = climate[n, 0] + T_k  # External air temperature (K)
    wind_speed = climate[n, 2]  # External wind speed (m/s)
    p_w = C_w * R * T_i / M_w  # Partial pressure of water [Pa]
    rho_i = ((atm - p_w) * M_a + p_w * M_w) / (R * T_i)  # Internal density of air [kg/m^3]
    LAI = SLA * C_leaf  # Leaf area index
    C_ce = 4.0e-4 * M_c * atm / (R * T_ext)  # External carbon dioxide concentration [kg/m^3]
    h = 6.626e-34  # Planck's constant in Joule*Hz^{-1}

    daynum.append(day(t))  # Day number

    # Option for printing progress of run in days - uncomment out if needed
    if daynum[(len(daynum) - 1)] > daynum[(len(daynum) - 2)]:
        print('Day', daynum[len(daynum) - 1])

    hour = np.floor(t / 3600) + 1

    ## Lights
    L_on = 0  # No additional lighting included
    AL_on = 0  # No ambient lighting included

    ## Convection
    # Convection internal air -> cover
    (QV_i_c, QP_i_c, Nu_i_c) = convection(d_c, A_c, T_i, T_c, ias, rho_i, c_i, C_w)

    # Convection internal air -> floor
    (QV_i_f, QP_i_f, Nu_i_f) = convection(d_f, A_f, T_i, T_f, ias, rho_i, c_i, C_w)

    # Convection internal air -> vegetation
    A_v_exp = LAI * A_v
    (QV_i_v, QP_i_v, Nu_i_v) = convection(d_v, A_v_exp, T_i, T_v, ias, rho_i, c_i, C_w)

    # Convection internal air -> mat
    A_m_wool = 0.75 * A_m  # Area of mat exposed

    (QV_i_m, QP_i_m, Nu_i_m) = convection(d_m, A_m_wool, T_i, T_m, ias, rho_i, c_i, C_w)

    # Convection internal air -> tray
    (QV_i_p, QP_i_p, Nu_i_p) = convection(d_p, A_p, T_i, T_p, ias, rho_i, c_i, C_w)

    ## Ventilation
    # Leakage (equations for orifice flow from Awbi, Ventilation of Buildings, Chapter 3)
    wind_speed_H = wind_speed * c * H ** a  # Wind speed at height H
    wind_pressure = Cp * 0.5 * rho_i * wind_speed_H ** 2  # Equals DeltaP for wind pressure
    stack_pressure_diff = rho_i * g * H * (T_i - T_ext) / T_i  # DeltaP for stack pressure

    Qw = Cd * crack_area * (2 * wind_pressure / rho_i) ** 0.5  # Flow rate due to wind pressure
    Qs = Cd * crack_area * (2 * abs(stack_pressure_diff) / rho_i) ** 0.5  # Flow rate due to stack pressure
    Qt = (Qw ** 2 + Qs ** 2) ** 0.5  # Total flow rate

    total_air_flow = Qt * crack_length_total / crack_length
    R_a_min = total_air_flow / V

    # Ventilation
    DeltaT_vent = T_i - T_sp_vent
    comp_dtv_low = DeltaT_vent > 0 and DeltaT_vent < 4
    comp_dtv_high = DeltaT_vent >= 4
    R_a = R_a_min + comp_dtv_low * (R_a_max - R_a_min) / 4 * DeltaT_vent + comp_dtv_high * (R_a_max - R_a_min)

    QV_i_e = R_a * V * rho_i * c_i * (T_i - T_ext)  # Internal air to outside air [J/s]

    ## Solar radiation
    # We first define the solar elevation angle that determines that absorption of solar radiation. Notation: r is direct radiation, f is diffuse radiation, whilst VIS and NIR stand for visible and near infra-red respectively.

    gamma = np.deg2rad(360. * (day(t) - 80.) / 365.)  # Year angle [rad] --- day counts from January 1st
    eqn_time = -7.13 * np.cos(gamma) - 1.84 * np.sin(gamma) - 0.69 * np.cos(2. * gamma) + 9.92 * np.sin(
        2. * gamma)  # Equation of time [min]
    az = np.deg2rad(360. * ((t / (3600.) % 24.) + eqn_time / 60. - 12.) / 24.)  # Azimuth [rad]
    delta = np.deg2rad(0.38 - 0.77 * np.cos(gamma) + 23.27 * np.cos(gamma))  # Declination angle [rad]
    lat = np.deg2rad(latitude)
    angler = np.arcsin(
        np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(az))  # Angle of elevation [rad]
    angle = np.rad2deg(angler)

    # Radiation from artifical lighting
    QS_al_NIR = 0.  # no artificial lighting
    QS_al_VIS = 0.

    # Solar radiation incident on the cover
    QS_tot_rNIR = 0.5 * SurfaceArea @ climate[n, 4:12]  # Direct
    QS_tot_rVIS = 0.5 * SurfaceArea @ climate[n, 4:12]
    QS_tot_fNIR = 0.5 * SurfaceArea @ climate[n, 12:20]  # Diffuse
    QS_tot_fVIS = 0.5 * SurfaceArea @ climate[n, 12:20]

    # Transmitted solar radiation
    QS_int_rNIR = tau_c_NIR * QS_tot_rNIR  # J/s total inside greenhouse
    QS_int_rVIS = tau_c_VIS * QS_tot_rVIS
    QS_int_fNIR = tau_c_NIR * QS_tot_fNIR
    QS_int_fVIS = tau_c_VIS * QS_tot_fVIS

    # Solar radiation absorbed by the cover and the obstructions
    QS_i = a_obs * (QS_int_rNIR + QS_int_rVIS + QS_int_fNIR + QS_int_fVIS) # J/s

    # Solar radiation absorbed by the vegetation
    # Area = A_v i.e. planted area
    # factor QS by A_v/A_f

    k_fVIS = 0.85  # Visible diffuse extinction coefficient [-]
    a_v_fVIS = 0.95 - 0.9 * np.exp(-k_fVIS * LAI)  # Visible diffuse absorption coefficient [-]

    k_rVIS = 0.88 + 2.6 * np.exp(-0.18 * angle)  # Visible direct extinction coefficient [-]
    a_v_rVIS = 0.94 - 0.95 * np.exp(-k_rVIS * LAI)  # Visible direct absorption coefficient [-]

    QS_v_rVIS = (QS_int_rVIS * (1 - a_obs) + QS_al_VIS) * a_v_rVIS * A_v / A_f
    QS_v_fVIS = (QS_int_fVIS * (1 - a_obs)) * a_v_fVIS * A_v / A_f
    QS_v_VIS = (QS_v_rVIS + QS_v_fVIS)  # Used for photosynthesis calc

    # CO2 exchange with outside
    MC_i_e = (R_a * (C_c - C_ce))  # [kg/m^3/s]

    day_hour_c = (hour / 24 - np.floor(hour / 24)) * 24
    track = day_hour_c > 6 and day_hour_c < 20
    Value = added_CO2 / Nz / 3600. / V

    MC_cc_i = Value * track

    ## Photosynthesis model - Vanthoor

    # Consider photosynthetically active radiation to be visible radiation

    T_25 = T_k + 25.  # K

    I_VIS = QS_v_VIS  # J/s incident on planted area

    PAR = I_VIS / heat_phot / N_A / A_v

    # The number of moles of photosynthetically active photons per unit area of planted floor [mol{phot}/m^2/s]
    # J/s/(J/photon)/(photons/mol)/m^2 cf Vanthoor 2.3mumol(photons)/J

    Gamma = max((c_Gamma * (T_v - T_k) / LAI + 20 * c_Gamma * (1 - 1 / LAI)),
                0)  # The CO2 compensation point [mol{CO2}/mol{air}]
    k_switch = C_buf_max  # kg/m^2/s
    h_airbuf_buf = 1 / (1 + np.exp(s_airbuf_buf * (C_buf - k_switch)))

    C_c_molar = (C_c / rho_i) * (M_a / M_c)
    C_stom = eta * C_c_molar  # Stomatal CO2 concentration [mol{CO2}/mol{air}]

    J_pot = LAI * J_max_25 * np.exp(E_j * (T_v - T_25) / (R * T_v * T_25)) * (
                1 + np.exp((S * T_25 - HH) / (R * T_25))) / (
                        1 + np.exp((S * T_v - HH) / (R * T_v)))  # [mol{e}/m^2{floor}s]
    J = (J_pot + alph * PAR - ((J_pot + alph * PAR) ** 2 - 4 * theta * J_pot * alph * PAR) ** 0.5) / (2 * theta)
    P = J * (C_stom - Gamma) / (4 * (C_stom + 2 * Gamma))  # Photosynthesis rate [mol{CO2}/s]
    Resp = P * Gamma / C_stom  # Photorespiration rate

    MC_i_buf = (M_carb * h_airbuf_buf * (P - Resp))  # The net photosynthesis rate [kg{CH2O}/m^2/s]

    ## Crop growth model
    # Flow of carbohydrates from buffer to fruit, leaves and stem
    C_buf_min = 0.05 * C_buf_max
    h_buforg_buf = 1 / (1 + np.exp(s_buforg_buf * (C_buf - C_buf_min)))

    # inhibition terms need temperatures in oC
    h_T_v = 1 / (1 + np.exp(s_min_T * ((T_v - T_k) - T_min_v))) / (1 + np.exp(s_max_T * ((T_v - T_k) - T_max_v)))
    h_T_v24 = 1 / (1 + np.exp(s_min_T24 * ((T_vmean - T_k) - T_min_v24))) / (
                1 + np.exp(s_max_T24 * ((T_vmean - T_k) - T_max_v24)))

    h_T_vsum = 0.5 * (T_vsum / T_sum_end + ((T_vsum / T_sum_end) ** 2 + 1e-4) ** 0.5) - 0.5 * (
                ((T_vsum - T_sum_end) / T_sum_end) + (((T_vsum - T_sum_end) / T_sum_end) ** 2 + 1e-4) ** 0.5)

    g_T_v24 = 0.047 * (T_vmean - T_k) + 0.06

    MC_buf_fruit = (h_buforg_buf * h_T_v * h_T_v24 * h_T_vsum * g_T_v24 * rg_fruit)
    MC_buf_leaf = (h_buforg_buf * h_T_v24 * g_T_v24 * rg_leaf)
    MC_buf_stem = (h_buforg_buf * h_T_v24 * g_T_v24 * rg_stem)

    # Growth respiration, which is CO2 leaving the buffer
    MC_buf_i = c_fruit_g * MC_buf_fruit + c_leaf_g * MC_buf_leaf + c_stem_g * MC_buf_stem

    # Maintenance respiration
    MC_fruit_i = (c_fruit_m * Q_10 ** (0.1 * (T_vmean - T_25)) * C_fruit * (1 - np.exp(-c_RGR * R_fruit)))
    MC_leaf_i = (c_leaf_m * Q_10 ** (0.1 * (T_vmean - T_25)) * C_leaf * (1 - np.exp(-c_RGR * R_leaf)))
    MC_stem_i = (c_stem_m * Q_10 ** (0.1 * (T_vmean - T_25)) * C_stem * (1 - np.exp(-c_RGR * R_stem)))

    ## ODE equations
    # Heater control logic
    heater_switch_temp = 28.0 + T_k  # Adjust this temperature threshold
    Q_heating = 0.0  # Default: heater is off
    if T_i < heater_switch_temp:
        Q_heating = 10000.0  # Adjust this value: power when the heater is on

    # Temperature components
    dT_i_dt = (1 / (V * rho_i * c_i)) * (-QV_i_m - QV_i_v - QV_i_f - QV_i_c - QV_i_e - QV_i_p + QS_i + Q_heating)

    # Carbon Dioxide
    dC_c_dt = MC_cc_i - MC_i_e + (M_c / M_carb) * (A_v / V) * (MC_buf_i + MC_fruit_i + MC_leaf_i + MC_stem_i - MC_i_buf)

    # Salaatia growth
    cLight = 3.0e8  # Speed of light [m/s]
    lambda_nm = 550  # Wavelength [nm]
    lambda_m = lambda_nm * 1e-9
    E = h * cLight / lambda_m
    u_par = PAR * E # [W/m^2]
    u_co2 = dC_c_dt * R * (T_i + T_k) / (M_c * atm) * 1.0e6
    dx_sdw_dt, dx_nsdw_dt = lettuce_growth_model(t, (x_sdw, x_nsdw), (T_i - T_k, u_par, u_co2))
    # print("T_i, u_par, u_co2", x_nsdw)
    return np.array([0,dT_i_dt,0,0,0,0,0,0,0,0,0,0,0,dC_c_dt,0,0,0,0,0,0,0,dx_sdw_dt,dx_nsdw_dt])
