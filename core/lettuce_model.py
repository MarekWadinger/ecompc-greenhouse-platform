"""Lettuce Growth Model"""
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
def lettuce_growth_model(
    _: int, x: tuple[float, float], u: tuple[float, float, float]
) -> list[float]:
    """Overall dynamic growth model.

    Args:
        _: Elapsed time for solve_ivp compatibility [s]
        x (tuple): System states [x_sdw, x_nsdw]
        u (tuple): System inputs [u_T, u_par, u_co2]

    Returns:
        [dx_sdw/dt, dx_nsdw/dt] - structural and non-structural dry weight
        increments [g m^{-2} s^{-1}]

    Examples:
    >>> lettuce_growth_model(1, (10, 10), (25, 0, 400))
    [2.8772827989339694e-05, -3.9089534986674615e-05]
    """
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
    return [dx_sdw_dt, dx_nsdw_dt]


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
