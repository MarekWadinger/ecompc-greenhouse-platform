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
# greenhouse_model.py
#   contains parameter values for fundamental constants, greenhouse construction and operation and plant geometry and growth.
# SampleWeather.csv
#   Hourly input weather data, in the format 'Hour, Ambient Temperature ($^o$C), Sky Temperature ($^o$C), Windspeed (m/s),
#   Relative Humidity (%), Direct Solar Radiation (order NE wall, NE Roof, SE Wall, SE Roof, SW Wall, SW Roof, NW Wall, NW Roof)
#   (W/m$^2$), Diffuse Solar Radiation (order as for Direct)(W/m$^2$)

#######################################################################################################################################

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from simpleParameters import *

sys.path.insert(1, str(Path().resolve()))
from core.greenhouse_model import M_c, R, T_k, atm, deltaT, model  # noqa: E402

## Specify initial conditions**

# Temperatures
T_c_0 = 20.0 + T_k  # Cover temperature [K]
T_i_0 = 12.0 + T_k  # Internal air temperature [K]
T_v_0 = 12.0 + T_k  # Vegetation temperature [K]
T_m_0 = 12.0 + T_k  # Growing medium temperature [K]
T_p_0 = 12.0 + T_k  # Tray temperature [K]
T_f_0 = 12.0 + T_k  # Floor temperature [K]
T_s1_0 = 12.0 + T_k  # Temperature of soil layer 1 [K]
T_s2_0 = 12.0 + T_k  # Temperature of soil layer 2 [K]
T_s3_0 = 12.0 + T_k  # Temperature of soil layer 3 [K]
T_s4_0 = 11.0 + T_k  # Temperature of soil layer 4 [K]
#
T_vmean_0 = 12.0 + T_k  # 24 hour mean vegetation temperature [K]
T_vsum_0 = 0  # Vegetation temperature sum [degC]
#
C_w_0 = 0.0085  # Density of water vapour [kg/m^3]
C_c_0 = 7.5869e-4  # CO_2 density
#
C_buf_0 = 0.01  # Mass of carbohydrate in buffer per unit per unit area of cultivated floor [kg/m^2]
C_fruit_0 = 0.001  # Mass of carbohydrate in fruit per unit per unit area of cultivated floor [kg/m^2]
C_leaf_0 = 0.01  # Mass of carbohydrate in leaves per unit per unit area of cultivated floor [kg/m^2]
C_stem_0 = 0.01  # Mass of carbohydrate in stem per unit per unit area of cultivated floor [kg/m^2]
R_fruit_0 = 0.0  # Relative growth rate of fruit averaged over 5 days [1/s]
R_leaf_0 = 0.0  # Relative growth rate of leaf averaged over 5 days [1/s]
R_stem_0 = 0.0  # Relative growth rate of stem averaged over 5 days [1/s]

x_sdw  = 0.5  # Structural dry weight of the plant [kg/m^2]
x_nsdw = 2.8  # Non-structural dry weight of the plant [kg/m^2]

z = [
    T_c_0,
    T_i_0,
    T_v_0,
    T_m_0,
    T_p_0,
    T_f_0,
    T_s1_0,
    T_s2_0,
    T_s3_0,
    T_s4_0,
    T_vmean_0,
    T_vsum_0,
    C_w_0,
    C_c_0,
    C_buf_0,
    C_fruit_0,
    C_leaf_0,
    C_stem_0,
    R_fruit_0,
    R_leaf_0,
    R_stem_0,
    x_sdw,
    x_nsdw
]

daynum = [0]

## Interpolate weather data

if __name__ == "__main__":
    climdat = np.genfromtxt(
        "SampleWeather.csv", delimiter=","
    )  # Hourly data

    len_climdat = len(climdat)
    mult = np.linspace(1, len_climdat, int((len_climdat - 1) * 3600 / deltaT))
    y_interp = interp1d(climdat[:, 0], climdat[:, 1:21], axis=0)

    climate = y_interp(mult)

    ## Simulate over time

    tic = time.time()

    sim_days = 10  # Number of days of simulation
    tf = 86400 * sim_days  # Time in seconds
    t = [0, tf]
    tval = np.linspace(0, tf, tf + 1)

    # Use solve_ivp with 'BDF' stiff solver to solve the ODEs
    params = [climate, daynum]

    output = solve_ivp(
        model, t, z, method="BDF", t_eval=tval, rtol=1e-5, args=params
    )

    # Time simulation and print time taken
    toc = time.time()

    xt = toc - tic

    print("Runtime(s) = ", f"{xt:.3}")

    ## Plot results

    Tout_i = np.transpose(output.y[1,:]-T_k) # [°C]
    Ccout = np.transpose(output.y[13,:])

    ## Temperatures

    time = output.t/(3600*24) # Time in days
    resolution_value = 1200

    ## Internal air

    fig1, ax = plt.subplots()
    ax.plot(time,Tout_i, color='b', label = 'Internal air')
    ax.set_title('Internal air and vegetation temperatures')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Day')
    ax.set_ylabel('Temperature ($^o$C)')
    plt.savefig('temp.png', format="png", dpi=resolution_value)

    ## CO_2

    Ccout_ppm = Ccout*R*(Tout_i+T_k)/(M_c*atm)*1.e6
    fig6, ax4 = plt.subplots()
    ax4.plot(time,Ccout_ppm, color='b')
    ax4.set_title('CO$_2$')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('CO$_2$ (ppm)')
    plt.savefig('co2.png', format="png", dpi=resolution_value)


    ## Salaattia

    dx_sdw_dt = np.transpose(output.y[21,:])

    fig7, ax5 = plt.subplots()
    ax5.plot(time,dx_sdw_dt, color='b')
    ax5.set_title('kapusta')
    ax5.set_xlabel('Day')
    ax5.set_ylabel('-')
    plt.savefig('salat.png', format="png", dpi=resolution_value)

    dx_nsdw_dt = np.transpose(output.y[22,:])

    fig8, ax6 = plt.subplots()
    ax6.plot(time,dx_nsdw_dt, color='g')
    ax6.set_title('kapusta')
    ax6.set_xlabel('Day')
    ax6.set_ylabel('-')
    plt.savefig('salat2.png', format="png", dpi=resolution_value)
