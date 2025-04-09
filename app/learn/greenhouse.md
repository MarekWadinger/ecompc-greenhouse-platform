A greenhouse is a structure designed to create an optimal environment for plant growth by controlling temperature, humidity, CO₂ levels, and light exposure. In this app, you can design your own greenhouse and explore how its physical properties affect climate control and plant growth.

### Shape Parameters

- **Length and Width**: Determine the floor area of your greenhouse
- **Wall Height**: Affects the internal volume and heat exchange surface
- **Roof Tilt**: Impacts solar radiation capture (higher angles are better for winter sun)

### Orientation

The **Azimuth** setting determines which direction your greenhouse faces:

- 0° = North-facing
- 90° = East-facing
- 180° = South-facing (optimal in Northern hemisphere)
- 270° = West-facing

Proper orientation maximizes sunlight exposure, which directly influences photosynthesis and energy balance.

### Mathematical Model

Behind the app's visualization is a detailed physical model representing the greenhouse climate dynamics. The model includes:

#### Energy Balance

The temperature dynamics in different greenhouse compartments (cover, internal air, vegetation, tray, etc.) are governed by heat transfer principles:

- **Convective Heat Transfer**:
  $ Q_{\text{conv}} = A_{\text{c}}\, \text{Nu}\, \lambda_{\text{air}} \frac{T_1 - T_2}{d_{\text{c}}} $

- **Radiative Heat Transfer**:
  $ Q_{\text{rad}} = \frac{\varepsilon_1 \varepsilon_2}{1 - \rho_1 \rho_2 F_{12} F_{21}} \sigma A_1 F_{12} (T_1^4 - T_2^4) $

- **Conductive Heat Transfer**:
  $ Q_{\text{cond}} = \frac{A \lambda_{\text{c}}}{d_\text{l}} (T_1 - T_2) $

Where $ T_1 $ and $ T_2 $ represent temperatures, $ A $ is the surface area, $ \lambda $ is thermal conductivity, and $ d $ represents characteristic lengths.

#### Mass Balance for Humidity and CO₂

The app also models water vapor and CO₂ concentrations, which are critical for plant growth:

- **External CO₂ Concentration**:
  $ C_{\text{ext}} = \frac{4 \times 10^{-4} M_c P_{\text{atm}}}{R T_{\text{ext}}} $

- **Internal CO₂ Concentration in ppm**:
  $ C_{\text{int, ppm}} = \frac{C_c R T_i}{M_c P_{\text{atm}}} \times 10^6 $

### Tips for Optimal Design

- **For Temperate Climates**: Use a steeper roof tilt (25-35°) to optimize solar capture
- **For Hot Climates**: Consider a lower roof tilt to reduce overheating
- **For Year-Round Growing**: In the Northern hemisphere, a South-facing orientation (180°) is usually optimal

### How These Parameters Affect Control

When you adjust the greenhouse parameters, you're changing fundamental properties that affect:

1. **Surface Area to Volume Ratio**: Influences heat loss and energy efficiency
2. **Solar Radiation Capture**: Affects natural heating and lighting for photosynthesis
3. **Air Movement Patterns**: Impacts temperature distribution and CO₂ circulation
4. **Actuator Scaling**: Changes the required capacity of heating, ventilation, and humidification systems

The model integrates these parameters into a dynamic system with state vector $x$ = [$T_c, T_i, T_v, T_m, T_p, T_f, T_s, C_w, C_c, x_{\text{sdw}}, x_{\text{nsdw}}$], representing temperatures of different compartments, humidity, CO₂ concentration, and plant biomass.

### Next Steps

After defining your greenhouse structure:

1. Click "Build Greenhouse" to save your design
2. Move to the "Location" section to set up weather forecasting
3. The visualization will update to show your greenhouse design
