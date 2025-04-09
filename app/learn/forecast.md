Forecasting plays a crucial role in the effectiveness of Model Predictive Control (MPC) for greenhouse climate regulation. By incorporating predictions of future conditions, the controller can make proactive decisions rather than merely reacting to changes.

### Using the Location Settings

- **City**: Enter your city name to fetch geographic coordinates
- **Date and Time**: Set when your simulation should start

After clicking "Fetch Forecast", the app will:

1. Retrieve your location's coordinates, timezone, and altitude
2. Get the current carbon intensity value for electricity in your region
3. Download weather forecast data for your simulation period

### External Climate Variables

The forecast chart shows key variables that affect your greenhouse:

- **Temperature**: External air temperature affects heating/cooling needs
- **Solar Radiation**: Drives photosynthesis and internal heating
- **Wind Speed**: Influences heat exchange through the greenhouse cover
- **Humidity**: Affects plant transpiration and internal moisture levels

These external conditions are modeled as time-varying parameters $ p(k) $ in the system dynamics equation:

$$ x(k+1) = f(x(k), u(k), p(k)) $$

Where $ x(k) $ is the state vector, $ u(k) $ is the control input vector, and $ p(k) $ represents the external climate conditions.

### Solar Radiation Modeling

Solar radiation is vital for plant growth and energy accumulation in greenhouses. The app models:

- **Global Horizontal Irradiance (GHI)**: Total solar radiation on a horizontal surface
- **Photosynthetically Active Radiation (PAR)**: The portion of radiation used for photosynthesis

For different orientations and tilts of the greenhouse surfaces, the model calculates:

- Direct radiation component
- Diffuse radiation component
- Reflected radiation from surrounding surfaces

These calculations use your greenhouse's orientation, location, and time to provide realistic solar inputs.

### Carbon Intensity Calculation

The carbon intensity value (gCO₂eq/kWh) represents how "clean" the electricity is in your region. This is incorporated into the economic calculations through:

$ E_{\text{CO}_2}(u) = \frac{I_{\text{CO}_2} \Delta t}{1000 \times 3600} P(u) $

Where:

- $ I_{\text{CO}_2} $ is the carbon intensity
- $ \Delta t $ is the time step
- $ P(u) $ is the power consumption based on control actions

The associated cost is calculated as:
$ C_{\text{CO}_2}(u) = C_{\text{CO}_2\text{cost}} E_{\text{CO}_2}(u) $

Where $ C_{\text{CO}_2\text{cost}} $ is the social cost of carbon.

### Proactive vs. Reactive Control

Traditional controllers react to changes after they occur. By incorporating forecasts, the MPC:

1. **Anticipates Changes**: Prepares for upcoming weather shifts proactively
2. **Optimizes Energy Use**: Heats or cools in advance of major temperature changes
3. **Reduces Costs**: Schedules energy-intensive operations during lower-price periods

This predictive capability is particularly valuable for greenhouse climate control, as it allows the system to prepare for significant weather changes before they occur.

### Tips for Effective Use

- **Start Date Selection**: Choose dates with interesting weather patterns to see how the controller adapts
- **Location Choice**: Try different climate zones to compare controller performance
- **Forecast Length**: Longer simulations show more interesting adaptive behavior

### Next Steps

After setting your location and fetching forecast data:

1. Review the weather forecast charts
2. Adjust Climate Controls if needed (optional)
3. Proceed to eMPC Design to start your simulation
