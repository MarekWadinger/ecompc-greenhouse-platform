Model Predictive Control (MPC) is an advanced control technique that uses a mathematical model of a system to predict future behavior and optimize control actions accordingly. This app implements Nonlinear Economic MPC (NEMPC), which directly optimizes economic performance rather than just tracking setpoints.

### MPC Parameters

- **Lettuce Price**: The market price for lettuce (EUR/kg) - affects how aggressively the controller prioritizes growth
- **Seeds Weight**: Initial planting density - higher values lead to more potential yield
- **Sampling Time**: How frequently the controller updates (seconds) - lower values give more precise control
- **Simulation Steps**: Total duration of your simulation - longer runs show more complete growth cycles
- **Prediction Horizon**: How far ahead the controller looks - longer horizons may improve decisions but increase complexity
- **Control Input Range**: Limits on actuation percentage - narrower ranges can save energy

### Theoretical Foundations

#### System Dynamics

The greenhouse system is modeled as a set of nonlinear differential equations:

$$ x(k+1) = f(x(k), u(k), p(k)) $$

Where:

- $ x(k) \in \mathbb{R}^{n_x} $ is the state vector, including temperatures, humidity, CO₂ concentration, and plant biomass
- $ u(k) \in \mathbb{R}^{n_u} $ is the control input vector for the actuators (heating, ventilation, humidification, CO₂ enrichment)
- $ p(k) $ represents time-varying parameters like external climate conditions

#### Economic Objective Function

The goal of NEMPC is to maximize the profit from lettuce production while minimizing operating costs. The objective function balances:

$$ \min_{{\{u(k)\}}_{k=0}^{N-1}} \sum_{k=0}^{N-1} (-R(k) + C_u(k)) $$

Where:

- $ R(k) = \frac{P_L A_c}{\rho_{dw}} \sum_{i\in \{ \text{sdw}, \text{nsdw} \}}(x_i(k) - x_i(0)) $ is the revenue from biomass accumulation
- $ C_u(k) = \sum_{i} C_{\text{energy}}(u_i(k)) + C_{\text{CO}_2}(u_i(k)) $ is the actuating cost
- $ P_L $ is the lettuce price
- $ A_c $ is the cultivated area
- $ \rho_{dw} $ is the dry-to-wet ratio

#### Constraints

The optimization is subject to system dynamics and operational constraints:

- $ u_{\min} \leq u(k) \leq u_{\max} $: Actuator limits
- $ x_{\min} \leq x(k) \leq x_{\max} $: State variable boundaries
- $ x(0) = x_{\text{initial}} $: Initial conditions

### How the Controller Works

At each time step, the controller:

1. Takes current greenhouse measurements
2. Looks ahead using weather forecasts and plant growth models
3. Finds the optimal balance of heating, ventilation, humidity, and CO₂
4. Applies the first set of calculated actions
5. Repeats the process with updated measurements

The controller balances three competing objectives:

1. **Revenue**: Maximizing lettuce growth and yield
2. **Costs**: Minimizing energy, water, and CO₂ usage
3. **Environmental Impact**: Considering carbon emissions

### Understanding the Results

After simulation, you'll see:

- **Growth Charts**: Track lettuce biomass increase over time
- **Climate Variables**: Monitor temperature, humidity, CO₂ levels
- **Control Actions**: See how actuators are used over time
- **Financial Summary**: Detailed breakdown of costs and revenue

### Why MPC is Ideal for Greenhouse Control

MPC is particularly well-suited for greenhouse climate control for several reasons:

1. **Complex Dynamics**: Handles the nonlinear interactions between temperature, humidity, CO₂, and plant growth
2. **Multiple Objectives**: Balances economic and environmental goals
3. **Predictive Capability**: Anticipates changes in weather and energy costs
4. **Constraint Handling**: Respects operational limitations of actuators and safety conditions

### Tips for Better Results

- **Adjust Prediction Horizon**: Longer horizons (10-30) often produce better results but take longer to compute
- **Balance Control Inputs**: Restricting the maximum input range can save energy costs
- **Experiment with Planting Density**: Higher initial biomass can lead to faster growth rates
- **Try Different Climates**: Compare results across various locations and seasons

The simulation results help you understand the complex tradeoffs between crop yield, energy use, and environmental impact.

### Interpreting Controller Behavior

- **Heating Usage**: Often highest during night or cold periods
- **CO₂ Enrichment**: Typically scheduled during daylight hours when photosynthesis is active
- **Ventilation**: Used to manage temperature and humidity, often increasing on hot days
- **Humidification**: Applied when air is too dry for optimal plant growth
