from typing import Literal, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from cycler import cycler
from plotly.subplots import make_subplots

pd.options.plotting.backend = "plotly"

line_styles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
    (0, (3, 5, 1, 5)),
    (0, (3, 10, 1, 10)),
    (0, (5, 1)),
    (0, (5, 10)),
]

# Combine default color cycle with the line styles
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
line_cycler = cycler("color", colors) + cycler("linestyle", line_styles)

# Apply the combined cycler to the axes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.serif": "Computer Modern",
    "axes.prop_cycle": line_cycler,
    "axes.grid": True,
    "figure.subplot.left": 0.1,
    "figure.subplot.bottom": 0.05,
    "figure.subplot.right": 0.95,
    "figure.subplot.top": 0.95,
})


# Configure default datetime formatting for plots
# Set up the default date formatter globally
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats = [
    "%y",  # ticks are mostly years
    "%b",  # ticks are mostly months
    "%d",  # ticks are mostly days
    "%H:%M",  # hrs
    "%H:%M",  # min
    "%S.%f",  # secs
]
# these are mostly just the level above...
formatter.zero_formats = [""] + formatter.formats[:-1]
# ...except for ticks that are mostly hours, then it is nice to have month-day:
formatter.zero_formats[3] = "%d-%b"

formatter.offset_formats = [
    "",
    "%Y",
    "%b %Y",
    "%d %b %Y",
    "%d %b %Y",
    "%d %b %Y %H:%M",
]

# Apply globally to matplotlib
plt.rcParams["date.autoformatter.year"] = formatter.formats[0]
plt.rcParams["date.autoformatter.month"] = formatter.formats[1]
plt.rcParams["date.autoformatter.day"] = formatter.formats[2]
plt.rcParams["date.autoformatter.hour"] = formatter.formats[3]
plt.rcParams["date.autoformatter.minute"] = formatter.formats[4]
plt.rcParams["date.autoformatter.second"] = formatter.formats[5]


# Helper function to apply to specific axes if needed
def configure_date_formatter(ax):
    """Configure concise date formatter for the given axis."""
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


plotly_colors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def set_size(
    width: float
    | int
    | Literal["article", "ieee", "ieee_full", "thesis", "beamer"] = "ieee",
    fraction=1.0,
    subplots=(1, 1),
    return_size: Literal["matplotlib", "plotly"] = "matplotlib",
):
    """Set figure dimensions to avoid scaling in LaTeX.

    By default, 1 pt ≈ 1.33 px, meaning an 8 pt font in LaTeX is roughly 10.67 px in Plotly.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the height which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "article":
        width_pt = 390.0
    elif width == "ieee":
        width_pt = 252.0
    elif width == "ieee_full":
        width_pt = 252.0 * 2
    elif width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Width of figure (in pts)
    fig_width_pt = width_pt
    fig_height_pt = (
        fig_width_pt * golden_ratio * ((subplots[0] * fraction) / subplots[1])
    )

    if return_size == "matplotlib":
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_height_pt * inches_per_pt

        return (fig_width_in, fig_height_in)

    return (fig_width_pt * 1.33, fig_height_pt * 1.33)


def set_plotly_defaults_for_latex():
    """
    Configure Plotly defaults to generate plots that are style-compliant with LaTeX documents.
    This includes setting appropriate fonts, sizes, and styling for academic publications.
    """
    import plotly.io as pio

    # Get default figure dimensions for academic papers
    default_width, default_height = set_size(
        width="ieee", return_size="plotly"
    )

    # Define LaTeX-friendly template based on plotly white
    pio.templates["latex"] = pio.templates["plotly_white"]

    # Update the template with LaTeX-friendly settings
    pio.templates["latex"].layout.update(
        font=dict(
            family="Computer Modern",  # LaTeX default font
            size=8 * 1.33,  # Reasonable size for documents
            color="black",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=0, t=20, b=0),
        legend=dict(
            font=dict(size=8 * 1.33),
            # borderwidth=0.5,
            # bordercolor="black",
            itemsizing="constant",
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,  # Position above the plot
            xanchor="right",
            x=1,
            itemwidth=30,
            tracegroupgap=0,
            traceorder="normal",
        ),
        # legend2 configuration for secondary legend
        legend2=dict(
            font=dict(size=8 * 1.33),
            # borderwidth=0.5,
            # bordercolor="black",
            itemsizing="constant",
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.4,  # Position below the plot
            xanchor="right",
            x=1,
            itemwidth=30,
            tracegroupgap=0,
            traceorder="normal",
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            linewidth=1,
            linecolor="black",
            mirror=False,
            ticks="outside",
            showline=True,
            zeroline=False,
            title=dict(font=dict(size=8 * 1.33)),
            tickfont=dict(size=8 * 1.33),
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            linewidth=1,
            linecolor="black",
            mirror=False,
            ticks="outside",
            showline=True,
            zeroline=False,
            title=dict(font=dict(size=8 * 1.33)),
            tickfont=dict(size=8 * 1.33),
        ),
        width=default_width,  # Convert inches to points
        height=default_height * 2,  # Convert inches to points
        title=dict(font=dict(size=8 * 1.33)),
        annotations=[dict(font=dict(size=8 * 1.33))],
        hovermode="x unified",
    )

    # Set the default template
    pio.templates.default = "latex"


# Initialize the LaTeX-friendly defaults
set_plotly_defaults_for_latex()


def format_str_(str_: str) -> str:
    r"""Format a string with underscores to be used as a label in a plot.

    Args:
        str_: String with underscores

    Returns:
        str: String with underscores and Latex subscripts

    Examples:
    >>> format_str_("x_sdw")
    'x_{\\mathrm{sdw}}'
    """
    str_splitted = str_.split("_")
    # Split the string at underscores and join it with "_{"
    result = r"_{\mathrm{".join(str_splitted)
    result += "}}" * (len(str_splitted) - 1)
    return result


def plot_response(
    t_out: list | pd.DatetimeIndex,
    y_out: list | pd.DataFrame,
    u_out: list | pd.DataFrame,
    y_ref: Union[list, None] = None,
    u_min: Union[list[float], None] = None,
    u_max: Union[list[float], None] = None,
    axs_: Union[np.ndarray, None] = None,
    y_legend: Union[list[str], None] = None,
    u_legend: Union[list[str], None] = None,
    y_label: Union[str, None] = None,
    u_label: Union[str, None] = None,
    t_label: Union[str, None] = None,
) -> np.ndarray:
    if axs_ is None:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=set_size(width="ieee", subplots=(2, 1)),
        )
    else:
        axs = axs_
        fig = axs[0].get_figure()
    axs[0].plot(t_out, y_out, label=y_legend)
    if y_ref is not None:
        axs[0]._get_lines.set_prop_cycle(None)
        axs[0].plot(t_out, y_ref, label=f"{y_label} (ref)", linestyle=":")
    if axs_ is None:
        axs[0].set_ylabel(y_label)
        axs[0].legend()

    if isinstance(u_out, pd.DataFrame):
        # Sort u_out columns based on maximum values in descending order
        max_values = u_out.max()
        sorted_columns = max_values.sort_values(ascending=False).index
        u_out = u_out[sorted_columns]
        axs[1].plot(
            t_out,
            u_out,
            label=u_out.columns,
        )
    else:
        axs[1].plot(
            t_out,
            u_out,
            label=u_legend,
        )
    if u_min is not None and u_max is not None:
        axs[1]._get_lines.set_prop_cycle(None)
        for u_min_, u_max_ in zip(u_min, u_max):
            color = axs[1]._get_lines.get_next_color()
            axs[1].axhline(u_min_, color=color, linestyle=":")
            axs[1].axhline(u_max_, color=color, linestyle=":")
        axs[1]._get_lines.set_prop_cycle(None)

    if axs_ is None:
        axs[1].set_xlabel(t_label)
        axs[1].set_ylabel(u_label)
        axs[1].legend()
        axs[1].tick_params(axis="x")

    for ax in axs:
        configure_date_formatter(ax)
    fig.align_ylabels()
    fig.tight_layout()
    # Place legend above each subplot with proper spacing
    for i, ax in enumerate(axs):
        # Get the current legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Remove the current legend
            if ax.get_legend():
                ax.get_legend().remove()

            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=2,
                frameon=False,
            )

    # Adjust figure layout to accommodate legends
    plt.subplots_adjust(hspace=0.32)

    return axs


def plot_states(
    df: pd.DataFrame,
    axs: np.ndarray,
    set_ylabel: bool = True,
    exclude: list[str] = [],
):
    i = 0
    for column in df.columns:
        if df[column].dtype == "float64":
            if not isinstance(column, str) or (
                isinstance(column, str)
                and not any([excl in column for excl in exclude])
            ):
                axs[i].plot(df[column])
                if set_ylabel:
                    axs[i].set_ylabel(f"${format_str_(column)}$")
                i += 1
    return axs


def plotly_response(
    _timestamps,
    y_nexts,
    u0s,
    u_min,
    u_max,
    y_visible: list[int] = [-1, -2],
    u_labels: list[str] | None = None,
):
    def add_bound_trace(_timestamps, u, fig: go.Figure, label, color):
        # TODO: write issue to plotly to fix showlegend=False behavior on add_hline
        kwargs = dict(
            name=label,
            line=dict(color=color, dash="dash"),
            legendgroup=label,
            showlegend=False,
        )

        if u.shape[0] == 1:
            # TODO: write issue to plotly to fix typing
            fig.add_hline(
                y=u[0],
                row=2,  # type: ignore
                col=1,  # type: ignore
                **kwargs,  # type: ignore
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=_timestamps,
                    y=u,
                    mode="lines",
                    **kwargs,
                ),
                row=2,
                col=1,
            )
        return fig

    y_visible = [y if y >= 0 else y + y_nexts.shape[1] for y in y_visible]
    # y_nexts_ = np.array(y_nexts).squeeze()
    u_min = np.array(u_min, ndmin=2)
    u_max = np.array(u_max, ndmin=2)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        # subplot_titles=("Lettuce Dry Weight (g)", "Actuation [%]"),
    )

    # Add each column from the DataFrame as a separate trace
    for i, column in enumerate(y_nexts.columns):
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=y_nexts[column],
                name=column,
                visible=True if i in y_visible else "legendonly",
            ),
            row=1,
            col=1,
        )

    # Plot Actuation
    if u_labels is None:
        if isinstance(u0s, pd.DataFrame):
            u_labels_ = u0s.columns
        else:
            u_labels_ = [f"u_{i}" for i in range(u0s.shape[1])]
    else:
        u_labels_ = u_labels

    for i, (label, color) in enumerate(zip(u_labels_, plotly_colors)):
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=u0s[label],
                mode="lines",
                name=label,
                line=dict(color=color),
                legend="legend2",
                legendgroup=label,
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        if u_min.shape[1] == 1:
            if i == 0:
                fig = add_bound_trace(
                    _timestamps, u_min[:, i], fig, None, "gray"
                )
                fig = add_bound_trace(
                    _timestamps, u_max[:, i], fig, None, "gray"
                )
        else:
            fig = add_bound_trace(_timestamps, u_min[:, i], fig, label, color)
            fig = add_bound_trace(_timestamps, u_max[:, i], fig, label, color)

    fig.update_yaxes(title_text="Dry Weight (g/m$^2$)", row=1, col=1)
    fig.update_yaxes(title_text="Actuation (%)", row=2, col=1)

    return fig


def plotly_greenhouse(
    length: float,
    width: float,
    height: float,
    roof_tilt: float,
    azimuth: float,
):
    """
    Create a 3D visualization of a greenhouse with a compass on the ground plane.

    Args:
        length: Length of the greenhouse in meters
        width: Width of the greenhouse in meters
        height: Height of the greenhouse walls in meters
        roof_tilt: Angle of the roof in degrees
        azimuth: Orientation angle in degrees (0 = North, 90 = East, etc.)
        fig: Plotly figure object to add the visualization to
    """
    fig = go.Figure()

    # Create the greenhouse
    plotly_3d_greenhouse(length, width, height, roof_tilt, azimuth, fig)

    # Add the compass
    plotly_3d_compass(length, width, azimuth, fig)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        autosize=True,  # Make the figure responsive to window size
        scene=dict(
            aspectmode="data",  # Ensure the aspect ratio is based on data
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False),
        ),
    )

    return fig


def plotly_3d_greenhouse(
    length: float,
    width: float,
    height: float,
    roof_tilt: float,
    azimuth: float,
    fig: go.Figure,
):
    """
    Create a 3D visualization of a greenhouse with specified dimensions and orientation.

    Args:
        length: Length of the greenhouse in meters
        width: Width of the greenhouse in meters
        height: Height of the greenhouse walls in meters
        roof_tilt: Angle of the roof in degrees
        azimuth: Orientation angle in degrees (0 = North, 90 = East, etc.)
        fig: Plotly figure object to add the greenhouse to
    """
    # Vertices of the greenhouse base (rectangle)
    base = np.array(
        [
            [0, 0, 0],
            [length, 0, 0],
            [length, width, 0],
            [0, width, 0],
            [0, 0, 0],
        ]  # Closed base
    )

    # Roof vertices
    peak_height = height + np.tan(np.radians(roof_tilt)) * (width / 2)
    roof = np.array([
        [0, 0, height],
        [length, 0, height],
        [length, width, height],
        [0, width, height],
        [0, width / 2, peak_height],
        [length, width / 2, peak_height],
    ])

    # Center the greenhouse
    center_x = length / 2
    center_y = width / 2

    # Adjust all vertices to be centered
    base[:, 0] -= center_x
    base[:, 1] -= center_y
    roof[:, 0] -= center_x
    roof[:, 1] -= center_y

    # Rotate the greenhouse according to azimuth
    azimuth_radians = np.radians(azimuth)
    rotation_matrix = np.array([
        [np.cos(azimuth_radians), -np.sin(azimuth_radians)],
        [np.sin(azimuth_radians), np.cos(azimuth_radians)],
    ])

    # Apply rotation to base and roof
    for i in range(len(base)):
        base[i, 0:2] = np.dot(base[i, 0:2], rotation_matrix.T)

    for i in range(len(roof)):
        roof[i, 0:2] = np.dot(roof[i, 0:2], rotation_matrix.T)

    # Define the faces of the greenhouse
    vertices = [
        [base[0], base[1], roof[1], roof[0]],  # Side wall 1
        [base[1], base[2], roof[2], roof[1]],  # Side wall 2
        [base[2], base[3], roof[3], roof[2]],  # Side wall 3
        [base[3], base[0], roof[0], roof[3]],  # Side wall 4
    ]

    # Plot the 4 side walls
    for face in vertices:
        x = [v[0] for v in face] + [face[0][0]]  # Close the face
        y = [v[1] for v in face] + [face[0][1]]
        z = [v[2] for v in face] + [face[0][2]]
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=plotly_colors[0], width=5),
                showlegend=False,
            )
        )

    vertices = [
        # Roof
        [roof[0], roof[1], roof[5], roof[4]],  # Roof face 1
        [roof[2], roof[3], roof[4], roof[5]],  # Roof face 2
        [roof[0], roof[3], roof[4]],  # Roof face 3
        [roof[1], roof[2], roof[5]],  # Roof face 4
    ]

    # Plot the roof faces
    for face in vertices:
        x = [v[0] for v in face] + [face[0][0]]  # Close the face
        y = [v[1] for v in face] + [face[0][1]]
        z = [v[2] for v in face] + [face[0][2]]
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=plotly_colors[0], width=5),
                showlegend=False,
            )
        )

    return peak_height


def plotly_3d_compass(
    length: float,
    width: float,
    azimuth: float,
    fig: go.Figure,
):
    """
    Create a 3D compass visualization on the ground plane.

    Args:
        length: Length reference for sizing the compass
        width: Width reference for sizing the compass
        peak_height: Height reference for scene configuration
        azimuth: Orientation angle in degrees to highlight on the compass
        fig: Plotly figure object to add the compass to
    """
    # Create a circle for the compass base
    radius = max(length, width) // 1.5
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.zeros_like(theta) - 0.1  # Slightly below ground level

    fig.add_trace(
        go.Scatter3d(
            x=x_circle,
            y=y_circle,
            z=z_circle,
            mode="lines",
            line=dict(color="gray", width=2),
            showlegend=False,
        )
    )

    # Add cardinal directions
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    for direction, angle in zip(directions, angles):
        angle_rad = np.radians(angle)
        x_pos = (radius + 0.2 * radius) * np.cos(angle_rad)
        y_pos = (radius + 0.2 * radius) * np.sin(angle_rad)

        fig.add_trace(
            go.Scatter3d(
                x=[x_pos],
                y=[y_pos],
                z=[0.05],  # Slightly above the compass circle
                mode="text",
                text=[direction],
                textposition="middle center",
                textfont=dict(size=14, color="black"),
                showlegend=False,
            )
        )

    # Add directional lines
    for angle in angles:
        angle_rad = np.radians(angle)
        x_line = [0, radius * np.cos(angle_rad)]
        y_line = [0, radius * np.sin(angle_rad)]
        z_line = [-0.1, -0.1]  # Same level as compass circle

        fig.add_trace(
            go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode="lines",
                line=dict(color="gray", width=2),
                showlegend=False,
            )
        )

    # Add an arrow to indicate the azimuth
    azimuth_rad = np.radians(azimuth)
    x_arrow = [0, radius * 0.8 * np.cos(azimuth_rad)]
    y_arrow = [0, radius * 0.8 * np.sin(azimuth_rad)]
    z_arrow = [-0.1, -0.1]  # Same level as compass circle

    fig.add_trace(
        go.Scatter3d(
            x=x_arrow,
            y=y_arrow,
            z=z_arrow,
            mode="lines",
            line=dict(color=plotly_colors[1], width=4),
            showlegend=False,
        )
    )

    # Set up the 3D scene
    fig.update_layout(
        scene=dict(
            # xaxis=dict(range=[-max_dim, max_dim], title=""),
            # yaxis=dict(range=[-max_dim, max_dim], title=""),
            # zaxis=dict(range=[-0.2, max_dim * 1.1], title="Height"),
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.5,
                    y=1.5,
                    z=1.0,
                )
            ),
        )
    )
    return fig


def plotly_weather(climate: pd.DataFrame):
    # Resample data to hourly intervals
    climate_hourly = climate.resample("1h").median()

    # Create a new figure
    fig = go.Figure()

    # Group columns by their base name (before "[")
    column_groups: dict[str, list[str]] = {}
    for col in climate_hourly.columns:
        base_name = col.split("[")[0].strip()
        if base_name not in column_groups:
            column_groups[base_name] = []
        column_groups[base_name].append(col)

    # Add traces for each group
    for i, (base_name, columns) in enumerate(column_groups.items()):
        # Use different colors for each column in the same group
        for j, col in enumerate(columns):
            color = plotly_colors[(i + j) % len(plotly_colors)]
            # Only show in legend for the first item in each group
            show_legend = j == 0
            # Get the unit from the column name if available
            unit = col.split("[")[-1].split("]")[0] if "[" in col else ""
            # Create display name with unit if needed
            display_name = f"{base_name} [{unit}]" if unit else base_name

            fig.add_trace(
                go.Scatter(
                    x=climate_hourly.index,
                    y=climate_hourly[col],
                    mode="lines",
                    name=display_name,
                    line=dict(color=color),
                    legendgroup=base_name,
                    showlegend=show_legend,
                    visible=True if i < 5 else "legendonly",
                )
            )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
    )
    return fig


def export_fig(fig: go.Figure) -> bytes:
    """Export figure to bytes.

    Args:
        fig: A plotly figure

    Returns:
        Bytes representation of the figure in PDF format
    """
    from io import BytesIO

    buf = BytesIO()

    fig.write_image(buf, format="pdf")

    buf.seek(0)
    return buf.getvalue()
