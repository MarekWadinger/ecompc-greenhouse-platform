from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "CMU Serif",
        "font.serif": "CMU Serif",
        "axes.grid": True,
        # "axes.labelsize": 8,
        # "font.size": 8,
        # "legend.fontsize": 8,
        # "xtick.labelsize": 8,
        # "ytick.labelsize": 8,
        "figure.subplot.left": 0.1,
        "figure.subplot.bottom": 0.2,
        "figure.subplot.right": 0.95,
        "figure.subplot.top": 0.85,
        # "backend": "macOsX"
    }
)


def set_size(
    width: float
    | int
    | Literal["article", "ieee", "thesis", "beamer"] = 307.28987,
    fraction=1.0,
    subplots=(1, 1),
):
    """Set figure dimensions to avoid scaling in LaTeX.

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
    elif width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = (
        fig_width_in * golden_ratio * ((subplots[0] * fraction) / subplots[1])
    )

    return (fig_width_in * 1.2, fig_height_in * 1.2)


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
    t_out: list,
    y_out: list,
    u_out: list,
    y_ref: Union[list, None] = None,
    u_min: Union[list[float], None] = None,
    u_max: Union[list[float], None] = None,
    axs_: Union[np.ndarray, None] = None,
    y_label: Union[list[str], None] = None,
    u_label: Union[list[str], None] = None,
) -> np.ndarray:
    if axs_ is None:
        _, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    else:
        axs = axs_
    axs[0].plot(t_out, y_out, label=y_label)
    if y_ref is not None:
        axs[0]._get_lines.set_prop_cycle(None)
        axs[0].plot(t_out, y_ref, label=r"$y_{\mathrm{ref}}$", linestyle=":")
    if axs_ is None:
        axs[0].set_ylabel("$y$")
        axs[0].set_title("a) Response of a System")
        axs[0].legend()

    axs[1].plot(
        t_out,
        u_out,
        label=u_label,
    )
    if u_min is not None and u_max is not None:
        axs[1]._get_lines.set_prop_cycle(None)
        for u_min_, u_max_ in zip(u_min, u_max):
            color = axs[1]._get_lines.get_next_color()
            axs[1].axhline(u_min_, color=color, linestyle=":")
            axs[1].axhline(u_max_, color=color, linestyle=":")
        axs[1]._get_lines.set_prop_cycle(None)

    if axs_ is None:
        axs[1].set_xlabel("$t$")
        axs[1].set_ylabel("$u$")
        axs[1].set_title("b) Control Action")
        axs[1].legend()

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
    ums,
):
    y_nexts_ = np.array(y_nexts).reshape(-1, 2)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Lettuce Dry Weight (g)", "Actuation [%]"),
    )

    # Plot Lettuce Dry Weight
    fig.add_trace(
        go.Scatter(
            x=_timestamps,
            y=y_nexts_[:, -2],
            mode="lines",
            name="structural",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=_timestamps,
            y=y_nexts_[:, -1],
            mode="lines",
            name="nonstructural",
        ),
        row=1,
        col=1,
    )

    # Plot Actuation
    labels = ["fan", "heater"]
    colors = ["#1f77b4", "#ff7f0e"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=np.array(u0s).reshape(-1, 2)[:, i],
                mode="lines",
                name=label,
                line=dict(color=color),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=np.array(ums).reshape(-1, 4)[:, i],
                mode="lines",
                name=f"{label} min",
                line=dict(color=color, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_timestamps,
                y=np.array(ums).reshape(-1, 4)[:, i + 2],
                mode="lines",
                name=f"{label} max",
                line=dict(color=color, dash="dash"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=600, width=800, title_text="Greenhouse Control")
    fig.update_yaxes(title_text="Lettuce Dry Weight (g)", row=1, col=1)
    fig.update_yaxes(title_text="Actuation [%]", row=2, col=1)

    return fig


def plot_greenhouse(length, width, height, roof_tilt, azimuth):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="polar")
    plot_3d_greenhouse(length, width, height, roof_tilt, ax=ax1)
    plot_compass_with_greenhouse(azimuth, length, width, ax=ax2)
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5)
    return fig


# Function to plot a 3D greenhouse
def plot_3d_greenhouse(length, width, height, roof_tilt, ax=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    else:
        fig = ax.get_figure()

    # Vertices of the greenhouse base (rectangle)
    base = np.array(
        [
            [0, 0, 0],
            [length, 0, 0],
            [length, width, 0],
            [0, width, 0],
            [0, 0, 0],  # Close the base
        ]
    )

    # Height of the roof peak (using roof tilt)
    peak_height = height + np.tan(np.radians(roof_tilt)) * (width / 2)

    # Roof vertices
    roof = np.array(
        [
            [0, 0, height],
            [length, 0, height],
            [length, width, height],
            [0, width, height],
            [0, width / 2, peak_height],  # Roof peak
            [length, width / 2, peak_height],  # Roof peak
        ]
    )

    # Define the faces of the greenhouse
    vertices = [
        # Base
        [base[0], base[1], base[2], base[3]],
        # Walls
        [base[0], base[1], roof[1], roof[0]],  # Side wall 1
        [base[1], base[2], roof[2], roof[1]],  # Side wall 2
        [base[2], base[3], roof[3], roof[2]],  # Side wall 3
        [base[3], base[0], roof[0], roof[3]],  # Side wall 4
        # Roof
        [roof[0], roof[1], roof[5], roof[4]],  # Roof face 1
        [roof[2], roof[3], roof[4], roof[5]],  # Roof face 2
        [roof[0], roof[3], roof[4]],  # Roof face 3
        [roof[1], roof[2], roof[5]],  # Roof face 4
    ]

    # Create the 3D polyhedron using the vertices
    ax.add_collection(
        Poly3DCollection(
            vertices,
            facecolors="tab:blue",
            linewidths=1,
            edgecolors="tab:blue",
            alpha=0.05,
        )
    )

    # Set labels and limits for clarity
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    ax.set_zlabel("Height (m)")
    max_shape = max([length, width])
    ax.set_xlim([-1 * max_shape * 0.1, max_shape * 1.1])
    ax.set_ylim([-1 * max_shape * 0.1, max_shape * 1.1])
    ax.set_zlim([-1 * length * 0.1, peak_height * 1.1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Set the rotation of the 3D plot
    ax.view_init(elev=20, azim=30)

    # TODO: make titles at the same height
    # ax.set_title("3D Greenhouse Model")

    return fig


# Function to plot 2D compass
def plot_compass_with_greenhouse(azimuth, length, width, ax=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        fig = ax.get_figure()

    # Create a circular compass with directions
    directions = ["N", "E", "S", "W"]
    angles = [90, 180, 270, 0]

    # Set up the ticks and labels for the compass
    ax.set_xticks(np.radians(angles))
    ax.set_xticklabels(directions)
    ax.set_yticks([])

    # Calculate the azimuth in radians
    azimuth_radians = np.radians(azimuth + 90)

    # Calculate the four corners of the rectangle in Cartesian coordinates
    corners = np.array(
        [
            [-length / 2, -width / 2],  # Bottom-left
            [length / 2, -width / 2],  # Bottom-right
            [length / 2, width / 2],  # Top-right
            [-length / 2, width / 2],  # Top-left
        ]
    )

    # Rotate the rectangle to align with the azimuth angle
    rotation_matrix = np.array(
        [
            [np.cos(azimuth_radians), -np.sin(azimuth_radians)],
            [np.sin(azimuth_radians), np.cos(azimuth_radians)],
        ]
    )
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Convert the rotated corners to polar coordinates for plotting
    angles = np.arctan2(rotated_corners[:, 1], rotated_corners[:, 0])
    radii = np.sqrt(rotated_corners[:, 0] ** 2 + rotated_corners[:, 1] ** 2)

    # Close the rectangle by appending the first point again
    angles = np.concatenate([angles, [angles[0]]])
    radii = np.concatenate([radii, [radii[0]]])

    # Plot the rectangle by connecting the corners
    ax.plot(
        angles, radii, color="tab:blue", linestyle="-", linewidth=2, alpha=0.7
    )

    # TODO: Draw an arrow indicating the azimuth
    ax.annotate(
        "",
        xy=(azimuth_radians, 1),
        xytext=(0, 0),
        arrowprops=dict(facecolor="red", shrink=0.05),
    )

    # TODO: make titles at the same height
    # ax.set_title(f"Greenhouse Orientation: {azimuth}°")
    return fig
