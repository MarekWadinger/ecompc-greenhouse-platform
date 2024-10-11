from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams.update(
    {
        "figure.subplot.left": 0.1,
        "figure.subplot.bottom": 0.2,
        "figure.subplot.right": 0.95,
        "figure.subplot.top": 0.85,
    }
)

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
    | Literal["article", "ieee", "ieee_full", "thesis", "beamer"] = 307.28987,
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
    elif width == "ieee_full":
        width_pt = 252.0 * 2
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
        axs[1].tick_params(axis="x", rotation=15)

    fig.align_ylabels()
    fig.tight_layout()

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
        # subplot_titles=("Lettuce Dry Weight (g)", "Actuation [%]"),
    )

    # Add each column from the DataFrame as a separate trace
    for i, column in enumerate(y_nexts.columns):
        fig.add_trace(
            go.Scatter(
                x=y_nexts.index,
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
                legendgroup=label,
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        print()
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

    fig.update_yaxes(title_text="Lettuce Dry Weight (g)", row=1, col=1)
    fig.update_yaxes(title_text="Actuation [%]", row=2, col=1)
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        )
    )

    return fig


def plotly_greenhouse(length, width, height, roof_tilt, azimuth):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "polar"}]],
        # subplot_titles=("3D Greenhouse Model", "Greenhouse Orientation"),
    )

    # Plot the 3D Greenhouse
    plotly_3d_greenhouse(length, width, height, roof_tilt, azimuth, fig, 1, 1)

    # Plot the 2D Compass with Greenhouse Orientation
    plotly_compass_with_greenhouse(azimuth, length, width, fig, 1, 2)

    fig.update_layout(height=600, width=1200)
    return fig


def plotly_3d_greenhouse(
    length: float,
    width: float,
    height: float,
    roof_tilt: float,
    azimuth,
    fig: go.Figure,
    row: int,
    col: int,
):
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
    roof = np.array(
        [
            [0, 0, height],
            [length, 0, height],
            [length, width, height],
            [0, width, height],
            [0, width / 2, peak_height],
            [length, width / 2, peak_height],
        ]
    )

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
            ),
            row=row,
            col=col,
        )
    vertices = [
        # Roof
        [roof[0], roof[1], roof[5], roof[4]],  # Roof face 1
        [roof[2], roof[3], roof[4], roof[5]],  # Roof face 2
        [roof[0], roof[3], roof[4]],  # Roof face 3
        [roof[1], roof[2], roof[5]],  # Roof face 4
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
            ),
            row=row,
            col=col,
        )

    fig.update_scenes(
        camera=dict(
            eye=dict(
                x=np.cos(np.radians(azimuth)),  # Controls azimuth angle
                y=np.sin(np.radians(azimuth)),
                z=0.5,  # You can adjust the z to control elevation
            )
        ),
        xaxis=dict(title="Length", range=[0, length]),
        yaxis=dict(title="Width", range=[0, width]),
        zaxis=dict(title="Wall Height", range=[0, peak_height]),
        aspectratio=dict(
            x=0.75, y=width / length * 0.75, z=peak_height / length * 0.75
        ),
        row=row,
        col=col,
    )


def plotly_compass_with_greenhouse(
    azimuth: float,
    length: float,
    width: float,
    fig: go.Figure,
    row: int,
    col: int,
):
    # Directions and their angles (0° for N, 90° for E, etc.)
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    # Azimuth in degrees
    azimuth_angle = azimuth % 360  # Ensure azimuth is within [0, 360]

    # Base plot (empty compass)
    fig.add_trace(
        go.Scatterpolar(
            r=[1, 1, 1, 1],  # Radius for directions
            theta=angles,  # Angles for N, E, S, W
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # Compute the four corners of the greenhouse rectangle in Cartesian coordinates
    corners = np.array(
        [
            [-length, -width],  # Bottom-left
            [length, -width],  # Bottom-right
            [length, width],  # Top-right
            [-length, width],  # Top-left
        ]
    )

    # Convert azimuth to radians and create a rotation matrix
    azimuth_radians = np.radians(azimuth)
    rotation_matrix = np.array(
        [
            [np.cos(azimuth_radians), -np.sin(azimuth_radians)],
            [np.sin(azimuth_radians), np.cos(azimuth_radians)],
        ]
    )

    # Rotate the corners by the azimuth angle
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Convert rotated Cartesian coordinates to polar coordinates (angle in degrees, radius)
    theta = np.degrees(
        np.arctan2(rotated_corners[:, 1], rotated_corners[:, 0])
    )
    r = np.sqrt(rotated_corners[:, 0] ** 2 + rotated_corners[:, 1] ** 2)

    # Close the rectangle by appending the first point again
    theta = np.append(theta, theta[0])
    r = np.append(r, r[0])

    # Add the rotated greenhouse rectangle to the polar plot
    fig.add_trace(
        go.Scatterpolar(
            r=r,  # Radii
            theta=theta,  # Angles in degrees
            mode="lines",
            line=dict(color=plotly_colors[0], width=2),
            fill="toself",
            name="Greenhouse",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # Add an arrow to indicate the azimuth
    fig.add_trace(
        go.Scatterpolar(
            r=[0, max(length, width)],  # From center to outer edge
            theta=[0, azimuth_angle],  # Pointing in the direction of azimuth
            mode="lines",
            line=dict(color=plotly_colors[1], width=3),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            angularaxis=dict(
                tickvals=angles,  # Set angular tick values
                ticktext=directions,  # N, E, S, W labels
                direction="clockwise",  # Set azimuth to increment clockwise
            ),
        ),
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent entire background
    )

    return fig
