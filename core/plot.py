from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
