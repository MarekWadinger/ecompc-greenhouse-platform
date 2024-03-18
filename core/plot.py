from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "axes.labelsize": 8,
        "axes.grid": True,
        "font.size": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.subplot.left": 0.1,
        "figure.subplot.bottom": 0.2,
        "figure.subplot.right": 0.95,
        "figure.subplot.top": 0.85,
        # "backend": "macOsX"
    }
)


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
    if y_ref:
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
