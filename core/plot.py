from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def plot_response(
    t_out: list,
    y_out: list,
    u_out: list,
    y_ref: Union[list, None] = None,
    u_min: Union[list[float], None] = None,
    u_max: Union[list[float], None] = None,
    axs_: Union[np.ndarray, None] = None,
) -> np.ndarray:
    if axs_ is None:
        _, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    else:
        axs = axs_
    axs[0].plot(
        t_out, y_out, label=[r"$x_{\mathrm{sdw}}$", r"$x_{\mathrm{nsdw}}$"]
    )
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
        label=[
            r"$u_{\mathrm{T}}$",
            r"$u_{\mathrm{par}}$",
            r"$u_{\mathrm{CO_2}}$",
        ],
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
