"""Free-function plot helpers used by `CycleBehavMethods` and
`CycleLengthAnalyses`. Ported verbatim from `whoop_analyses/utils.py`:

  - `get_plotting_params`         (utils.py:907-925)
  - `setup_axes`                  (utils.py:751-788)
  - `fixed_yticks`                (utils.py:846-905)
  - `single_var_point_plot`       (utils.py:427-534)
  - `_get_counts_and_mean_BCI`    (utils.py:386-406)
  - `_add_counts_to_point_plot`   (utils.py:409-424)
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def get_plotting_params(dpi=500, fontsize=8,
                        label_fontsize_factor=1.1,
                        legend_fontsize_factor=0.9,
                        figure_fontsize_factor=1.5):
    label_fontsize = fontsize * label_fontsize_factor
    legend_fontsize = fontsize * legend_fontsize_factor
    title_fontsize = label_fontsize
    figure_fontsize = fontsize * figure_fontsize_factor
    sns.set_style("whitegrid")
    rc_params = {
        'figure.dpi': dpi,
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'font.size': figure_fontsize,
        'legend.fontsize': legend_fontsize,
        'legend.title_fontsize': legend_fontsize,}
    return rc_params


def setup_axes(
    ax,
    spine_lw=1.0,
    spine_color="k",
    grid_lw=0.5,
    spine_list=None,
    tick_params=None,
):

    if tick_params is None:
        tick_params = dict(
            axis="both",
            direction="out",
            length=2,
            width=spine_lw,
            color=spine_color,
            which="major",
            pad=0.5,
        )

    ax.spines[:].set_visible(False)

    if spine_list is None:
        spine_list = ["bottom", "left"]

    for sp in spine_list:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(spine_lw)
        ax.spines[sp].set_color(spine_color)
        tick_params[sp] = True

    if "polar" in tick_params:
        tick_params.pop("polar")

    ax.tick_params(**tick_params)

    ax.set_axisbelow(True)
    ax.grid(linewidth=grid_lw, zorder=0)


def fixed_yticks(ax, n_ticks=5, data_range=None, buffer=0.05, n_digits_input=None, symmetrical_around_zero=False):
    if data_range is None:
        data_range = ax.get_ylim()
    else:
        assert len(data_range) == 2

    def round_to_nice(value):
        nice_values = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
        return nice_values[np.argmin([abs(value - nice) for nice in nice_values])]

    def format_tick_consistent(tick, n_digits):
        return f'{tick:.{n_digits}f}'

    y_range = data_range[1] - data_range[0]
    y0 = data_range[0] + buffer * y_range
    y1 = data_range[1] - buffer * y_range

    if symmetrical_around_zero:
        max_abs_val = max(abs(y0), abs(y1))
        y0, y1 = -max_abs_val, max_abs_val
        y_range = y1 - y0

    ideal_tick_spacing = y_range / (n_ticks - 1)
    tick_spacing = round_to_nice(ideal_tick_spacing)

    tick_start = np.floor(y0 / tick_spacing) * tick_spacing
    tick_end = np.ceil(y1 / tick_spacing) * tick_spacing

    yticks = np.linspace(tick_start, tick_end, n_ticks)

    if n_digits_input is None:
        min_diff = np.min(np.diff(yticks))
        if min_diff < 1:
            n_digits = int(np.ceil(-np.log10(min_diff)))
        else:
            n_digits = 0
    else:
        n_digits = n_digits_input

    ax.set_yticks(yticks)

    formatted_ticks = [format_tick_consistent(tick, n_digits) for tick in yticks]

    ax.set_yticklabels(formatted_ticks)

    tick_range = yticks[-1] - yticks[0]
    tick_buffer = buffer * tick_range
    ax.set_ylim(yticks[0] - tick_buffer, yticks[-1] + tick_buffer)


def _get_counts_and_mean_BCI(
    data, grouper, y_var, B=1000, estimator=np.nanmean
):
    grouped_data = pd.DataFrame(data[grouper].value_counts().sort_index())
    grouped_data["low"] = np.nan
    grouped_data["high"] = np.nan
    grouped_data["mean"] = np.nan
    for ii, gg in enumerate(grouped_data.index):
        try:
            a, b = stats.bootstrap(
                (data.loc[data[grouper] == gg, y_var].values,),
                estimator,
                n_resamples=B,
            ).confidence_interval
            grouped_data.loc[gg, ["low", "high"]] = a, b
            grouped_data.loc[gg, "mean"] = estimator(
                data.loc[data[grouper] == gg, y_var].values
            )
        except:
            pass
    return grouped_data


def _add_counts_to_point_plot(grouped_data, ax, dx, dy_factor=None):
    ylims = ax.get_ylim()
    ax_range = ylims[1] - ylims[0]
    dy = ax_range * dy_factor if dy_factor is not None else 0.01

    for ii, ag in enumerate(grouped_data.index):
        if np.isnan(grouped_data.loc[ag, "high"]):
            continue
        ax.text(
            ii + dx,
            grouped_data.loc[ag, "high"] + dy,
            f"{grouped_data.loc[ag, 'count']}",
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="bottom",
        )


def single_var_point_plot(
    data,
    x_var,
    y_var,
    ax,
    estimator=np.nanmean,
    dx=0,
    color="0.3",
    add_counts=True,
    ms=5,
    lw=2.5,
    join_points=True,
    marker_edge_color=None,
    marker_edge_width=0.4,
    dy_factor=None
):
    gdata = _get_counts_and_mean_BCI(
        data, grouper=x_var, y_var=y_var, estimator=estimator
    )
    n_x = len(gdata)

    if not join_points:
        for ii, ag in enumerate(gdata.index):
            if not np.isnan(gdata.loc[ag, "mean"]):
                ax.plot(
                    ii + dx,
                    gdata.loc[ag, "mean"],
                    marker="o",
                    color=color,
                    lw=lw * 0.9,
                    markersize=ms,
                    markeredgecolor=marker_edge_color,
                    markeredgewidth=marker_edge_width if marker_edge_color is not None else 0,
                    zorder=10,
                )

                ax.plot(
                    np.array((ii, ii)) + dx,
                    gdata.iloc[ii][["low", "high"]],
                    color=color,
                    lw=lw,
                    zorder=9
                )
            else:
                continue
    else:
        ax.plot(np.arange(n_x) + dx, gdata["mean"], marker="o",
                color=color, lw=lw * join_points,
                markersize=ms, markeredgecolor=marker_edge_color,
                markeredgewidth=marker_edge_width if marker_edge_color is not None else 0,
                zorder=10)
        for ii in range(n_x):
            ax.plot(
                np.array((ii, ii)) + dx,
                gdata.iloc[ii][["low", "high"]],
                color=color,
                lw=lw * 0.9,
                zorder=9
            )

    if add_counts:
        _add_counts_to_point_plot(gdata, ax, dx, dy_factor)
    ax.set_xticks(range(n_x))
    ax.set_xticklabels(gdata.index.values)
    return ax


def get_variable_weights(data, var, bins=None, labels=None):
    """
    Compute weights for a variable by binning and inverse log-count weighting.
    Returns a DataFrame with the variable, bin, and weight.
    """
    # Bin the variable if bins are provided
    if bins is not None:
        binned = pd.cut(data[var], bins=bins, labels=labels, include_lowest=True)
    else:
        binned = data[var]

    # Compute weights: inverse log of bin counts
    bin_counts = binned.value_counts().sort_index()
    weights = 1 / np.log(bin_counts.replace(0, np.nan))
    weights = weights.replace([np.inf, -np.inf], 0).fillna(0)

    # Assign weights to each row
    weight_col = binned.map(weights)

    min_val = bins[0] if bins is not None else None
    max_val = bins[-1] if bins is not None else None
    # Handle out-of-bounds if min_val/max_val provided
    if min_val is not None and max_val is not None and bins is not None:
        lowest_bin = binned.cat.categories[0]
        highest_bin = binned.cat.categories[-1]
        weight_low = weights.loc[lowest_bin] if lowest_bin in weights else 0
        weight_high = weights.loc[highest_bin] if highest_bin in weights else 0
        weight_col[data[var] < min_val] = weight_low
        weight_col[data[var] > max_val] = weight_high

    data[f"{var}_weight"] = weight_col
    return data


def cname2hex(cname):
    colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS) # dictionary. key: names, values: hex codes
    try:
        hex = colors[cname]
        return hex
    except KeyError:
        print(cname, ' is not registered as default colors by matplotlib!')
        return None


def hex2rgb(hex, normalize=False):
    h = hex.strip('#')
    rgb = np.asarray(list(int(h[i:i + 2], 16) for i in (0, 2, 4)))
    return rgb


def draw_rectangle_gradient(ax, x1, y1, width, height, color1='white', color2='blue', alpha1=0.0, alpha2=0.5, n=100):
    # convert color names to rgb if rgb is not given as arguments
    if not color1.startswith('#'):
        color1 = cname2hex(color1)
    if not color2.startswith('#'):
        color2 = cname2hex(color2)
    color1 = hex2rgb(color1) / 255.  # np array
    color2 = hex2rgb(color2) / 255.  # np array


    # Create an array of the linear gradient between the two colors
    gradient_colors = []
    for segment in np.linspace(0, width, n):
        interp_color = [(1 - segment / width) * color1[j] + (segment / width) * color2[j] for j in range(3)]
        interp_alpha = (1 - segment / width) * alpha1 + (segment / width) * alpha2
        gradient_colors.append((*interp_color, interp_alpha))
    for i, color in enumerate(gradient_colors):
        ax.add_patch(plt.Rectangle((x1 + width/n * i, y1), width/n, height, color=color, linewidth=0, zorder=0))
    return ax


def add_legend(ax, title=None, labels=None, handles=None,
                bbox_to_anchor=[1, 0.05, 0.2, 0.8], loc=3, reverse_labels=False, **kwargs):
    if handles is None:
        handles, labels2 = ax.get_legend_handles_labels()
        if labels is None:
            labels = labels2

    if reverse_labels:
        handles = handles[::-1]
        labels = labels[::-1]

    if len(handles) == 0:
        return

    if 'handlelength' not in kwargs:
        kwargs['handlelength'] = 1
    if 'labelspacing' not in kwargs:
        kwargs['labelspacing'] = 0.5

    ax.legend().remove()
    f = ax.figure
    l = f.legend(
        handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, title=title,
        frameon=True, fancybox=True,
        **kwargs
    )
    l.get_frame().set_linewidth(0)
    l.get_frame().set_facecolor('0.97')
    l.get_frame().set_alpha(0.9)

    return l
