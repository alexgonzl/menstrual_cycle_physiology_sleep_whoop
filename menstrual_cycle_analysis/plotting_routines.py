import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


class PLOTTING_ROUTINES:

    DEFAULT_PLOTTING_PARAMS = {
        'dpi': 500,
        'fontsize': 8,
        'label_fontsize_factor': 1.1,
        'legend_fontsize_factor': 0.9,
        'figure_fontsize_factor': 1.5
    }

    def __init__(self, **plotting_params):
        self.plotting_params = self.DEFAULT_PLOTTING_PARAMS
        self.plotting_params.update(plotting_params)
        self.rc_params = self.set_plotting_params(**self.plotting_params)

    def set_plotting_params(self, **kwargs):
        
        sns.set_style("whitegrid")
        
        if not kwargs:
            kwargs = self.DEFAULT_PLOTTING_PARAMS
        label_fontsize = kwargs['fontsize'] * kwargs['label_fontsize_factor']
        legend_fontsize = kwargs['fontsize'] * kwargs['legend_fontsize_factor']
        title_fontsize = label_fontsize
        figure_fontsize = kwargs['fontsize'] * kwargs['figure_fontsize_factor']
                
        rc_params = {
            'figure.dpi': kwargs['dpi'],
            'axes.titlesize': title_fontsize,
            'axes.labelsize': label_fontsize,
            'xtick.labelsize': kwargs['fontsize'],
            'ytick.labelsize': kwargs['fontsize'],
            'font.size': figure_fontsize,
            'legend.fontsize': legend_fontsize,
            'legend.title_fontsize': legend_fontsize}
        return rc_params
    
    def setup_axes(self, ax, spine_lw=1.0, spine_color="k",
    grid_lw=0.5, spine_list=None, tick_params=None, **kwargs):

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

    def single_var_point_plot(self,
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
        """
        Plots point estimates (with confidence intervals) of a single variable grouped by another variable.
        Parameters
        ----------
        data : pandas.DataFrame
            The input data containing the variables to plot.
        x_var : str
            The column name in `data` to group by (x-axis categories).
        y_var : str
            The column name in `data` to compute the summary statistic for (y-axis values).
        ax : matplotlib.axes.Axes
            The matplotlib axes object to plot on.
        estimator : callable, optional
            Function to compute the central tendency (e.g., np.nanmean, np.nanmedian). Default is np.nanmean.
        dx : float, optional
            Horizontal offset to apply to the x positions of the points. Default is 0.
        color : str or tuple, optional
            Color for the points and lines. Default is "0.3" (gray).
        add_counts : bool, optional
            Whether to annotate the plot with counts for each group. Default is True.
        ms : float, optional
            Marker size for the points. Default is 5.
        lw : float, optional
            Line width for the error bars and connecting lines. Default is 2.5.
        join_points : bool, optional
            If True, connect the points with lines. If False, plot points individually. Default is True.
        marker_edge_color : str or None, optional
            Color for the marker edge. If None, no edge is drawn. Default is None.
        marker_edge_width : float, optional
            Width of the marker edge. Default is 0.4.
        dy_factor : float or None, optional
            Factor to adjust the vertical position of the count annotations. Default is None.
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot added.
        Notes
        -----
        This function uses a helper function `_get_counts_and_mean_BCI` to compute group means and confidence intervals,
        and `_add_counts_to_point_plot` to annotate counts. Error bars represent confidence intervals for each group.
        """

        gdata = self._get_counts_and_mean_BCI(
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
            self._add_counts_to_point_plot(gdata, ax, dx, dy_factor)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(gdata.index.values)
        return ax

    def fix_yticks(self, ax, n_ticks=5, data_range=None, 
                     buffer=0.05, n_digits_input=None, symmetrical_around_zero=False):
        if data_range is None:
            data_range = ax.get_ylim()
        else:
            assert len(data_range) == 2

        # Function to round tick spacing to the nearest "nice" value
        def round_to_nice(value):
            nice_values = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
            return nice_values[np.argmin([abs(value - nice) for nice in nice_values])]

        # Function to format tick labels with consistent significant figures
        def format_tick_consistent(tick, n_digits):
            return f'{tick:.{n_digits}f}'

        # Calculate the y-range and apply buffer
        y_range = data_range[1] - data_range[0]
        y0 = data_range[0] + buffer * y_range
        y1 = data_range[1] - buffer * y_range

        # Adjust for symmetry around zero if the option is enabled
        if symmetrical_around_zero:
            max_abs_val = max(abs(y0), abs(y1))
            y0, y1 = -max_abs_val, max_abs_val
            y_range = y1 - y0  # Recalculate range for symmetry

        # Calculate ideal tick spacing and round to the nearest "nice" value
        ideal_tick_spacing = y_range / (n_ticks - 1)
        tick_spacing = round_to_nice(ideal_tick_spacing)

        # Adjust the starting and ending tick values to "nice" values
        tick_start = np.floor(y0 / tick_spacing) * tick_spacing
        tick_end = np.ceil(y1 / tick_spacing) * tick_spacing

        # Generate evenly spaced ticks between adjusted start and end
        yticks = np.linspace(tick_start, tick_end, n_ticks)

        # If no specific precision is provided, calculate dynamically based on smallest difference
        if n_digits_input is None:
            min_diff = np.min(np.diff(yticks))
            if min_diff < 1:
                n_digits = int(np.ceil(-np.log10(min_diff)))  # Dynamic precision based on smallest difference
            else:
                n_digits = 0  # No decimal places needed for large tick differences
        else:
            n_digits = n_digits_input  # Use the input precision

        # Set the yticks
        ax.set_yticks(yticks)

        # Format tick labels with consistent significant digits
        formatted_ticks = [format_tick_consistent(tick, n_digits) for tick in yticks]

        # Apply dynamic formatting to tick labels
        ax.set_yticklabels(formatted_ticks)

        # Apply the buffer to the y-limits AFTER setting ticks
        tick_range = yticks[-1] - yticks[0]
        tick_buffer = buffer * tick_range
        ax.set_ylim(yticks[0] - tick_buffer, yticks[-1] + tick_buffer)
        
    def draw_rectangle_gradient(self, ax, x1, y1, width, height, color1='white', color2='blue', alpha1=0.0, alpha2=0.5, n=100):
        # convert color names to rgb if rgb is not given as arguments
        if not color1.startswith('#'):
            color1 = self._cname2hex(color1)
        if not color2.startswith('#'):
            color2 = self._cname2hex(color2)
        color1 = self._hex2rgb(color1) / 255.  # np array
        color2 = self._hex2rgb(color2) / 255.  # np array


        # Create an array of the linear gradient between the two colors
        gradient_colors = []
        for segment in np.linspace(0, width, n):
            interp_color = [(1 - segment / width) * color1[j] + (segment / width) * color2[j] for j in range(3)]
            interp_alpha = (1 - segment / width) * alpha1 + (segment / width) * alpha2
            gradient_colors.append((*interp_color, interp_alpha))
        for i, color in enumerate(gradient_colors):
            ax.add_patch(plt.Rectangle((x1 + width/n * i, y1), width/n, height, color=color, linewidth=0, zorder=0))
        return ax

    ## PRIVATE METHODS
    def _get_counts_and_mean_BCI(self, 
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

    def _add_counts_to_point_plot(self, grouped_data, ax, dx, dy_factor=None):
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

    def _cname2hex(self, cname):
        colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS) # dictionary. key: names, values: hex codes
        try:
            hex = colors[cname]
            return hex
        except KeyError:
            print(cname, ' is not registered as default colors by matplotlib!')
            return None

    def _hex2rgb(self, hex):
        h = hex.strip('#')
        rgb = np.asarray(list(int(h[i:i + 2], 16) for i in (0, 2, 4)))
        return rgb


   