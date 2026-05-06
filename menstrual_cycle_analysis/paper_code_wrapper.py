"""`CycleLengthAnalyses` — verbatim port of the source class with notebook 01
scope kept (Fig 1 + S1-S4). Methods unused by notebook 01 are intentionally
not included; they will be ported when the notebooks that need them are added.

Source: `whoop_analyses/whoop_analyses/paper_code_wrapper.py` (CycleLengthAnalyses).

Modifications from source:
  - Constructor takes `CBM=None` only (no `data` fallback that constructs
    `WHOOP_USER_TABLES`).
  - `paper_figures_path` is set to `config.FIGURES_DIR` instead of pulled from
    `WHOOP_USER_TABLES`.
  - Method bodies are otherwise byte-identical to the source.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import config
from ._plot_utils import get_plotting_params, setup_axes
from .stats.contrasts import StatisticalPredictionHandler


class CycleLengthAnalyses:
    """Analysis class for menstrual cycle length data and relationships."""

    DEFAULT_AGE_AXIS_WIDTH = 3
    DEFAULT_BMI_AXIS_WIDTH = 3
    DEFAULT_W_SPACING = 0.2
    DEFAULT_H_SPACING = 0.2
    DEFAULT_CL_AXIS_HEIGHT = 2
    DPI = 500

    def __init__(self, CBM, cycle_type='j'):
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.plotting_params = get_plotting_params(fontsize=8, figure_fontsize_factor=1.5)

        # Core setup
        self.CBM = CBM
        self.data = CBM.data

        self._setup_layout_params()
        self._setup_plotting_params()
        self._default_model_terms()
        self._setup_modeling_params()

        self.paper_figures_path = config.FIGURES_DIR

        # State tracking
        self.added_sleep = False
        self.added_workout = False
        self.added_phase = False
        self.models = {}

    # =================== INITIALIZATION METHODS ===================
    def _setup_layout_params(self, layout_params=None):
        """Initialize layout and spacing parameters, optionally from a dictionary."""
        params = layout_params or {}
        self._age_axis_width = params.get('age_axis_width', self.DEFAULT_AGE_AXIS_WIDTH)
        self._BMI_axis_width = params.get('BMI_axis_width', self.DEFAULT_BMI_AXIS_WIDTH)
        self._w_spacing = params.get('w_spacing', self.DEFAULT_W_SPACING)
        self._h_spacing = params.get('h_spacing', self.DEFAULT_H_SPACING)
        self._CL_axis_height = params.get('CL_axis_height', self.DEFAULT_CL_AXIS_HEIGHT)

    def _setup_plotting_params(self):
        """Initialize plotting style parameters."""
        self._model_plot_color = "#2964d9"
        self._model_plot_alpha = 0.5
        self._model_plot_lw = 2
        self._model_plot_ls = '-'

    def _default_model_terms(self):
        self.control_terms = ['age', 'age2', 'BMI', 'BMI2']
        self.seasonal_terms = ['cos_season', 'sin_season']

        self.all_sleep_terms = ['sl_onset_cos', 'sl_onset_sin', 'sl_onset_var',
                                'sl_dur_mean', 'sl_dur_mean2', 'sl_dur_lvar', 'sl_dur_lvar2']
        self.all_wo_terms = ['wo_time_cos', 'wo_time_sin',
                             'total_wo_duration', 'norm_intensity', 'eTRIMP', 'eTRIMP2']

    def _setup_modeling_params(self):
        """Initialize statistical modeling parameters and configurations."""
        self.gee_cov_structures = dict(ind=sm.genmod.cov_struct.Independence(),
                                       exch=sm.genmod.cov_struct.Exchangeable(),
                                       ar=sm.genmod.cov_struct.Autoregressive())

        self.gee_family = dict(gaussian=sm.families.Gaussian(),
                               binomial=sm.families.Binomial())

        cycle_table = self.CBM.tables['cycle'].reset_index()
        user_table = self.CBM.tables['user'].reset_index()
        mp = dict(
            sl_dur_age=dict(data=user_table,
                            y_var='sl_dur_mean',
                            x_var='age',
                            group_var='age_b',
                            yticks=[6.5, 7.0, 7.5]),
            sl_dur_bmi=dict(data=user_table,
                            y_var='sl_dur_mean',
                            x_var='BMI',
                            group_var='BMI_b2',
                            yticks=[6.5, 7.0, 7.5]),
            eTRIMP_age=dict(data=user_table,
                            y_var='eTRIMP',
                            x_var='age',
                            group_var='age_b',
                            yticks=[30, 90, 150]),
            eTRIMP_bmi=dict(data=user_table,
                            y_var='eTRIMP',
                            x_var='BMI',
                            group_var='BMI_b2',
                            yticks=[30, 90, 150]),
            cl_mean_age=dict(
                data=cycle_table,
                y_var='cycle_length',
                x_var='age',
                group_var='age_b',
                plot_var='mean_cl',
                weight_var='age_b_weights',
                model_func=self._fit_gee_model,
                model_params=dict(
                    cov_struct=self.gee_cov_structures['exch'],
                    family=self.gee_family['gaussian']
                ),
                control_terms=self.control_terms,
                yticks=[26, 28, 30],
                seasonal_terms=self.seasonal_terms,
                additional_terms=[],
                wo_terms=[],
                sl_terms=[],
                categorical_terms=[],
                bin_edges=self.CBM.age_bin_edges,
            ),
            cl_p3_age=dict(
                data=cycle_table,
                model_func=self._fit_gee_model,
                x_var='age',
                group_var='age_b',
                y_var='a_delta_cl_ge_3',
                plot_var='a_delta_cl_ge_3',
                weight_var='age_b_weights',
                control_terms=self.control_terms,
                yticks=[0.2, 0.3, 0.4],
                seasonal_terms=self.seasonal_terms,
                additional_terms=['median_cl'],
                interaction_terms=['median_cl*age', 'median_cl*age2'],
                wo_terms=[],
                sl_terms=[],
                categorical_terms=[],
                model_params=dict(
                    cov_struct=self.gee_cov_structures['exch'],
                    family=self.gee_family['binomial'],
                    time='data_day',
                ),
                bin_edges=self.CBM.age_bin_edges,
            ),
            cl_sd_age=dict(
                data=cycle_table,
                model_func=self._fit_gee_model,
                x_var='age',
                group_var='age_b',
                y_var='sd_cl',
                plot_var='sd_cl',
                weight_var='age_b_weights',
                control_terms=self.control_terms,
                yticks=[2.5, 3.5, 4.5],
                seasonal_terms=self.seasonal_terms,
                additional_terms=['mean_cl'],
                interaction_terms=['mean_cl*age'],
                wo_terms=[],
                sl_terms=[],
                categorical_terms=[],
                model_params=dict(
                    cov_struct=self.gee_cov_structures['exch'],
                    family=self.gee_family['gaussian']
                ),
                bin_edges=self.CBM.age_bin_edges,
            ),
            cl_mean_bmi=dict(
                data=cycle_table,
                x_var='BMI',
                group_var='BMI_b',
                y_var='cycle_length',
                plot_var='mean_cl',
                weight_var='BMI_b_weights',
                model_func=self._fit_gee_model,
                model_params=dict(
                    cov_struct=self.gee_cov_structures['exch'],
                    family=self.gee_family['gaussian']
                ),
                control_terms=self.control_terms,
                yticks=[26, 28, 30],
                seasonal_terms=self.seasonal_terms,
                additional_terms=[],
                wo_terms=[],
                sl_terms=[],
                categorical_terms=[],
                bin_edges=self.CBM.bmi_bin_edges,
            ),
            cl_p3_bmi=dict(
                data=cycle_table,
                x_var='BMI',
                group_var='BMI_b',
                y_var='a_delta_cl_ge_3',
                plot_var='a_delta_cl_ge_3',
                weight_var='BMI_b_weights',
                model_func=self._fit_gee_model,
                model_params=dict(
                    cov_struct=self.gee_cov_structures['exch'],
                    family=self.gee_family['binomial']
                ),
                control_terms=self.control_terms,
                yticks=[0.2, 0.3, 0.4],
                seasonal_terms=self.seasonal_terms,
                additional_terms=['median_cl'],
                interaction_terms=['median_cl*age'],
                wo_terms=[],
                sl_terms=[],
                categorical_terms=[],
                bin_edges=self.CBM.bmi_bin_edges,
            ),
            cl_mean_eTRIMP=dict(
                data=cycle_table,
                y_var='cycle_length',
                plot_var='mean_cl',
                x_var='eTRIMP',
                group_var='eTRIMP_bin',
                weight_var='eTRIMP_bin_weights' if 'eTRIMP_bin_weights' in cycle_table.columns else None,
                bin_edges=self.CBM.eTRIMP_bin_edges[:-1],
                model_func=self._fit_gee_model,
                model_params=dict(
                    cov_struct=self.gee_cov_structures['exch'],
                    family=self.gee_family['gaussian']
                ),
                control_terms=self.control_terms,
                yticks=[26, 28, 30],
                seasonal_terms=self.seasonal_terms,
                additional_terms=[],
                wo_terms=self.all_wo_terms,
                sl_terms=self.all_sleep_terms,
                categorical_terms=[],
            ),
            cl_p3_eTRIMP=dict(
                data=cycle_table,
                y_var='a_delta_cl_ge_3',
                plot_var='a_delta_cl_ge_3',
                x_var='eTRIMP',
                group_var='eTRIMP_bin',
                weight_var='eTRIMP_bin_weights' if 'eTRIMP_bin_weights' in cycle_table.columns else None,
                model_func=self._fit_gee_model,
                model_params=dict(
                    cov_struct=self.gee_cov_structures['exch'],
                    family=self.gee_family['binomial']
                ),
                control_terms=self.control_terms,
                yticks=[0.2, 0.3, 0.4],
                seasonal_terms=self.seasonal_terms,
                additional_terms=['median_cl'],
                interaction_terms=['median_cl*age'],
                wo_terms=self.all_wo_terms,
                sl_terms=self.all_sleep_terms,
                categorical_terms=[],
                bin_edges=self.CBM.eTRIMP_bin_edges[:-1],
            ),
        )
        self.model_params = mp

    def get_model(self, model_name, overwrite=False):
        """Get or create a statistical model."""
        if model_name not in self.models or overwrite:
            mp = self.model_params[model_name]
            func = mp['model_func']
            m = func(**mp)
            self.models[model_name] = m

        return self.models[model_name]

    def _fit_gee_model(self, y_var, data, terms=None, **mp):
        """Fit generalized estimating equation model with interaction term support."""
        if terms is None:
            terms = self._collect_model_terms(**mp)

        # Separate interaction terms from regular terms for data validation
        regular_terms, interaction_terms = self._separate_interaction_terms(terms)

        weight_var = mp.get('weight_var', None)

        # Only check regular terms for missing data
        data_clean = data.dropna(subset=regular_terms + [y_var])

        formula = f'{y_var} ~ {"+".join(terms)}'

        m = smf.gee(
            formula=formula,
            groups=data_clean['n_id'],
            data=data_clean,
            weights=data_clean[weight_var] if weight_var else None,
            **mp['model_params']
        ).fit()
        return m

    def _separate_interaction_terms(self, terms):
        """Separate regular terms from interaction terms."""
        regular_terms = []
        interaction_terms = []

        for term in terms:
            if '*' in term or ':' in term:
                interaction_terms.append(term)
                if '*' in term:
                    base_terms = term.split('*')
                else:
                    base_terms = term.split(':')

                for base_term in base_terms:
                    base_term = base_term.strip()
                    if base_term not in regular_terms:
                        regular_terms.append(base_term)
            else:
                regular_terms.append(term)

        return regular_terms, interaction_terms

    def _collect_model_terms(self, **mp):
        """Collect all model terms from parameter dictionary, including interactions."""
        terms = []
        for key, values in mp.items():
            if 'terms' in key and values:
                terms += values
        return terms

    def save_fig(self, fig, name, extension='svg'):
        """Save figure with specified name and extension."""
        fig_path = self.paper_figures_path / f"{name}.{extension}"
        fig.savefig(fig_path, dpi=self.DPI, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")

    # =================== F1 ===================
    def cl_x_age_plots(self, model_CI='conditional', save_fig=False):
        """Generate cycle length vs age plots (Figure 1)."""

        with plt.rc_context(rc=self.plotting_params):
            f, ax = plt.subplots(2, 1, dpi=500, figsize=(self._age_axis_width,
                                                         self._CL_axis_height * 2 + self._h_spacing),
                                 constrained_layout=False)

            self._cl_age_panels(panel='cl_mean_age', ax=ax[0], plot_model=True, model_CI=model_CI)

            ax[0].set_xlabel('')
            ax[0].set_xticklabels('')
            ax[0].set_ylabel("Cycle Length \n[days]")
            ax[0].grid(axis='x', lw=0)

            self._cl_age_panels(panel='cl_p3_age', ax=ax[1], plot_model=True, add_counts=False, model_CI=model_CI)
            ax[1].grid(axis='x', lw=0)
            ax[1].set_ylabel("Cycle Length Deviation \n[proportion ≥ 3 days]")

            self._row_figure_labels(f, ax, x_pos=-0.04, nrows=2)

        if save_fig:
            self.save_fig(f, "fig1_cycle_length_x_age", extension='svg')
            self.save_fig(f, "fig1_cycle_length_x_age", extension='png')

        return f, ax

    def _cl_age_panels(self, panel, ax, plot_model: bool = True, plot_points: bool = True,
                       add_counts: bool = True, model_CI: str = 'conditional'):
        """Create individual panels for age-related plots."""

        model_name = panel
        mp = self.model_params[model_name]
        data = self.CBM.tables['user'].reset_index()
        m = self.get_model(model_name)
        cycle_data = mp['data']
        plot_var = mp['plot_var']

        self.CBM.plot_var_x_age(y_var=plot_var, ax=ax, lw=2 * plot_points, ms=4 * plot_points, join_points=False,
                                yticks=mp['yticks'], data=data, add_counts=add_counts)

        if plot_model:
            model_axis_rescale, model_axis_shift, eval_vals = self._get_axis_rescaling_params(self.CBM.age_bin_edges)

            preds = self.get_conditional_preds_CI(data=cycle_data, fitted_model=m, eval_term=mp['x_var'], eval_vals=eval_vals)

            self._plot_model_continous_pred(preds, x_var='age', y_var='pred',
                                            x_axis_shift=model_axis_shift,
                                            x_axis_rescale=model_axis_rescale, ax=ax)
        ax.grid(axis='x', lw=0)

    def cl_x_age_stats(self):
        """Calculate statistics for cycle length vs age relationships."""
        np.random.seed(42)
        print()
        print("------------")
        print("Mean cycle length")
        print(f"{self.CBM.tables['user'].mean_cl.mean(): 0.1f}")
        model_name = 'cl_mean_age'
        m = self.get_model(model_name)
        mp = self.model_params[model_name]
        data = mp['data']
        sph = StatisticalPredictionHandler(m, data)
        print(sph.get_conditional_predictions(eval_term='age', eval_vals=[24, 44]))
        print('contrast')
        print(sph.calculate_conditional_contrast(term_of_interest='age', values_to_compare=[24, 44]).round(1)[['contrast', 'contrast_ci_lower', 'contrast_ci_upper']])

        print()
        print("------------")
        print("Proportion of cycles with delta >= 3 days")
        print(f"{self.CBM.tables['user'].a_delta_cl_ge_3.mean(): 0.2f}")
        model_name = 'cl_p3_age'
        m = self.get_model(model_name, overwrite=True)
        mp = self.model_params[model_name]
        data = mp['data']
        sph = StatisticalPredictionHandler(m, data)
        print(sph.get_conditional_predictions(eval_term='age', eval_vals=[24, 33, 44]).round(2))
        print(sph.calculate_min_term_ci(term='age').round(2))

        print()
        print("------------")
        print("Cycle length standard deviation")
        print(f"{self.CBM.tables['user'].sd_cl.mean(): 0.2f}")
        model_name = 'cl_sd_age'
        m = self.get_model(model_name, overwrite=True)
        mp = self.model_params[model_name]
        data = mp['data']
        sph = StatisticalPredictionHandler(m, data)
        print(sph.get_conditional_predictions(eval_term='age', eval_vals=[24, 32, 33, 44]).round(2))
        print(sph.calculate_min_term_ci(term='age').round(2))

    # =================== F2/S3 (cl_x_behav) ===================
    def cl_x_behav_plots(self, behavior='sl', panel_labels=("a.", "b."), safe_fig=False):
        """
        Generate cycle length vs behavior plots.
        Generalizes to both 'sl_dur' and 'sl_dur_lvar' behaviors.
        """
        if behavior == 'sl' or behavior == 'sl_dur':
            panel1 = 'cl_mean_sl_dur'
            panel2 = 'cl_p3_sl_dur'
        elif behavior == 'sl_lvar' or behavior == 'sl_dur_lvar':
            panel1 = 'cl_mean_sl_lvar'
            panel2 = 'cl_p3_sl_lvar'
        elif behavior == 'eTRIMP' or behavior == 'wo':
            panel1 = 'cl_mean_eTRIMP'
            panel2 = 'cl_p3_eTRIMP'
            xlabel = 'Activity Levels [intensity x mins]'
        else:
            raise ValueError(f"Unsupported behavior: {behavior}")

        with plt.rc_context(rc=self.plotting_params):
            f, ax = plt.subplots(2, 1, dpi=500, figsize=(self._age_axis_width,
                                                         self._CL_axis_height * 2 + self._h_spacing),
                                 constrained_layout=False)
            self._cl_behav_panels(panel=panel1, ax=ax[0], plot_model=True)
            ax[0].set_xlabel('')
            ax[0].set_xticklabels('')
            ax[0].set_ylabel("Cycle Length \n[days]")
            ax[0].grid(axis='x', lw=0)

            self._cl_behav_panels(panel=panel2, ax=ax[1], plot_model=True, add_counts=False)
            ax[1].grid(axis='x', lw=0)
            ax[1].set_ylabel("Cycle Length Deviation \n[proportion ≥ 3 days]")

            self._row_figure_labels(f, ax, x_pos=-0.04, nrows=2)

            if behavior == 'wo':
                ax[1].set_xlabel(xlabel)

            if safe_fig:
                figname_by_behavior = {
                    'wo': "figS3_cycle_length_x_workout",
                    'eTRIMP': "figS3_cycle_length_x_workout",
                    'sl': "fig2_cycle_length_x_sleep_duration",
                    'sl_dur': "fig2_cycle_length_x_sleep_duration",
                    'sl_lvar': "fig2_cycle_length_x_sleep_variability",
                    'sl_dur_lvar': "fig2_cycle_length_x_sleep_variability",
                }
                figname = figname_by_behavior.get(behavior, f"cl_x_{behavior}")
                self.save_fig(f, figname, extension='svg')
                self.save_fig(f, figname, extension='png')
        return f, ax

    def _cl_behav_panels(self, panel, ax, plot_model: bool = True, plot_points: bool = True,
                         add_counts: bool = True, model_CI='conditional'):
        """Create individual panels for behavior-related plots."""

        model_name = panel
        mp = self.model_params[model_name]
        m = self.get_model(model_name)
        data = self._group_by_behavior(mp['data'], group_var=mp['group_var'], cycle_count_thr=1)
        x_var = mp['x_var']
        plot_var = mp['plot_var']
        bin_edges = mp['bin_edges']

        self.CBM.plot_var_x_behav(y_var=plot_var, behav_var=mp['group_var'],
                                  ax=ax, lw=2 * plot_points, ms=4 * plot_points, join_points=False,
                                  yticks=mp['yticks'], data=data, add_counts=add_counts)

        if plot_model:
            model_axis_rescale, model_axis_shift, eval_vals = self._get_axis_rescaling_params(bin_edges)

            preds = self.get_conditional_preds_CI(data=data, fitted_model=m, eval_term=x_var, eval_vals=eval_vals)

            self._plot_model_continous_pred(preds, x_var=x_var, y_var='pred',
                                            x_axis_shift=model_axis_shift,
                                            x_axis_rescale=model_axis_rescale, ax=ax)

        if x_var == 'sl_dur_lvar':
            labels = []
            vals = np.round(self.CBM.SL_DUR_SD_BINS * 60).astype(int)
            for ii, (v1, v2) in enumerate(zip(vals[:-1], vals[1:])):
                if ii == 0:
                    labels.append(f'[{v1},{v2}]')
                else:
                    labels.append(f'({v1},{v2}]')
            ax.set_xticklabels(labels)
            ax.set_xlabel("Sleep Duration S.D. [mins]")
        ax.grid(axis='x', lw=0)

    # =================== Supplemental S2/S4/S1 ===================
    def cl_x_BMI_stats(self):
        """Calculate statistics for cycle length vs BMI relationships."""
        pass

    def cl_sd_x_age_plot(self):
        """Generate cycle length standard deviation vs age plot."""

        with plt.rc_context(rc=self.plotting_params):
            f, ax = plt.subplots(1, 1, dpi=500, figsize=(self._age_axis_width, self._CL_axis_height),
                                 constrained_layout=True)

            self._cl_age_panels(panel='cl_sd_age', ax=ax,
                                plot_model=True,
                                plot_points=True)
            ax.set_ylabel("Cycle Length \nS.D. [days]")

            return f, ax

    def cl_x_bmi_plots(self, save_fig=False):
        """Generate cycle length vs BMI plots."""
        with plt.rc_context(rc=self.plotting_params):
            f, ax = plt.subplots(2, 1, dpi=500, figsize=(self._age_axis_width,
                                                         self._CL_axis_height * 2 + self._h_spacing),
                                 constrained_layout=False)

            self._cl_bmi_panels(panel='cl_mean_bmi', ax=ax[0], plot_model=True)
            ax[0].set_xlabel('')
            ax[0].set_xticklabels('')
            ax[0].set_ylabel("Cycle Length \n[days]")
            ax[0].grid(axis='x', lw=0)

            self._cl_bmi_panels(panel='cl_p3_bmi', ax=ax[1], plot_model=True, add_counts=False)
            ax[1].set_ylabel("Cycle Length Deviation \n[proportion ≥ 3 days]")

            self._row_figure_labels(f, ax, x_pos=-0.04, nrows=2)

            if save_fig:
                self.save_fig(f, "figS2_cycle_length_x_bmi", extension='svg')
                self.save_fig(f, "figS2_cycle_length_x_bmi", extension='png')

        return f, ax

    def _cl_bmi_panels(self, panel, ax, plot_model: bool = True, plot_points: bool = True,
                       add_counts=True, model_CI: str = 'conditional'):
        """Create individual panels for BMI-related plots."""

        model_name = panel
        mp = self.model_params[model_name]
        data = self.CBM.tables['user']
        m = self.get_model(model_name)
        cycle_data = mp['data']
        plot_var = mp['plot_var']

        self.CBM.plot_var_x_BMI(y_var=plot_var, ax=ax, lw=2 * plot_points, ms=4 * plot_points, join_points=False,
                                yticks=mp['yticks'], data=data, add_counts=add_counts)

        if plot_model:
            model_axis_rescale, model_axis_shift, eval_vals = self._get_axis_rescaling_params(self.CBM.bmi_bin_edges)

            model_axis_shift += 1  # Adjust shift to center the model line
            preds = self.get_conditional_preds_CI(data=cycle_data, fitted_model=m, eval_term=mp['x_var'], eval_vals=eval_vals)

            self._plot_model_continous_pred(preds, x_var='BMI', y_var='pred',
                                            x_axis_shift=model_axis_shift,
                                            x_axis_rescale=model_axis_rescale, ax=ax)
        ax.grid(axis='x', lw=0)

    def cl_x_age_dist_plots(self, save_fig=True):
        with plt.rc_context(rc=self.plotting_params):
            f, axs = plt.subplots(2, 1, dpi=500, figsize=(self._age_axis_width * 1.25,
                                                          self._CL_axis_height * 1.1 * 2 + self._h_spacing),
                                  constrained_layout=False)

            legend_fontsize = self.plotting_params.get('legend.fontsize') * 0.75

            data = self.CBM.tables['user']
            age_bins = np.arange(18, 51, 4)
            cl_bins = np.arange(21, 37, 1)
            cld_bins = np.around(np.arange(0.0, 0.9, 0.05), 2)

            age_labels = age_bins[:-1] + 1
            age_labels2 = [f'({ab},{ab + 4}]' for ab in age_bins[:-1]]
            age_labels2[0] = '[' + age_labels2[0][1:]
            cl_labels = cl_bins[:-1]
            cld_labels = cld_bins[:-1]

            ages = data['age'].values
            cls = data['mean_cl'].values
            clds = data['a_delta_cl_ge_3'].values

            age_binned = pd.cut(ages, bins=age_bins, include_lowest=True)
            cl_binned = pd.cut(cls, bins=cl_bins, include_lowest=True)
            cld_binned = pd.cut(clds, bins=cld_bins, include_lowest=True)

            # First plot -------------------------------------------------------
            ax = axs[0]
            vmin = 0
            vmax = 22

            A = pd.crosstab(age_binned, cl_binned).sort_index().astype(int).values
            A_age_norm = (A / A.sum(axis=1, keepdims=True)) * 10000 // 100

            ax = sns.heatmap(A_age_norm.T, xticklabels=age_labels, yticklabels=cl_labels, cmap='magma',
                             annot=True, fmt='g', annot_kws={'size': legend_fontsize},
                             cbar_kws={'label': 'Percentage (%)'}, vmin=vmin, vmax=vmax, ax=ax, cbar=False)

            for text in ax.texts:
                if text.get_text() == '0' or float(text.get_text()) == 0.0:
                    text.set_text('')

            counts_per_age = A.sum(axis=1)
            xticks = ax.get_xticks()
            for x, n in zip(xticks, counts_per_age):
                ax.text(x, 1.0, f'{int(n)}', ha='center', va='bottom',
                        transform=ax.get_xaxis_transform(), fontsize=legend_fontsize * 1.1)
            ax.invert_yaxis()
            ax.tick_params(axis='y', which='major', length=2, width=1, direction='out', bottom=True, left=True)

            ax.set_xticklabels([])
            yticks = ax.get_yticks() - 0.5
            ax.set_yticks(yticks[::2])
            ax.set_yticklabels(cl_labels[::2])
            ax.set_ylabel('Cycle Length [days]')

            l = self._add_colorbar_legend(ax, cmap='magma', levels=np.arange(vmin, vmax + 0.1, 1), step=5,
                                          position='right', spacing_delta=0.03, height_factor=0.5, width_factor=0.05)
            l.set_ylabel('Age Group \nPercentage (%)')

            # Second plot ------------------------------------------------------
            ax = axs[1]
            vmin = 0
            vmax = 17
            A = pd.crosstab(age_binned, cld_binned).sort_index().astype(int).values
            A_age_norm = (A / A.sum(axis=1, keepdims=True)) * 10000 // 100

            ax = sns.heatmap(A_age_norm.T, xticklabels=age_labels2, yticklabels=cld_labels, cmap='magma',
                             annot=True, fmt='g', annot_kws={'size': legend_fontsize},
                             cbar_kws={'label': 'Percentage (%)'}, cbar=False, vmin=vmin, vmax=vmax, ax=ax)

            for text in ax.texts:
                if text.get_text() == '0' or float(text.get_text()) == 0.0:
                    text.set_text('')
            ax.invert_yaxis()

            ax.tick_params(axis='both', which='major', length=2, width=1, direction='out', bottom=True, left=True)

            ax.set_xlabel('Age')
            ax.set_xticks(xticks)
            ax.set_xticklabels(age_labels2, rotation=45)

            yticks = ax.get_yticks() - 0.5
            ax.set_yticks(yticks[::2])
            ax.set_yticklabels(cld_labels[::2])
            ax.set_ylabel('Cycle Length Deviation \n [proportion ≥ 3 days]')

            l = self._add_colorbar_legend(ax, cmap='magma', levels=np.arange(vmin, vmax + 0.1, 1), step=5,
                                          position='right', spacing_delta=0.03, height_factor=0.5, width_factor=0.05)
            l.set_ylabel('Age Group \nPercentage (%)')

            self._row_figure_labels(f, axs, x_pos=-0.02, nrows=2)

            if save_fig:
                self.save_fig(f, "figS1_cycle_length_age_distribution", extension='png')

    # ----------- Auxiliary Functions --------------##
    def get_conditional_preds_CI(self, data, fitted_model, eval_term, eval_vals, alpha=0.05):
        """Wrapper around StatisticalPredictionHandler for conditional predictions."""
        handler = StatisticalPredictionHandler(fitted_model, data)
        return handler.get_conditional_predictions(eval_term, eval_vals, alpha)

    def _add_axes_labels(self, ax, labels, x_offset=0, y_offset=0):
        label_size = self.plotting_params.get('figure.labelsize')
        label_weight = self.plotting_params.get('figure.labelweight')

        for a, l in zip(ax, labels):
            a.annotate(l, xy=(0, 1), xytext=(x_offset, y_offset),
                       ha='right', va='bottom',
                       xycoords='axes fraction',
                       textcoords='offset points',
                       fontsize=label_size, fontweight=label_weight)

    def _group_by_behavior(self, data, group_var='sl_dur_bin', cycle_count_thr=1):
        """Group data by behavioral variables."""
        gdf = data.groupby(['n_id', group_var], observed=True)
        df = gdf.mean(numeric_only=True)
        df['counts'] = gdf.size()
        df['median_cl'] = gdf['length'].apply(np.nanmedian)
        df = df.reset_index()
        df = self.CBM._add_behav_cat(df)
        df = df[df['counts'] >= cycle_count_thr]
        return df

    def _plot_model_continous_pred(self, df, x_var, y_var, ax, x_axis_shift=0, x_axis_rescale=1):
        """Plot the predicted values for a continuous variable."""
        df = df.copy(deep=True)
        df[x_var] = df[x_var] * x_axis_rescale + x_axis_shift
        ax.plot(df[x_var], df[y_var], color=self._model_plot_color,
                alpha=self._model_plot_alpha,
                lw=self._model_plot_lw, ls=self._model_plot_ls, zorder=1)
        ax.fill_between(df[x_var], df[f'{y_var}_ci_lower'], df[f'{y_var}_ci_upper'],
                        alpha=self._model_plot_alpha * 0.5, color=self._model_plot_color, lw=0, zorder=1)

    def _get_axis_rescaling_params(self, bin_edges):
        """Calculate axis rescaling parameters for plotting."""

        bin_step = (bin_edges[-2] - bin_edges[1]) / (len(bin_edges) - 3)
        bin_centers = np.arange(bin_edges[1] - bin_step / 2, bin_edges[-2] + bin_step / 2 + bin_step, bin_step)

        b0 = bin_centers[0] - bin_step / 2
        b1 = bin_centers[-1] + bin_step / 2

        model_axis_rescale = 1 / bin_step
        model_axis_shift = -b0 * model_axis_rescale - 0.5
        eval_vals = np.linspace(b0, b1, 100)

        return model_axis_rescale, model_axis_shift, eval_vals

    def _row_figure_labels(self, f, ax, nrows, x_pos=0.02):
        label_size = self.plotting_params.get('font.size')
        label_weight = 'bold'

        labels = [f"{chr(97 + i)}." for i in range(nrows)]

        for kk in range(nrows):
            try:
                ax0_bbox = ax[kk].get_position()
            except (TypeError, AttributeError, IndexError):
                ax0_bbox = getattr(ax, 'get_position', lambda: None)()

            f.text(x_pos, ax0_bbox.y1, labels[kk],
                   ha='left', va='center',
                   fontsize=label_size, fontweight=label_weight)

    def _add_colorbar_legend(self, ax, cmap, levels, step, spacing_delta=0.4, position='bottom',
                             height_factor=0.15, width_factor=1.0):
        """Add a colorbar legend."""
        p = ax.get_position()
        f = ax.figure
        width = p.width * width_factor
        height = height_factor * p.height
        if position == 'bottom':
            y0 = p.y0 - spacing_delta * p.height
            x0 = p.x0
        elif position == 'top':
            y0 = p.y0 + spacing_delta * p.height
            x0 = p.x0
        elif position == 'right':
            y0 = p.y0
            x0 = p.x0 + p.width + spacing_delta
        else:
            raise ValueError("position must be 'bottom' or 'top'")
        l1 = f.add_axes([x0, y0, width, height])

        if isinstance(cmap, (list, tuple)):
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(cmap)

        if position in ['bottom', 'top']:
            l1.imshow(np.vstack((levels, levels)), aspect='auto', cmap=cmap)
            setup_axes(l1, spine_list=['bottom'])
            l1.set_xticks(np.arange(1, len(levels), step))
            l1.set_xticklabels(levels[1::step].astype(int))
            l1.yaxis.set_ticks([])
            l1.yaxis.set_ticklabels([])
        else:
            l1.imshow(np.vstack((levels, levels)).T, aspect='auto', cmap=cmap)
            setup_axes(l1, spine_list=['right'], spine_lw=0.75)
            l1.yaxis.tick_right()
            l1.yaxis.set_label_position("right")
            l1.set_yticks(np.arange(0, len(levels), step))
            l1.set_yticklabels(levels[::step].astype(int))
            l1.xaxis.set_ticks([])
            l1.xaxis.set_ticklabels([])
            l1.invert_yaxis()
        l1.grid(False)

        return l1
