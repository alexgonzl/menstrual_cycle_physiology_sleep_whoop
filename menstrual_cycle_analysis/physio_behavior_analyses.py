"""`PhysioBehaviorAnalyses` — port of source class for notebook 05 (Fig 4 + S14).

Source: `whoop_analyses/whoop_analyses/paper_code_wrapper.py:1250-end`.

Phase E-1 slice: __init__, plot_phase_behav_change, phase_behav_change_stats,
_set_phase_figure_layout, save_fig.

Phase E-2/E-3 methods (get_physio_behav_change_models,
plot_model_physio_response_x_phase, plot_physio_behav_change_by_phase_continous,
plot_model_physio_response_x_phase_all) will be added in subsequent phases.

Modifications from source:
  - paper_figures_path comes from config.FIGURES_DIR rather than `wt`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import config
from .paper_code_wrapper import CycleLengthAnalyses
from .physio_methods import PhysioBehavChangeMethods, PhysioMethods
from .stats.contrasts import StatisticalPredictionHandler


class PhysioBehaviorAnalyses:
    def __init__(self, cla: CycleLengthAnalyses,
                 pm: PhysioMethods, physio_prefix='pct'):

        self.cla = cla
        self.pm = pm
        self.physio_prefix = physio_prefix

        pm.get_reference_table()
        pm.process_physio_data()

        self.CORE_BIOMETRICS = pm.CORE_BIOMETRICS
        self.reference_table = pm.reference_table
        self.plotting_params = cla.plotting_params
        self.DPI = cla.DPI

        self.pbcm = PhysioBehavChangeMethods(pm=pm)
        self.pbcm._define_behav_change_variables()
        self.pbcm.get_behav_change_summary_by_phase(behav_var='acr_sleep_dur', physio_prefix=physio_prefix)
        self.pbcm.get_behav_change_summary_by_phase(behav_var='acr_eTRIMP', physio_prefix=physio_prefix)

        self.behavior_models = {}
        self.behav_physio_models = {}

        self.general_terms = ['age', 'age2', 'BMI', 'BMI2', 'phase',
                              'cycle_length', 'cos_season', 'sin_season']
        self.sleep_terms = ['sl_dur_mean', 'sl_dur_mean2', 'sl_onset_cos',
                            'sl_onset_sin', 'sl_onset_var', 'sl_dur_lvar',
                            'sl_dur_lvar2']
        self.wo_terms = ['wo_dur_mean', 'wo_norm_int_mean', 'wo_eTRIMP_mean',
                         'wo_eTRIMP_mean2', 'wo_time_cos', 'wo_time_sin']

        self.paper_figures_path = config.FIGURES_DIR

        self.f_label_buffer = 0.02
        self.figsize = (6, 1.2)

    def save_fig(self, fig, name, extension='svg'):
        fig_path = self.paper_figures_path / f"{name}.{extension}"
        fig.savefig(fig_path, dpi=self.DPI, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")

    def _set_phase_figure_layout(self):
        f = plt.figure(figsize=self.figsize, dpi=self.plotting_params.get('figure.dpi'))

        x0 = 0.07
        wspace = 0.03
        width = (1 - x0 - 2 * wspace) / 3
        y0 = 0.15
        height = 0.8
        axes = [
            f.add_axes([x0, y0, width, height]),
            f.add_axes([x0 + width + wspace, y0, width, height]),
            f.add_axes([x0 + 2 * width + 2 * wspace, y0, width, height]),
        ]
        return f, axes

    # =================== Fig 4a ===================
    def plot_phase_behav_change(self, behav_var='acr_sleep_dur', figure_label=None,
                                save_fig=True, filename_prefix=None):
        with plt.rc_context(rc=self.plotting_params):
            fig, axes = self._set_phase_figure_layout()
            fig, _ = self.pbcm.plot_behav_change_x_phase(behav_var=behav_var,
                                                        figure_label=None, fig=fig, axes=axes)

            if figure_label is not None:
                label_size = self.plotting_params.get('font.size')
                label_weight = 'bold'
                fig.text(self.f_label_buffer, 1, figure_label, ha="right", va="bottom", fontsize=label_size,
                         fontweight=label_weight, transform=fig.transFigure)

            if save_fig:
                self.save_fig(fig, filename_prefix, extension='svg')
                self.save_fig(fig, filename_prefix, extension='png')
        return fig, axes

    # =================== Stats ===================
    def phase_behav_change_stats(self, behav_var='acr_sleep_dur'):
        df = self.pbcm.behav_change_phase_data[behav_var]

        y_var = behav_var
        if behav_var == 'acr_sleep_dur':
            remove_terms = ['sl_dur_mean', 'sl_dur_mean2']
        elif behav_var == 'acr_eTRIMP':
            remove_terms = ['wo_eTRIMP_mean', 'wo_eTRIMP_mean2']

        chronic_term = f'chronic_{y_var}'
        interaction_terms = [f'phase:{chronic_term}']
        seasonal_terms = ['cos_season', 'sin_season']

        all_individual_terms = self.general_terms + self.sleep_terms + self.wo_terms + [chronic_term]
        all_individual_terms = [t for t in all_individual_terms if t not in remove_terms]

        all_terms = all_individual_terms + interaction_terms + seasonal_terms
        formula = f"{y_var} ~ {' + '.join(all_terms)}"

        df = df.dropna(subset=all_individual_terms + [y_var])

        print("Number of subjects:", df['n_id'].nunique())
        print("Number of observations:", len(df))
        print("Number of cycles:", len(df.groupby(['n_id', 'j_cycle_num']).size()))

        ## Mean Difference
        self.behavior_models[behav_var] = {}
        print(f"Calculating mean difference model for {behav_var} across phases")
        m = smf.gee(formula, data=df, groups=df['n_id'], family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable()).fit()

        sph = StatisticalPredictionHandler(m, df)
        result = sph.calculate_conditional_contrast('phase', values_to_compare=['premenstrual', 'menstrual', 'postmenstrual'])
        print(result.round(2).T)

        if behav_var == 'acr_sleep_dur':
            print()
            print("Mean sleep duration change difference across phases in minutes")
            print(result['contrast'] * 7.5 * 60 / 100)

        print("------------------")
        print()

        self.behavior_models[behav_var]['mean_diff'] = m

        ## Large Decrease
        print(f"Calculating large decrease model for {behav_var} across phases")
        m = smf.gee(f"large_decrease ~ {' + '.join(all_terms)}", data=df, groups=df['n_id'],
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()

        sph = StatisticalPredictionHandler(m, df)
        rc = sph.calculate_conditional_contrast('phase', values_to_compare=['premenstrual', 'menstrual', 'postmenstrual']).round(4).T
        print(rc)

        print("Reverse odds")
        print((1 / rc.loc[['odds_ratio', 'ci_l_or', 'ci_u_or']]).round(2))
        print("------------------")
        print()

        self.behavior_models[behav_var]['large_decrease'] = m

        ## Large Increase
        print(f"Calculating large increase model for {behav_var} across phases")
        m = smf.gee(f"large_increase ~ {' + '.join(all_terms)}", data=df, groups=df['n_id'],
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()

        sph = StatisticalPredictionHandler(m, df)
        rc = sph.calculate_conditional_contrast('phase', values_to_compare=['premenstrual', 'menstrual', 'postmenstrual']).round(4).T
        print(rc)
        print("Reverse odds")
        print((1 / rc.loc[['odds_ratio', 'ci_l_or', 'ci_u_or']]).round(2))
        self.behavior_models[behav_var]['large_increase'] = m
