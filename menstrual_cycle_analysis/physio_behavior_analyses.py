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
from ._plot_utils import setup_axes, single_var_point_plot
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

    # =================== Fig 4b/4c models + plots ===================
    def get_physio_behav_change_models(self, behav_var='acr_sleep_dur', physio_prefix=None):
        """Fit per-biometric GEEs of physiological response to behavior change."""
        if behav_var == 'acr_sleep_dur':
            remove_terms = ['sl_dur_mean', 'sl_dur_mean2']
        elif behav_var == 'acr_eTRIMP':
            remove_terms = ['wo_eTRIMP_mean', 'wo_eTRIMP_mean2']

        interaction_terms = [
            'phase:acr',
            'phase:acr2',
            'age:cycle_length',
            'phase:chronic',
            'phase:cycle_length',
            'chronic:acr',
        ]
        weight_var = 'weights'

        seasonal_terms = ['cos_season', 'sin_season', 'weekend']
        all_individual_terms = self.general_terms + self.sleep_terms + self.wo_terms + ['acr', 'acr2', 'chronic']
        all_individual_terms = [t for t in all_individual_terms if t not in remove_terms]
        all_terms = all_individual_terms + interaction_terms + seasonal_terms

        df = self.pbcm.behav_change_phase_data[behav_var]
        df['chronic'] = df[f'chronic_{behav_var}']
        df['acr'] = df[f"{behav_var}"]
        df['acr2'] = df[f"{behav_var}2"]

        if physio_prefix is None:
            physio_prefix = self.physio_prefix

        models = {}

        for biometric in self.CORE_BIOMETRICS:
            physio_var = f"{physio_prefix}_{biometric}"

            df2 = df.copy()
            df2.dropna(subset=all_individual_terms + [physio_var, weight_var], inplace=True)

            print("\n--------------------")
            print(f"Fitting model for {biometric}:")
            print("Number of subjects:", df2['n_id'].nunique())
            print("Number of observations:", len(df2))
            print("Number of cycles:", len(df2.groupby(['n_id', 'j_cycle_num']).size()))
            print("---------------------\n")

            formula = f"{physio_var} ~ 1+ " + " + ".join(all_terms)
            models[biometric] = smf.gee(formula, data=df2, groups=df2['n_id'],
                                        family=sm.families.Gaussian(),
                                        cov_struct=sm.cov_struct.Exchangeable(),
                                        weights=df2[weight_var]).fit()

        self.behav_physio_models[behav_var] = models
        return models

    def plot_physio_behav_change_by_phase_continous(self, behav_var='acr_sleep_dur', physio_var='RHR',
                                                    physio_prefix=None, figure_label=None,
                                                    save_fig=False, filename_prefix=None):
        """Fig 4b — wraps pbcm.plot_physio_behav_change_by_phase."""
        if physio_prefix is None:
            physio_prefix = self.physio_prefix

        with plt.rc_context(rc=self.plotting_params):
            fig, axes = self._set_phase_figure_layout()
            fig, axes = self.pbcm.plot_physio_behav_change_by_phase(
                behav_var=behav_var, physio_var=physio_var, physio_prefix=physio_prefix,
                add_phase_title=False, figure_label=None, fig=fig, axes=axes)

            if figure_label is not None:
                label_size = self.plotting_params.get('font.size')
                label_weight = 'bold'
                fig.text(self.f_label_buffer, 1, figure_label, ha="right", va="top",
                         fontsize=label_size, fontweight=label_weight,
                         transform=fig.transFigure)

            if save_fig:
                self.save_fig(fig, filename_prefix, extension='svg')
                self.save_fig(fig, filename_prefix, extension='png')
            return fig, axes

    def plot_model_physio_response_x_phase(self,
                                           behav_var='acr_sleep_dur', physio_var='RHR',
                                           physio_prefix=None, add_phase_title=False,
                                           figure_label=None, save=False, filename_prefix=None,
                                           show_xticks=True):
        """Fig 4c — model line + per-bin scatter, per phase, for a single biometric."""
        if physio_prefix is None:
            physio_prefix = self.physio_prefix
        prefix_physio_var = f'{physio_prefix}_{physio_var}'

        df = self.pbcm.behav_change_phase_data[behav_var]
        cfg = self.pbcm.behav_configs[behav_var]

        acr_eval_vals = cfg['acr_eval_vals']
        label_mapping = cfg['label_mapping']

        model_rescale, model_shift, model_evals = self.cla._get_axis_rescaling_params(cfg['group_bins'])

        model = self.behav_physio_models[behav_var][physio_var]
        out = pd.DataFrame()
        for phase in df['phase'].unique():
            sph = StatisticalPredictionHandler(model, df[df.phase == phase])
            temp = sph.get_conditional_predictions('acr', eval_vals=acr_eval_vals)
            temp['phase'] = phase
            out = pd.concat([out, temp], ignore_index=True)

        ylim_config = {
            'pct_RHR': {'ylim': [-2.5, 2.5], 'yticks': [-2, 0, 2]},
            'pct_RR': {'ylim': [-0.8, 0.8], 'yticks': [-0.6, 0, 0.6]},
            'pct_skin_temp': {'ylim': [-0.4, 0.4], 'yticks': [-0.3, 0, 0.3]},
            'pct_HRV': {'ylim': [-8, 8], 'yticks': [-6, 0, 6]},
            'pct_blood_oxygen': {'ylim': [-0.2, 0.2], 'yticks': [-0.1, 0, 0.1]},
        }

        ylim_config |= {f"pct_behavioral_{kk}": ylim_config[f"pct_{kk}"] for kk in self.CORE_BIOMETRICS}
        ylim_config |= {'pct_behavioral_RHR': {'ylim': [-3, 3], 'yticks': [-2.5, 0, 2.5]}}

        with plt.rc_context(rc=self.plotting_params):

            fig, axes = self._set_phase_figure_layout()
            for ii, phase in enumerate(self.pbcm.phases):
                ax = axes[ii]
                setup_axes(ax)

                phase_data = df[df['phase'] == phase].copy()

                offset = phase_data.loc[phase_data[f'{behav_var}_bin_groups'] == 'no_change', prefix_physio_var].mean()
                physio_var_offset = f'{prefix_physio_var}_offset'
                phase_data[physio_var_offset] = phase_data[prefix_physio_var] - offset

                subj_phase_data = (phase_data.groupby(['n_id', f'{behav_var}_bin_groups'], observed=True)
                                   [physio_var_offset].mean().reset_index())

                for change_group, color in self.pbcm.change_colors.items():
                    group_data = subj_phase_data[subj_phase_data[f'{behav_var}_bin_groups'] == change_group].dropna()
                    group_data[f'{behav_var}_bin_groups'].cat.remove_unused_categories()
                    if len(group_data) > 0:
                        single_var_point_plot(
                            data=group_data,
                            x_var=f'{behav_var}_bin_groups',
                            y_var=physio_var_offset,
                            ax=ax,
                            join_points=False,
                            color=color,
                            marker_edge_color='0.4',
                            ms=5,
                            lw=2,
                            dy_factor=None
                        )

                offset_x = 0.9 if physio_var in ['HRV', 'blood_oxygen'] else 0.05
                offset_ha = 'right' if physio_var in ['HRV', 'blood_oxygen'] else 'left'
                ax.text(
                    offset_x, 0.1,
                    f'Offset: {offset:.2f}',
                    ha=offset_ha, va='bottom',
                    transform=ax.transAxes,
                    fontsize=9, color='0.2'
                )

                ax.set_xticks(np.arange(len(label_mapping)))
                ax.grid(axis='x', linestyle='None', color='1')
                if show_xticks:
                    ax.set_xlabel(cfg['xlabel'])
                    ax.set_xticklabels(list(label_mapping.values()), rotation=45)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel('')

                if add_phase_title:
                    ax.text(
                        0.5, 1.05, phase.capitalize(),
                        ha='center', va='bottom',
                        fontsize=self.plotting_params.get('font.size'),
                        fontweight='bold',
                        transform=ax.transAxes
                    )

                phase_out = out[out['phase'] == phase].copy()
                phase_out['acr'] = phase_out['acr'] * model_rescale + model_shift

                for kk in ['pred', 'pred_ci_lower', 'pred_ci_upper']:
                    phase_out[kk + '_offset'] = phase_out[kk] - offset

                ax.plot(phase_out['acr'], phase_out['pred_offset'],
                        color='#2964d9',
                        lw=2, label='Model fit', zorder=1, alpha=0.5)
                ax.fill_between(
                    phase_out['acr'],
                    phase_out['pred_ci_lower_offset'],
                    phase_out['pred_ci_upper_offset'],
                    color='#2964d9', alpha=0.15, label='95% CI', lw=0, zorder=1
                )

                if prefix_physio_var in ylim_config:
                    ax.set_ylim(ylim_config[prefix_physio_var]['ylim'])
                    ax.set_yticks(ylim_config[prefix_physio_var]['yticks'])

                if ii != 0:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                else:
                    yticklabels = [f"{tick:.1f}" for tick in ax.get_yticks()]
                    ax.set_yticklabels(yticklabels)
                    ax.set_ylabel(f"{self.pbcm.plotting_physio_labels_short[physio_var]} [%]")

            if figure_label is not None:
                label_size = self.plotting_params.get('font.size')
                label_weight = 'bold'
                fig.text(self.f_label_buffer, 1, figure_label, ha="right", va="top",
                         fontsize=label_size, fontweight=label_weight,
                         transform=fig.transFigure)

            if save:
                self.save_fig(fig, filename_prefix, extension='svg')
                self.save_fig(fig, filename_prefix, extension='png')

            return fig, axes

    # =================== S14 (5-biometric grid) ===================
    def plot_model_physio_response_x_phase_all(self,
                                               behav_var='acr_sleep_dur', physio_prefix=None,
                                               save=False, filename_prefix=None):
        """S14 — 5x3 grid (biometrics x phases) of model fit + per-bin scatter."""
        if physio_prefix is None:
            physio_prefix = self.physio_prefix

        df = self.pbcm.behav_change_phase_data[behav_var]
        cfg = self.pbcm.behav_configs[behav_var]

        phases = df['phase'].unique()
        acr_eval_vals = cfg['acr_eval_vals']
        label_mapping = cfg['label_mapping']

        model_rescale, model_shift, model_evals = self.cla._get_axis_rescaling_params(cfg['group_bins'])

        ylim_config = {
            'pct_RHR': {'ylim': [-2.5, 2.5], 'yticks': [-2, 0, 2]},
            'pct_RR': {'ylim': [-0.8, 0.8], 'yticks': [-0.6, 0, 0.6]},
            'pct_skin_temp': {'ylim': [-0.4, 0.4], 'yticks': [-0.3, 0, 0.3]},
            'pct_HRV': {'ylim': [-8, 8], 'yticks': [-6, 0, 6]},
            'pct_blood_oxygen': {'ylim': [-0.2, 0.2], 'yticks': [-0.1, 0, 0.1]},
        }
        ylim_config |= {
            'pct_behavioral_RHR': {'ylim': [-3, 3], 'yticks': [-2.5, 0, 2.5]},
            'pct_behavioral_HRV': {'ylim': [-8, 8], 'yticks': [-6, 0, 6]},
            'pct_behavioral_RR': {'ylim': [-1.4, 1.4], 'yticks': [-1.0, 0, 1.0]},
            'pct_behavioral_skin_temp': {'ylim': [-0.55, 0.55], 'yticks': [-0.4, 0, 0.4]},
            'pct_behavioral_blood_oxygen': {'ylim': [-0.4, 0.4], 'yticks': [-0.3, 0, 0.3]},
        }

        with plt.rc_context(rc=self.plotting_params):
            fig, axes = plt.subplots(5, 3, figsize=(6, 6), sharex='col', sharey='row', constrained_layout=True)

            for jj, physio_var in enumerate(self.CORE_BIOMETRICS):
                prefix_physio_var = f'{physio_prefix}_{physio_var}'
                model = self.behav_physio_models[behav_var][physio_var]
                out = pd.DataFrame()
                for phase in phases:
                    sph = StatisticalPredictionHandler(model, df[df.phase == phase])
                    temp = sph.get_conditional_predictions('acr', eval_vals=acr_eval_vals)
                    temp['phase'] = phase
                    out = pd.concat([out, temp], ignore_index=True)

                row_axes = axes[jj, :]
                for ii, phase in enumerate(phases):
                    ax = row_axes[ii]
                    setup_axes(ax)

                    phase_data = df[df['phase'] == phase].copy()

                    offset = phase_data.loc[phase_data[f'{behav_var}_bin_groups'] == 'no_change', prefix_physio_var].mean()
                    physio_var_offset = f'{prefix_physio_var}_offset'
                    phase_data[physio_var_offset] = phase_data[prefix_physio_var] - offset

                    subj_phase_data = (phase_data.groupby(['n_id', f'{behav_var}_bin_groups'], observed=True)
                                       [physio_var_offset].mean().reset_index())

                    for change_group, color in self.pbcm.change_colors.items():
                        group_data = subj_phase_data[subj_phase_data[f'{behav_var}_bin_groups'] == change_group].dropna()
                        group_data[f'{behav_var}_bin_groups'].cat.remove_unused_categories()
                        if len(group_data) > 0:
                            single_var_point_plot(
                                data=group_data,
                                x_var=f'{behav_var}_bin_groups',
                                y_var=physio_var_offset,
                                ax=ax,
                                join_points=False,
                                color=color,
                                marker_edge_color='0.4',
                                ms=5,
                                lw=2,
                                dy_factor=None
                            )

                    offset_x = 0.9 if physio_var in ['HRV', 'blood_oxygen'] else 0.05
                    offset_ha = 'right' if physio_var in ['HRV', 'blood_oxygen'] else 'left'
                    ax.text(
                        offset_x, 0.1,
                        f'Offset: {offset:.2f}',
                        ha=offset_ha, va='bottom',
                        transform=ax.transAxes,
                        fontsize=9, color='0.2'
                    )

                    ax.set_xticks(np.arange(len(label_mapping)))
                    ax.grid(axis='x', linestyle='None', color='1')
                    ax.set_xticklabels([])
                    ax.set_xlabel('')

                    phase_out = out[out['phase'] == phase].copy()
                    phase_out['acr'] = phase_out['acr'] * model_rescale + model_shift

                    for kk in ['pred', 'pred_ci_lower', 'pred_ci_upper']:
                        phase_out[kk + '_offset'] = phase_out[kk] - offset

                    ax.plot(phase_out['acr'], phase_out['pred_offset'],
                            color='#2964d9',
                            lw=2, label='Model fit', zorder=1, alpha=0.5)
                    ax.fill_between(
                        phase_out['acr'],
                        phase_out['pred_ci_lower_offset'],
                        phase_out['pred_ci_upper_offset'],
                        color='#2964d9', alpha=0.15, label='95% CI', lw=0, zorder=1
                    )

                    if prefix_physio_var in ylim_config:
                        ax.set_ylim(ylim_config[prefix_physio_var]['ylim'])
                        ax.set_yticks(ylim_config[prefix_physio_var]['yticks'])

                        if ii != 0:
                            ax.set_ylabel('')
                        else:
                            ax.set_ylabel(f"{self.pbcm.plotting_physio_labels_short[physio_var]} [%]")

                        yticklabels = [f"{tick:.1f}" for tick in ax.get_yticks()]
                        ax.set_yticklabels(yticklabels)

            for ax, phase in zip(axes[0, :], phases):
                ax.text(
                    0.5, 1.05, phase.capitalize(),
                    ha='center', va='bottom',
                    fontsize=self.plotting_params.get('font.size'),
                    fontweight='bold',
                    transform=ax.transAxes)

            for ax in axes[-1, :]:
                ax.set_xlabel(cfg['xlabel'])
                ax.set_xticklabels(list(label_mapping.values()), rotation=45)

            if save:
                self.save_fig(fig, filename_prefix, extension='svg')
                self.save_fig(fig, filename_prefix, extension='png')

            return fig, axes
