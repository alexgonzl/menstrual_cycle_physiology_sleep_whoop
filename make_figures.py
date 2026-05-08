"""Regenerate every published figure from the two CSVs.

CLI:
    python make_figures.py                 # all figures
    python make_figures.py fig3 figS9      # just these two
    python make_figures.py --list          # print canonical figure names

R + rpy2 + mgcv are required (notebook 03's GAMs are part of the figure
set). On a cold-cache run, expect ~30-60 min for the GAM fit; warm runs
take a few minutes.

Cache invalidation is manual: `rm -rf data/cache/` to force a fresh
build (do this after editing the source CSV).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Silence the verbose rpy2 R-callback noise from mgcv::predict.gam
# ("Smoothness uncertainty corrected covariance not available", etc.).
try:
    from rpy2.rinterface_lib import callbacks
    callbacks.consolewrite_warnerror = lambda *args, **kwargs: None
    callbacks.consolewrite_print = lambda *args, **kwargs: None
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from menstrual_cycle_analysis import (
    Biometrics_VAR,
    CycleLengthAnalyses,
    PhysioBehaviorAnalyses,
    PhysioMethods,
    config,
    load_paper_data,
)
from menstrual_cycle_analysis.r_utils import load_gam_models, save_gam_models


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str, formats=("svg", "png")) -> None:
    """Save fig to figures/<name>.{formats}, with bbox_inches='tight' and the
    project's standard DPI for PNG. Print the names that landed.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in formats:
        path = config.FIGURES_DIR / f"{name}.{ext}"
        if ext == "png":
            fig.savefig(path, bbox_inches="tight", dpi=config.FIGURE_DPI)
        else:
            fig.savefig(path, bbox_inches="tight")
        saved.append(path.name)
    print("Saved:", " + ".join(saved))


# ---------------------------------------------------------------------------
# Context: data + CBM + cla + PM + GAMs, all loaded once.
# ---------------------------------------------------------------------------

class _Context:
    def __init__(self):
        print("Loading data...")
        self.day_df, self.CBM = load_paper_data()
        self.CBM.add_sleep_behaviors("user")
        self.CBM.add_workout_behaviors("user")
        self.CBM.add_sleep_behaviors("cycle")
        self.CBM.add_workout_behaviors("cycle")

        self.cla = CycleLengthAnalyses(CBM=self.CBM)

        self.PM = PhysioMethods(cbm=self.CBM)
        self.PM.get_reference_table()
        self.PM.process_physio_data(overwrite=True)
        self.PM.add_wo_data()
        self.PM.add_sleep_2_ref()
        self.PM.add_wo_2_ref()
        self.PM.add_acute_chronic_behaviors_data()
        self.PM.add_acute_chronic_behaviors_ref()
        self.PM.add_seasonalilty_2_ref()

        # GAMs: load if cached, else fit. Same pattern as notebook 03.
        self.PM.gam_additional_covariates = config.GAM_ADDITIONAL_COVARIATES
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._gam_metrics_path = config.MODELS_DIR / "gam_models_metrics.parquet"
        biometrics = [f"pct_{vv}" for vv in self.PM.CORE_BIOMETRICS]
        cache_complete = (
            all((config.MODELS_DIR / f"{b}.rds").exists() for b in biometrics)
            and self._gam_metrics_path.exists()
        )
        if cache_complete:
            print("Loading cached GAMs from models/...")
            self.PM.gam_models = {}
            load_gam_models(self.PM, config.MODELS_DIR, keys=biometrics)
            self.PM.gam_models_metrics = pd.read_parquet(self._gam_metrics_path)
            self.PM.prep_data_gam_cycle_model(
                prefix="pct",
                additional_covariates=self.PM.gam_additional_covariates,
            )
        else:
            print("Fitting GAMs (this takes ~30-60 min on first run)...")
            self.PM.fit_gam_models(
                additional_covariates=self.PM.gam_additional_covariates,
                overwrite=False,
            )
            save_gam_models(self.PM, config.MODELS_DIR, overwrite=False)
            self.PM.gam_models_metrics.to_parquet(self._gam_metrics_path)

        # gam_sim_data is reused by Fig 3 + S10/S11 builders.
        self._gam_sim_data_28x32 = None  # built lazily by Fig 3
        self._gam_sim_data_dense = None  # built lazily by S10
        self._out_simp = None            # built lazily by S10

        # PBA is reused by Fig 4 a/b/c + S14.
        self._PBA = None
        self._PBA_with_models = None

    @property
    def gam_sim_data_28x32(self):
        if self._gam_sim_data_28x32 is None:
            self._gam_sim_data_28x32 = self.PM.get_gam_cl_age_sim_data()
        return self._gam_sim_data_28x32

    @property
    def gam_sim_data_dense(self):
        if self._gam_sim_data_dense is None:
            self._gam_sim_data_dense = self.PM.get_gam_sim_data_for_vars(
                var_grid_dict={
                    "d": np.arange(0, 35, 0.25),
                    "age": np.arange(18, 51, 1),
                    "cl": np.arange(21, 36, 1),
                },
                se_fit=False,
            )
        return self._gam_sim_data_dense

    @property
    def out_simp(self):
        if self._out_simp is None:
            self._out_simp = self.PM.get_gam_critical_values_simple(
                self.gam_sim_data_dense,
            )
        return self._out_simp

    @property
    def PBA(self):
        """First PBA — its construction has side effects on PM (adds menstrual
        phases to reference_table). Required before the second PBA.
        """
        if self._PBA is None:
            self._PBA = PhysioBehaviorAnalyses(cla=self.cla, pm=self.PM)
        return self._PBA

    @property
    def PBA_with_models(self):
        """Second PBA, built after the behavioral-prefix process_physio_data
        and used by Fig 4 a/b/c + S14.
        """
        if self._PBA_with_models is None:
            _ = self.PBA  # ensure first PBA's side effects run
            self.PM.process_physio_data(
                preset="behavioral", prefix="behavioral",
                types=["pct"], overwrite=True,
            )
            self._PBA_with_models = PhysioBehaviorAnalyses(
                cla=self.cla, pm=self.PM, physio_prefix="pct_behavioral",
            )
            self._PBA_with_models.f_label_buffer = 0.025
            self._PBA_with_models.get_physio_behav_change_models(
                behav_var="acr_sleep_dur", physio_prefix="pct_behavioral",
            )
        return self._PBA_with_models


# ---------------------------------------------------------------------------
# Per-figure builders
# ---------------------------------------------------------------------------
# Filenames mirror the names committed under figures/. The corresponding
# package method is named in a comment for each builder.

def _build_fig1(ctx):
    """Cycle length × age (paper_code_wrapper.cl_x_age_plots)."""
    f, _ = ctx.cla.cl_x_age_plots(save_fig=False)
    _save(f, "fig1_cycle_length_x_age")
    plt.close(f)
    print("--- fig1: cycle-length × age stats ---")
    ctx.cla.cl_x_age_stats()


def _build_fig2(ctx):
    """Cycle length × sleep (paper_code_wrapper.cl_x_sleep_plots)."""
    ctx.cla._setup_layout_params(dict(w_spacing=0.3, h_spacing=0.3))
    ctx.cla.cl_x_sleep_plots(save_fig=False)
    f = plt.gcf()
    _save(f, "fig2_cycle_length_x_sleep")
    plt.close(f)
    print("--- fig2: sleep-duration GEE summaries ---")
    for panel in ("cl_mean_sl_dur", "cl_p3_sl_dur",
                  "cl_mean_sl_lvar", "cl_p3_sl_lvar"):
        m = ctx.cla.get_model(panel)
        print(f"\n## {panel}")
        print(m.summary())
        ctx.cla.cl_x_behav_stats(
            panel=panel,
            get_min=panel.endswith("sl_dur"),
            calculate_within_subject_contrast=True,
        )


def _build_fig3(ctx):
    """GAM biometrics × age × cycle length (physio_methods.plot_gam_biometrics_cl_age)."""
    f, axes = ctx.PM.plot_gam_biometrics_cl_age(gam_data=ctx.gam_sim_data_28x32)
    # Tighter blood_oxygen y-axis on the bottom row.
    for jj in range(2):
        axes[4, jj].set_ylim([-0.12, 0.12])
        axes[4, jj].set_yticks([-0.1, 0, 0.1])
        axes[4, jj].set_yticklabels([-0.1, 0, 0.1])
    _save(f, "fig3_gam_biometrics_cl_age")
    plt.close(f)
    print("--- fig3: GAM age contrasts ---")
    for biom in ctx.PM.CORE_BIOMETRICS:
        print("-" * 80)
        ctx.PM.print_gam_age_contrast(biometric=f"pct_{biom}")
    print("\n--- fig3: GAM cycle-length contrasts ---")
    for biom in ctx.PM.CORE_BIOMETRICS:
        print("-" * 80)
        ctx.PM.print_gam_cl_contrast(biometric=f"pct_{biom}")


def _build_fig4(ctx):
    """Sleep × phase natural experiment, three subpanels (a/b/c)."""
    pba = ctx.PBA_with_models

    # 4a: sleep-duration change distribution by phase
    f, _ = pba.plot_phase_behav_change(
        behav_var="acr_sleep_dur", figure_label="a.",
        save_fig=False, filename_prefix="fig4a_sleep_dur_histograms",
    )
    _save(f, "fig4a_sleep_dur_histograms")
    plt.close(f)

    # 4b: RHR by cycle day per phase (continuous)
    f, _ = pba.plot_physio_behav_change_by_phase_continous(
        behav_var="acr_sleep_dur", physio_var="RHR",
        figure_label="b.", save_fig=False,
        filename_prefix="fig4b_RHR_by_cycle_day_per_phase",
    )
    _save(f, "fig4b_RHR_by_cycle_day_per_phase")
    plt.close(f)

    # 4c: RHR vs sleep change model
    f, _ = pba.plot_model_physio_response_x_phase(
        physio_var="RHR", save=False, figure_label="c.",
        filename_prefix="fig4c_RHR_vs_sleep_change_per_phase",
    )
    _save(f, "fig4c_RHR_vs_sleep_change_per_phase")
    plt.close(f)

    print("--- fig4: phase × sleep-change stats ---")
    pba.phase_behav_change_stats()


def _build_figS1(ctx):
    """Cycle length × age distribution (paper_code_wrapper.cl_x_age_dist_plots)."""
    f, _ = ctx.cla.cl_x_age_dist_plots(save_fig=False)
    _save(f, "figS1_cycle_length_age_distribution", formats=("png",))
    plt.close(f)


def _build_figS2(ctx):
    """Cycle length × BMI (paper_code_wrapper.cl_x_bmi_plots)."""
    f, _ = ctx.cla.cl_x_bmi_plots(save_fig=False)
    _save(f, "figS2_cycle_length_x_bmi")
    plt.close(f)


def _build_figS3(ctx):
    """Cycle length × workout (paper_code_wrapper.cl_x_behav_plots, behavior='wo')."""
    f, _ = ctx.cla.cl_x_behav_plots(behavior="wo", safe_fig=False)
    _save(f, "figS3_cycle_length_x_workout")
    plt.close(f)


def _build_figS4(ctx):
    """Cycle length SD × age (paper_code_wrapper.cl_sd_x_age_plot)."""
    f, _ = ctx.cla.cl_sd_x_age_plot()
    _save(f, "figS4_cycle_length_sd_x_age")
    plt.close(f)


def _build_figS5(ctx):
    """Behaviors × age × BMI (paper_code_wrapper.behav_x_age_bmi_plots)."""
    f, _ = ctx.cla.behav_x_age_bmi_plots(save_fig=False)
    _save(f, "figS5_behaviors_x_age_bmi")
    plt.close(f)


def _build_figS6(ctx):
    """User-level biometrics vs age (physio_methods.plot_user_level_biometrics_vs_x)."""
    f, _ = ctx.PM.plot_user_level_biometrics_vs_x(prefix="b", x_var="age_b")
    _save(f, "figS6_b_user_level_biometrics_vs_age")
    plt.close(f)


def _build_figS7(ctx):
    """User-level biometrics vs BMI."""
    f, _ = ctx.PM.plot_user_level_biometrics_vs_x(prefix="b", x_var="BMI_b2")
    _save(f, "figS7_b_user_level_biometrics_vs_BMI")
    plt.close(f)


def _build_figS8(ctx):
    """Empirical biometrics × age × cycle length (physio_methods.plot_biometrics_cl_age)."""
    f, _ = ctx.PM.plot_biometrics_cl_age(prefix="pct")
    _save(f, "figS8_pct_biometrics_cl_age_data")
    plt.close(f)


def _build_figS9(ctx):
    """VAR(3) inter-biometric residual correlations (pca_analyses.Biometrics_VAR)."""
    BV = Biometrics_VAR(pm=ctx.PM)
    BV.prepare_data()
    BV.fit_model(lag_order=3)
    print("--- figS9: residual correlation matrix ---")
    print(BV.get_residual_corr_matrix().round(2))
    f, _ = BV.plot_model_res_matrices()
    _save(f, "figS9_var_residual_correlations")
    plt.close(f)


def _build_figS10(ctx):
    """Critical-value heatmaps (physio_methods.plot_biometrics_critical_values_heatmaps2). PNG only."""
    f, _ = ctx.PM.plot_biometrics_critical_values_heatmaps2(ctx.out_simp)
    _save(f, "figS10_critical_values_heatmaps", formats=("png",))
    plt.close(f)
    print("--- figS10: CL contrasts at age=32 ---")
    cl1, cl2, age = 24, 34, 32
    print("=" * 80)
    print(f"CL Contrast at age={age}")
    for biom in ctx.PM.CORE_BIOMETRICS:
        print("-" * 80)
        A = ctx.out_simp[biom].min_loc.unstack().T
        print(f"{biom} min locs {A.loc[age].round()[[cl1, cl2]]}\n")
        A = ctx.out_simp[biom].max_loc.unstack().T
        print(f"{biom} max locs {A.loc[age].round()[[cl1, cl2]]}\n")
    print("=" * 80)
    print("\n--- figS10: age contrasts at cl=28 ---")
    age1, age2 = 24, 44
    for cl in [28]:
        print("=" * 40)
        print(f"Age Contrast at cl={cl}")
        for biom in ctx.PM.CORE_BIOMETRICS:
            print("-" * 40)
            A = ctx.out_simp[biom].min_loc.unstack().T
            print(f"{biom} min locs {A[cl].round().loc[[age1, age2]]}\n")
            A = ctx.out_simp[biom].max_loc.unstack().T
            print(f"{biom} max locs {A[cl].round().loc[[age1, age2]]}\n")
        print("=" * 40)


def _build_figS11(ctx):
    """Magnitude range × cycle length × age (physio_methods.plot_max_min_*). PNG only."""
    gam_results = ctx.PM.gam_biom_delta_x_cl_age()
    within_cycle_results = ctx.PM.model_biom_delta_x_cl_age()
    f, _ = ctx.PM.plot_max_min_biometric_cycle_age_cl_variation_gam(
        prefix="pct",
        gam_results=gam_results,
        within_cycle_results=within_cycle_results,
    )
    _save(f, "figS11_pct_delta_mag_cl_age", formats=("png",))
    plt.close(f)


def _build_figS12(ctx):
    """Individual-cycle traces at three CLs (physio_methods.individual_biometrics_x_cl)."""
    f, _ = ctx.PM.individual_biometrics_x_cl()
    _save(f, "figS12_individual_biometrics_x_cl")
    plt.close(f)


def _build_figS13(ctx):
    """User-level biometrics vs sleep duration."""
    f, axes = ctx.PM.plot_user_level_biometrics_vs_x(
        prefix="b", x_var="sl_dur_mean_bin",
    )
    axes[-1, 0].set_xticklabels(ctx.CBM.sl_dur_labels, rotation=45)
    axes[-1, 1].set_xticklabels(ctx.CBM.sl_dur_labels, rotation=45)
    _save(f, "figS13_b_user_level_biometrics_vs_sl_dur_mean_bin")
    plt.close(f)


def _build_figS14(ctx):
    """All-biometrics response per phase (physio_behavior_analyses.plot_model_physio_response_x_phase_all)."""
    pba = ctx.PBA_with_models
    f, _ = pba.plot_model_physio_response_x_phase_all(
        save=False, filename_prefix="figS14_all_biometrics_response_per_phase",
    )
    _save(f, "figS14_all_biometrics_response_per_phase")
    plt.close(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURE_BUILDERS = {
    "fig1":   _build_fig1,
    "fig2":   _build_fig2,
    "fig3":   _build_fig3,
    "fig4":   _build_fig4,
    "figS1":  _build_figS1,
    "figS2":  _build_figS2,
    "figS3":  _build_figS3,
    "figS4":  _build_figS4,
    "figS5":  _build_figS5,
    "figS6":  _build_figS6,
    "figS7":  _build_figS7,
    "figS8":  _build_figS8,
    "figS9":  _build_figS9,
    "figS10": _build_figS10,    # PNG only (heatmap)
    "figS11": _build_figS11,    # PNG only (heatmap)
    "figS12": _build_figS12,
    "figS13": _build_figS13,
    "figS14": _build_figS14,
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate published figures into figures/. "
            "Cache invalidation is manual: rm -rf data/cache/."
        ),
    )
    parser.add_argument(
        "figures", nargs="*",
        help="Figure names to build (e.g. fig3 figS9). Default: all.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print canonical figure names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for name in FIGURE_BUILDERS:
            print(name)
        return

    targets = args.figures or list(FIGURE_BUILDERS)
    unknown = [t for t in targets if t not in FIGURE_BUILDERS]
    if unknown:
        parser.error(
            f"Unknown figure(s): {', '.join(unknown)}. "
            f"Use --list to see valid names."
        )

    np.random.seed(0)
    ctx = _Context()
    for name in targets:
        print(f"\n========== {name} ==========")
        FIGURE_BUILDERS[name](ctx)


if __name__ == "__main__":
    main()
