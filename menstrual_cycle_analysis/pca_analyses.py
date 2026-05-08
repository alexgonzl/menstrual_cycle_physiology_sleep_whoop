"""`Biometrics_VAR` — port of the VAR(3) inter-biometric residual analysis
used to produce S9.

Source: `whoop_analyses/whoop_analyses/pca_analyses.py:306-409`.

Method bodies are byte-identical to the source.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.api import VAR
from tqdm import tqdm

from .physio_methods import PhysioMethods


class Biometrics_VAR:
    def __init__(self, pm: PhysioMethods):
        self.pm = pm
        self.feature_names = [f"f_{biom}" for biom in self.pm.CORE_BIOMETRICS]

    def prepare_data(self):
        filter_fn = self.pm._filters['var']
        X = self.pm.data[self.pm.CORE_BIOMETRICS + ['n_id']]
        Y = X.copy(deep=True)
        Y[self.feature_names] = 0
        for user in tqdm(self.pm.users):
            for biom in self.pm.CORE_BIOMETRICS:
                idx = X['n_id'] == user
                y = X[idx][biom].copy(deep=True)
                y = filter_fn(y)
                b = X.loc[idx, biom].mean()
                z = (y / b) * 100
                Y.loc[idx, f"f_{biom}"] = z
        Y.dropna(inplace=True)
        Y = Y[self.feature_names]
        self.data = Y

    def fit_model(self, lag_order=3, trend='ct'):
        model = VAR(self.data[self.feature_names]).fit(maxlags=lag_order, trend='ct')
        self.model = model

    def summary(self):
        return self.model.summary()

    def get_residual_corr_matrix(self):
        if not self.model:
            raise ValueError("Model is not fitted yet.")
        return self.model.resid.corr()

    def get_exp_var_matrix(self, lags_ahead=3):
        if not self.model:
            raise ValueError("Model is not fitted yet.")
        fevd_data = self.model.fevd(lags_ahead).decomp
        fedv_matrix = fevd_data[:, -1, :]
        return fedv_matrix

    def plot_model_res_matrices(self):
        A = self.get_residual_corr_matrix()
        B = self.get_exp_var_matrix()

        labels = [self.pm.plotting_physio_labels_short[biometric]
                  for biometric in self.pm.CORE_BIOMETRICS]
        with plt.rc_context(self.pm.plotting_params):
            f, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=self.pm.PLOT_DPI)

            ax = axs[0]
            corr_sub = A
            mask = ~np.tril(A, -1).astype(bool)

            sns.heatmap(
                np.around(corr_sub, 2),
                annot=True,
                cmap='icefire',
                center=0,
                cbar_kws={'label': 'Residual Correlation'},
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                square=True,
                cbar=False,
                mask=mask,
                linewidths=1
            )

            ax.grid(False)
            ax.set_title("Residual Correlation Matrix")
            ax.set_ylabel("Biometric")
            ax.set_xlabel("Biometric")

            ax2 = axs[1]
            sns.heatmap(
                np.around(B, 2),
                annot=True,
                cmap='magma',
                cbar_kws={'label': 'FEVD (Variance Explained)'},
                xticklabels=labels,
                yticklabels=labels,
                ax=ax2,
                square=True,
                cbar=False,
                linewidths=1
            )
            ax2.grid(False)
            ax2.set_xlabel("Explaining Variable")
            ax2.set_ylabel("Explained Variable")
            ax2.set_title("Variance Explained")

            _w_spacing = 0.04
            label_size = self.pm.plotting_params.get('font.size')
            label_weight = 'bold'
            a0p = ax.get_position()
            a1p = ax2.get_position()
            f.text(a0p.x0 - _w_spacing, a0p.y1, "a.", ha='right', va='bottom',
                   fontsize=label_size, fontweight=label_weight)
            f.text(a1p.x0 - _w_spacing, a1p.y1, "b.", ha='right', va='bottom',
                   fontsize=label_size, fontweight=label_weight)

            return f, axs
