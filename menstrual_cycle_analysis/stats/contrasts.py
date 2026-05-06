"""`StatisticalPredictionHandler` — verbatim port of the slice used by
notebooks 01–02 (`get_conditional_predictions`, `calculate_conditional_contrast`,
`calculate_within_subject_contrast`, `calculate_min_term_ci` + their helpers).

Source: `whoop_analyses/whoop_analyses/statistical_prediction_methods.py`
(StatisticalPredictionHandler).

Methods unused by these notebooks (marginal predictions / contrasts, raw /
subject-averaged contrasts, interaction contrasts, simple-interaction effects)
are intentionally not included; they will be ported when later notebooks need
them. Method bodies that are kept are byte-identical to the source.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats


class StatisticalPredictionHandler:
    """Statistical prediction and contrast methods for fitted models."""

    def __init__(self, model, data: pd.DataFrame):
        self.model = model
        self.data = data.copy()
        self.is_logistic = self._detect_logistic_model()
        self.model_params = self._extract_model_params()
        self.model_variables = self._get_model_variables()

    # =========================================================================
    # CORE MODEL UTILITIES
    # =========================================================================

    def _detect_logistic_model(self) -> bool:
        family = getattr(self.model.model, 'family', None)
        if family:
            return family.__class__.__name__.lower() == 'binomial' and family.link.__class__.__name__.lower() == 'logit'
        return False

    def _extract_model_params(self) -> Dict[str, Any]:
        params = {'success': True}
        model = getattr(self.model, 'model', None)

        if model:
            params['formula'] = getattr(model, 'formula', None)
            params['family'] = getattr(model, 'family', None)
            params['cov_struct'] = getattr(model, 'cov_struct', None)
            params['model_type'] = 'GEE' if hasattr(model, 'cov_struct') else 'GLM' if params['family'] else 'OLS'
            params['groups_var'] = self._find_column(['n_id', 'user_id', 'subject_id', 'id'])
            params['weights_var'] = self._find_column(['weights', 'weight', 'wt'])

            params['exog_names'] = model.formula.split('~')[0].strip(' ')

        return params

    def _find_column(self, candidates: List[str]) -> str:
        for col in candidates:
            if col in self.data.columns:
                return col
        return None

    def _get_predictions(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(data)

    def _update_derived_terms(self, data: pd.DataFrame, changed_var: str) -> pd.DataFrame:
        # Update polynomial terms
        for power in range(2, 6):
            poly_col = f"{changed_var}{power}"
            if poly_col in data.columns:
                data[poly_col] = data[changed_var] ** power
        # Update interaction terms (works for any order of terms, e.g., a:b or b:a)
        for col in data.columns:
            if ':' in col and changed_var in col.split(':'):
                vars_in_interaction = col.split(':')
                # Only update if all variables in the interaction are present in data
                if all(v in data.columns for v in vars_in_interaction):
                    prod = 1
                    for v in vars_in_interaction:
                        prod = prod * data[v]
                    data[col] = prod
        return data

    def _create_base_frame(self, data: pd.DataFrame = None) -> pd.DataFrame:
        if data is None:
            data = self.data

        model_vars = self._get_model_variables()
        base_data = {}

        individual_id_var = self.model_params.get('groups_var', 'n_id')
        for col in model_vars:
            if col in data.columns:
                if data[col].dtype.kind in 'fi':
                    subject_means = data.groupby(individual_id_var)[col].mean()
                    base_data[col] = subject_means.mean()
                else:
                    base_data[col] = data[col].mode()[0]
            else:
                if '[' in col or ':' in col or 'T.' in col:
                    base_var = col.split('[')[0].split(':')[0].split('T.')[0]
                    if base_var in data.columns:
                        base_data[base_var] = data[base_var].mode()[0]
                    else:
                        raise KeyError(f"Base variable '{base_var}' for '{col}' not found.")
                else:
                    raise KeyError(f"Variable '{col}' not found in the dataset.")

        return pd.DataFrame(base_data, index=[0])

    def _get_model_variables(self) -> set:
        model_vars = set()

        # Method 1: From model parameters (coefficient names)
        if hasattr(self.model, 'params'):
            param_names = list(self.model.params.index)
            for param in param_names:
                if param != 'Intercept':
                    if ':' in param:
                        model_vars.update(param.split(':'))
                    elif param.endswith('2'):
                        base_var = param[:-1]
                        if base_var in self.data.columns:
                            model_vars.add(base_var)
                        model_vars.add(param)
                    elif '[' in param or 'T.' in param:
                        base_var = param.split('[')[0].split('T.')[0]
                        if base_var in self.data.columns:
                            model_vars.add(base_var)
                    else:
                        model_vars.add(param)

        # Method 2: From formula if available
        if self.model_params.get('formula'):
            try:
                formula_desc = patsy.ModelDesc.from_formula(self.model_params['formula'])
                for term in formula_desc.rhs_termlist:
                    for factor in term.factors:
                        var_name = factor.name()
                        if var_name in self.data.columns:
                            model_vars.add(var_name)
            except Exception:
                pass

        # Filter to only include variables that exist in the data
        model_vars = {var for var in model_vars if var in self.data.columns}

        return model_vars

    # =========================================================================
    # PREDICTION METHODS
    # =========================================================================

    def get_conditional_predictions(self, eval_term, eval_vals, alpha: float = 0.05, fixed_values: dict = None) -> pd.DataFrame:
        base_frame = self._create_base_frame()

        if isinstance(eval_term, str):
            eval_term_list = [eval_term]
            if isinstance(eval_vals, (np.ndarray, pd.Series)):
                eval_vals_list = [[val] for val in eval_vals.tolist()]
            else:
                eval_vals_list = [[val] for val in list(eval_vals)]
        else:
            eval_term_list = list(eval_term)
            if isinstance(eval_vals, (np.ndarray, pd.Series)):
                eval_vals_list = [list(val) for val in eval_vals.tolist()]
            else:
                eval_vals_list = [list(val) for val in list(eval_vals)]

        frames = []
        for vals in eval_vals_list:
            frame = base_frame.copy()
            for col, val in zip(eval_term_list, vals):
                frame[col] = val
                frame = self._update_derived_terms(frame, col)
            frames.append(frame)

        combined_frame = pd.concat(frames, ignore_index=True)
        if fixed_values is not None:
            for key, value in fixed_values.items():
                combined_frame[key] = value

        pred_results = self.model.get_prediction(combined_frame).summary_frame(alpha=alpha)

        result_dict = {col: [vals[i] for vals in eval_vals_list] for i, col in enumerate(eval_term_list)}
        result_dict['pred'] = pred_results['mean']
        result_dict['pred_ci_lower'] = pred_results['mean_ci_lower']
        result_dict['pred_ci_upper'] = pred_results['mean_ci_upper']

        return pd.DataFrame(result_dict)

    # =========================================================================
    # CONTRAST METHODS
    # =========================================================================

    def calculate_conditional_contrast(
        self,
        term_of_interest: str,
        values_to_compare: List[float],
        alpha: float = 0.05,
        fixed_values: dict = None
    ) -> pd.DataFrame:
        from itertools import combinations

        base_frame = self._create_base_frame()

        if fixed_values is not None:
            for k, v in fixed_values.items():
                base_frame[k] = v

        frames = [
            self._update_derived_terms(base_frame.assign(**{term_of_interest: val}), term_of_interest)
            for val in values_to_compare
        ]
        combined_frame = pd.concat(frames, ignore_index=True)

        design_info = self.model.model.data.design_info
        X_list = patsy.build_design_matrices([design_info], combined_frame)
        X_all = X_list[0]

        pred_obj = self.model.get_prediction(combined_frame)
        pred_results = pred_obj.summary_frame(alpha=alpha)

        results = []
        for (i, v1), (j, v2) in combinations(enumerate(values_to_compare), 2):
            L = X_all[j, :] - X_all[i, :]
            contrast_estimate = np.dot(L, self.model.params)
            contrast_variance = np.dot(L, np.dot(self.model.cov_params(), L))
            contrast_se_linear = np.sqrt(contrast_variance)

            z_critical = stats.norm.ppf(1 - alpha / 2)
            ci_lower_linear = contrast_estimate - z_critical * contrast_se_linear
            ci_upper_linear = contrast_estimate + z_critical * contrast_se_linear

            pred1_response = pred_results.iloc[i]['mean']
            pred2_response = pred_results.iloc[j]['mean']
            contrast_response = pred2_response - pred1_response

            res = {
                'term': term_of_interest,
                'value1': v1,
                'value2': v2,
                'mean1': pred1_response,
                'mean2': pred2_response,
                'contrast': contrast_response,
                'contrast_linear': contrast_estimate,
                'contrast_se': contrast_se_linear,
                'contrast_ci_lower': ci_lower_linear,
                'contrast_ci_upper': ci_upper_linear,
            }

            if self.is_logistic:
                se1_linear = np.sqrt(np.dot(X_all[i, :], np.dot(self.model.cov_params(), X_all[i, :])))
                se2_linear = np.sqrt(np.dot(X_all[j, :], np.dot(self.model.cov_params(), X_all[j, :])))

                se1_response = se1_linear * pred1_response * (1 - pred1_response)
                se2_response = se2_linear * pred2_response * (1 - pred2_response)

                or_results = self._calculate_odds_ratio_and_ci(pred1_response, pred2_response, se1_response, se2_response, alpha)
                res.update(or_results)

            results.append(res)

        return pd.DataFrame(results)

    def _calculate_odds_ratio_and_ci(self, pred1: float, pred2: float, se1: float, se2: float,
                                     alpha: float = 0.05) -> Dict[str, float]:
        epsilon = 1e-6
        pred1, pred2 = max(min(pred1, 1 - epsilon), epsilon), max(min(pred2, 1 - epsilon), epsilon)

        odds1, odds2 = pred1 / (1 - pred1), pred2 / (1 - pred2)
        odds_ratio = odds2 / odds1
        log_or = np.log(odds_ratio)

        se_log_or = np.sqrt((se1**2) / (pred1**2 * (1 - pred1)**2) + (se2**2) / (pred2**2 * (1 - pred2)**2))
        z_critical = stats.norm.ppf(1 - alpha / 2)
        log_ci_lower, log_ci_upper = log_or - z_critical * se_log_or, log_or + z_critical * se_log_or

        return {
            'odds_ratio': odds_ratio,
            'ci_l_or': np.exp(log_ci_lower),
            'ci_u_or': np.exp(log_ci_upper)
        }

    # =========================================================================
    # MINIMUM-POINT CONFIDENCE INTERVALS
    # =========================================================================

    def calculate_min_term_ci(self, term, alpha=0.05, interaction_terms=None,
                              interaction_values=None, n_bootstrap=1000, method='delta'):
        if interaction_terms is not None and isinstance(interaction_terms, str):
            interaction_terms = [interaction_terms]

        model_params = self._prepare_quadratic_terms(term, interaction_terms)

        all_results = []

        values_to_process = self._expand_interaction_values(model_params['interaction_names'], interaction_values)

        for interaction_dict in values_to_process:
            min_x = self._calculate_minimum_point(
                model_params['beta_term'],
                model_params['beta_term2'],
                model_params['interaction_coeffs'],
                interaction_dict
            )

            if method == 'bootstrap':
                ci_lower, ci_upper = self._bootstrap_confidence_intervals(
                    term, model_params, interaction_dict, n_bootstrap, alpha
                )
            else:
                ci_lower, ci_upper = self._delta_confidence_intervals(
                    term, model_params, interaction_dict, alpha
                )

            result = {'min_x': min_x, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
            result.update(interaction_dict)
            all_results.append(result)

        return pd.DataFrame(all_results)

    def _expand_interaction_values(self, interaction_names, interaction_values=None):
        if not interaction_names:
            return [{}]

        if interaction_values is None:
            interaction_values = {}

        processed_values = {}
        for interaction_term in interaction_names:
            if interaction_term in interaction_values:
                values = interaction_values[interaction_term]
                if not isinstance(values, (list, tuple, np.ndarray)):
                    values = [values]
                processed_values[interaction_term] = values
            elif interaction_term in self.data.columns:
                processed_values[interaction_term] = [self.data[interaction_term].mean()]
            else:
                raise ValueError(f"Interaction term '{interaction_term}' not found in data and no value provided")

        keys = processed_values.keys()
        values = list(processed_values.values())

        if not values:
            return [{}]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _prepare_quadratic_terms(self, term, interaction_terms=None):
        model_results = self.model
        term2 = f"{term}2"

        if term2 not in model_results.params.index:
            raise ValueError(f"No quadratic term '{term2}' found in model parameters")

        beta_term = model_results.params[term]
        beta_term2 = model_results.params[term2]

        interaction_coeffs = {}
        interaction_names = {}

        if interaction_terms:
            if not isinstance(interaction_terms, list):
                interaction_terms = [interaction_terms]

            for interaction_term in interaction_terms:
                interaction_name = f"{interaction_term}:{term}"
                if interaction_name in model_results.params.index:
                    interaction_coeffs[interaction_term] = model_results.params[interaction_name]
                    interaction_names[interaction_term] = interaction_name
                else:
                    interaction_name = f"{term}:{interaction_term}"
                    if interaction_name in model_results.params.index:
                        interaction_coeffs[interaction_term] = model_results.params[interaction_name]
                        interaction_names[interaction_term] = interaction_name
                    else:
                        raise ValueError(f"No interaction between {term} and {interaction_term} found in model")
        else:
            for param in model_results.params.index:
                if ':' in param:
                    parts = param.split(':')
                    if term in parts:
                        other_term = parts[0] if parts[0] != term else parts[1]
                        interaction_coeffs[other_term] = model_results.params[param]
                        interaction_names[other_term] = param

        return {
            'beta_term': beta_term,
            'beta_term2': beta_term2,
            'term': term,
            'term2': term2,
            'interaction_coeffs': interaction_coeffs,
            'interaction_names': interaction_names
        }

    def _calculate_minimum_point(self, beta_term, beta_term2, interaction_coeffs, interaction_values):
        adjusted_beta = beta_term
        for term_name, coef in interaction_coeffs.items():
            if term_name in interaction_values:
                value = interaction_values[term_name]
                if isinstance(value, (list, tuple, np.ndarray)):
                    raise TypeError(f"Expected a single value for interaction term '{term_name}', got {value}")
                adjusted_beta += coef * value

        return -adjusted_beta / (2 * beta_term2)

    def _bootstrap_confidence_intervals(self, term, model_params, interaction_values, n_bootstrap, alpha):
        beta_term = model_params['beta_term']
        beta_term2 = model_params['beta_term2']
        term2 = model_params['term2']
        interaction_coeffs = model_params['interaction_coeffs']
        interaction_names = model_params['interaction_names']

        param_names = [term, term2] + list(interaction_names.values())

        model_results = self.model
        param_means = model_results.params[param_names].values
        param_cov = model_results.cov_params().loc[param_names, param_names].values

        bootstrap_mins = []

        try:
            bootstrap_params = np.random.multivariate_normal(
                mean=param_means, cov=param_cov, size=n_bootstrap
            )

            for i in range(n_bootstrap):
                b = bootstrap_params[i, 0]
                b2 = bootstrap_params[i, 1]

                if b2 <= 0:
                    continue

                adj_b = b
                for j, interaction_term in enumerate(interaction_names):
                    int_coef = bootstrap_params[i, j + 2]
                    adj_b += int_coef * interaction_values[interaction_term]

                bootstrap_mins.append(-adj_b / (2 * b2))

            if len(bootstrap_mins) < n_bootstrap * 0.1:
                print(f"Warning: Only {len(bootstrap_mins)}/{n_bootstrap} valid bootstrap samples")

            ci_lower = np.percentile(bootstrap_mins, (alpha / 2) * 100)
            ci_upper = np.percentile(bootstrap_mins, (1 - alpha / 2) * 100)

        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Bootstrap error: {e}")
            ci_lower, ci_upper = self._delta_confidence_intervals(
                term, model_params, interaction_values, alpha
            )

        return ci_lower, ci_upper

    def _delta_confidence_intervals(self, term, model_params, interaction_values, alpha):
        beta_term = model_params['beta_term']
        beta_term2 = model_params['beta_term2']
        term2 = model_params['term2']
        interaction_coeffs = model_params['interaction_coeffs']
        interaction_names = model_params['interaction_names']

        min_x = self._calculate_minimum_point(beta_term, beta_term2, interaction_coeffs, interaction_values)

        adjusted_beta = beta_term
        for term_name, coef in interaction_coeffs.items():
            if term_name in interaction_values:
                adjusted_beta += coef * interaction_values[term_name]

        param_names = [term, term2] + list(interaction_names.values())
        model_results = self.model
        cov_matrix = model_results.cov_params().loc[param_names, param_names].values

        gradient = np.zeros(len(param_names))
        gradient[0] = -1 / (2 * beta_term2)
        gradient[1] = adjusted_beta / (2 * beta_term2**2)

        for i, interaction_term in enumerate(interaction_names):
            gradient[i + 2] = -interaction_values[interaction_term] / (2 * beta_term2)

        variance = gradient.T @ cov_matrix @ gradient

        std_error = np.sqrt(variance)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = min_x - z * std_error
        ci_upper = min_x + z * std_error

        return ci_lower, ci_upper

    # =========================================================================
    # WITHIN-SUBJECT CONTRASTS
    # =========================================================================

    def _find_individuals_with_both_values(self, bin_var: str, values_to_compare: List[float],
                                           individual_id_var: str, data=None) -> List[Any]:
        """Find individuals with observations in both bins of the binned variable."""
        value1, value2 = values_to_compare
        if data is None:
            data = self.data
        individuals_value1 = set(data[data[bin_var] == value1][individual_id_var])
        individuals_value2 = set(data[data[bin_var] == value2][individual_id_var])
        return list(individuals_value1.intersection(individuals_value2))

    def _fit_within_subject_model(self, data: pd.DataFrame, individual_id_var: str, weights_var: str):
        """Fit a GEE model for within-subject analysis, incorporating weights."""
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        formula = self.model_params['formula']
        family = self.model_params.get('family', sm.families.Gaussian())

        weights = data[weights_var] if weights_var in data.columns else None

        model = smf.gee(
            formula=formula,
            groups=data[individual_id_var],
            data=data,
            family=family,
            cov_struct=sm.cov_struct.Exchangeable(),
            weights=weights
        ).fit(maxiter=100)

        return model

    def _preprocess_data(self, data: pd.DataFrame, term_of_interest: str, bin_var: str,
                         individual_id_var: str, weights_var: str) -> pd.DataFrame:
        required_columns = [term_of_interest, bin_var, individual_id_var, weights_var]
        data = self._validate_data(data, required_columns=required_columns)
        return data

    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        data = data.dropna(subset=required_columns)
        for col in required_columns:
            if col in data.columns:
                data = data[~data[col].isin([np.inf, -np.inf])]
        return data

    def calculate_within_subject_contrast(self, term_of_interest: str, values_to_compare: List[float],
                                          bin_var: str, individual_id_var: str = 'n_id',
                                          weights_var: str = 'weights', method: str = 'model',
                                          bootstrap_samples: int = 500, alpha: float = 0.05) -> pd.DataFrame:
        value1, value2 = values_to_compare

        # Step 1: Filter individuals with observations in both bins
        individuals_with_both = self._find_individuals_with_both_values(bin_var, values_to_compare, individual_id_var)
        if len(individuals_with_both) == 0:
            raise ValueError("No individuals found with observations in both bins.")

        within_subject_data = self.data[self.data[individual_id_var].isin(individuals_with_both)].copy()

        # Step 2: Recompute weights based on the filtered data
        within_subject_data[weights_var] = within_subject_data.groupby(bin_var)[bin_var].transform(
            lambda x: 1 / np.log(len(x)) if len(x) > 0 else 0)

        # Step 3: Drop rows with missing values in relevant columns
        required_columns = list(set([term_of_interest, bin_var, individual_id_var, weights_var] + list(self._get_model_variables())))
        within_subject_data = within_subject_data.dropna(subset=required_columns)

        # Step 4: Perform analysis based on the selected method
        if method == 'model':
            model = self._fit_within_subject_model(within_subject_data, individual_id_var, weights_var)
            contrast_results = self._calculate_model_based_contrast(model, term_of_interest, values_to_compare, alpha)
        elif method == 'bootstrap':
            contrast_results = self._bootstrap_within_subjects_marginal_odds_ratio(
                term_of_interest=term_of_interest,
                values_to_compare=values_to_compare,
                bin_var=bin_var,
                individual_id_var=individual_id_var,
                weights_var=weights_var,
                n_bootstrap=bootstrap_samples,
                alpha=alpha
            )
            contrast_results = pd.DataFrame([contrast_results])
        else:
            raise ValueError(f"Unknown method: {method}. Use 'model' or 'bootstrap'.")

        # Step 5: Add metadata and return results
        contrast_results['method'] = method
        contrast_results['n_individuals'] = len(individuals_with_both)
        contrast_results['n_observations'] = len(within_subject_data)
        obs_per_value = within_subject_data.groupby(bin_var).size().to_dict()
        for val in values_to_compare:
            contrast_results[f'n_obs_{val}'] = obs_per_value.get(val, 0)

        return contrast_results

    def _calculate_model_based_contrast(self, model, term_of_interest: str, values_to_compare: List[float],
                                        alpha: float = 0.05) -> pd.DataFrame:
        """Calculate contrasts using a fitted within-subject model."""
        value1, value2 = values_to_compare

        base_frame = self._create_base_frame()
        pred_data = pd.concat([base_frame, base_frame], ignore_index=True)
        pred_data[term_of_interest] = values_to_compare
        pred_data = self._update_derived_terms(pred_data, term_of_interest)

        pred_results = model.get_prediction(pred_data).summary_frame(alpha=alpha)
        pred1, pred2 = pred_results.iloc[0]['mean'], pred_results.iloc[1]['mean']
        se1, se2 = pred_results.iloc[0]['mean_se'], pred_results.iloc[1]['mean_se']

        contrast = pred2 - pred1
        contrast_se = np.sqrt(se1**2 + se2**2)
        z_critical = stats.norm.ppf(1 - alpha / 2)

        results = {
            'term': term_of_interest,
            'value1': value1,
            'value2': value2,
            'mean1': pred1,
            'mean2': pred2,
            'contrast': contrast,
            'contrast_se': contrast_se,
            'contrast_ci_lower': contrast - z_critical * contrast_se,
            'contrast_ci_upper': contrast + z_critical * contrast_se,
        }

        if self.is_logistic:
            or_results = self._calculate_odds_ratio_and_ci(pred1, pred2, se1, se2, alpha)
            results.update(or_results)

        return pd.DataFrame(results, index=[0])

    def _bootstrap_within_subjects_marginal_odds_ratio(self, term_of_interest: str, values_to_compare,
                                                       bin_var: str, individual_id_var: str, weights_var: str,
                                                       n_bootstrap: int = 1000, alpha: float = 0.05) -> Dict[str, float]:
        """Bootstrap the odds ratio and confidence intervals by resampling by subject."""
        bootstrap_or = []
        value1, value2 = values_to_compare
        epsilon = 1e-6

        individuals_with_both = self._find_individuals_with_both_values(bin_var, values_to_compare, individual_id_var)
        if len(individuals_with_both) == 0:
            raise ValueError("No individuals found with observations in both bins.")

        within_subject_data = self.data[self.data[individual_id_var].isin(individuals_with_both)].copy()

        within_subject_data[weights_var] = within_subject_data.groupby(bin_var)[bin_var].transform(
            lambda x: 1 / np.log(len(x)) if len(x) > 0 else 0)

        required_columns = list(set([term_of_interest, bin_var, individual_id_var, weights_var] + list(self._get_model_variables())))
        within_subject_data = within_subject_data.dropna(subset=required_columns)

        grouped_data = within_subject_data.groupby(individual_id_var)

        for i in range(n_bootstrap):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)

                    resampled_subjects = np.random.choice(list(grouped_data.groups.keys()), size=len(grouped_data), replace=True)
                    boot_data = pd.concat([grouped_data.get_group(subject) for subject in resampled_subjects])
                    boot_data = self._preprocess_data(boot_data, term_of_interest, bin_var, individual_id_var, weights_var)

                    individuals_with_both = self._find_individuals_with_both_values(bin_var, values_to_compare, individual_id_var, data=boot_data)
                    if len(individuals_with_both) == 0:
                        print(f"Skipping bootstrap iteration {i} due to insufficient data.")
                        continue
                    boot_data = boot_data[boot_data[individual_id_var].isin(individuals_with_both)]

                    boot_data[weights_var] = boot_data.groupby(bin_var)[bin_var].transform(
                        lambda x: 1 / np.log(len(x)) if len(x) > 0 else 0
                    )

                    model = self._fit_within_subject_model(boot_data, individual_id_var, weights_var)

                    pred_data1 = boot_data.copy(deep=True)
                    pred_data1[term_of_interest] = value1
                    pred_data1 = self._update_derived_terms(pred_data1, term_of_interest)
                    pred_data2 = boot_data.copy(deep=True)
                    pred_data2[term_of_interest] = value2
                    pred_data2 = self._update_derived_terms(pred_data2, term_of_interest)

                    pred1 = model.predict(pred_data1.iloc[0:1]).iloc[0]
                    pred2 = model.predict(pred_data2.iloc[1:2]).iloc[0]

                    pred1 = np.clip(pred1, epsilon, 1 - epsilon)
                    pred2 = np.clip(pred2, epsilon, 1 - epsilon)

                    odds1, odds2 = pred1 / (1 - pred1), pred2 / (1 - pred2)
                    bootstrap_or.append(odds2 / odds1)

            except Exception as e:
                print(f"Bootstrap iteration {i} failed: {e}")
                continue

        bootstrap_or = pd.Series(bootstrap_or).dropna()

        ci_lower = np.percentile(bootstrap_or, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_or, (1 - alpha / 2) * 100)

        return {
            'odds_ratio': np.mean(bootstrap_or),
            'ci_l_or': ci_lower,
            'ci_u_or': ci_upper
        }
