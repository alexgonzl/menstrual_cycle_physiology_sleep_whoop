"""
Sleep Analysis Module

Provides functionality for analyzing sleep patterns and metrics
in relation to menstrual cycle phases.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class SleepAnalyzer:
    """
    Analyzer for sleep patterns and metrics during menstrual cycles.
    
    This class provides methods for analyzing sleep duration, quality,
    and patterns across different phases of the menstrual cycle.
    """
    
    def __init__(self, use_r: bool = False):
        """
        Initialize the SleepAnalyzer.
        
        Parameters:
        -----------
        use_r : bool, default=False
            Whether to enable R integration for advanced statistical analysis.
        """
        self.use_r = use_r and R_AVAILABLE
        self.sleep_data = None
        
        if self.use_r:
            self.r_stats = importr('stats')
    
    def load_sleep_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load sleep data for analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Sleep data with required columns
            
        Returns:
        --------
        pd.DataFrame
            Loaded sleep data
        """
        required_columns = ['date', 'sleep_duration', 'sleep_efficiency']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            warnings.warn(f"Missing columns: {missing_columns}. Some analyses may not work.")
        
        self.sleep_data = data.copy()
        return self.sleep_data
    
    def calculate_sleep_metrics(self) -> Dict[str, float]:
        """
        Calculate basic sleep metrics.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing sleep metrics
        """
        if self.sleep_data is None:
            raise ValueError("No sleep data loaded. Call load_sleep_data() first.")
        
        metrics = {}
        
        if 'sleep_duration' in self.sleep_data.columns:
            metrics['mean_sleep_duration'] = self.sleep_data['sleep_duration'].mean()
            metrics['std_sleep_duration'] = self.sleep_data['sleep_duration'].std()
            metrics['min_sleep_duration'] = self.sleep_data['sleep_duration'].min()
            metrics['max_sleep_duration'] = self.sleep_data['sleep_duration'].max()
        
        if 'sleep_efficiency' in self.sleep_data.columns:
            metrics['mean_sleep_efficiency'] = self.sleep_data['sleep_efficiency'].mean()
            metrics['std_sleep_efficiency'] = self.sleep_data['sleep_efficiency'].std()
        
        if 'rem_sleep' in self.sleep_data.columns:
            metrics['mean_rem_sleep'] = self.sleep_data['rem_sleep'].mean()
            metrics['std_rem_sleep'] = self.sleep_data['rem_sleep'].std()
        
        if 'deep_sleep' in self.sleep_data.columns:
            metrics['mean_deep_sleep'] = self.sleep_data['deep_sleep'].mean()
            metrics['std_deep_sleep'] = self.sleep_data['deep_sleep'].std()
        
        return metrics
    
    def analyze_sleep_by_cycle_phase(self, cycle_phases: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Analyze sleep metrics by menstrual cycle phase.
        
        Parameters:
        -----------
        cycle_phases : pd.Series
            Series indicating cycle phase for each date
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Sleep metrics grouped by cycle phase
        """
        if self.sleep_data is None:
            raise ValueError("No sleep data loaded. Call load_sleep_data() first.")
        
        if len(cycle_phases) != len(self.sleep_data):
            raise ValueError("Length of cycle_phases must match sleep data length.")
        
        # Add cycle phases to sleep data
        analysis_data = self.sleep_data.copy()
        analysis_data['cycle_phase'] = cycle_phases
        
        phase_metrics = {}
        sleep_columns = ['sleep_duration', 'sleep_efficiency', 'rem_sleep', 'deep_sleep']
        available_columns = [col for col in sleep_columns if col in analysis_data.columns]
        
        for phase in analysis_data['cycle_phase'].unique():
            if pd.isna(phase):
                continue
            
            phase_data = analysis_data[analysis_data['cycle_phase'] == phase]
            phase_metrics[phase] = {}
            
            for col in available_columns:
                phase_metrics[phase][f'mean_{col}'] = phase_data[col].mean()
                phase_metrics[phase][f'std_{col}'] = phase_data[col].std()
                phase_metrics[phase][f'median_{col}'] = phase_data[col].median()
        
        return phase_metrics
    
    def detect_sleep_anomalies(self, method: str = "zscore", threshold: float = 2.5) -> pd.DataFrame:
        """
        Detect anomalies in sleep patterns.
        
        Parameters:
        -----------
        method : str, default="zscore"
            Method for anomaly detection ("zscore", "iqr")
        threshold : float, default=2.5
            Threshold for anomaly detection
            
        Returns:
        --------
        pd.DataFrame
            Data with anomaly flags
        """
        if self.sleep_data is None:
            raise ValueError("No sleep data loaded. Call load_sleep_data() first.")
        
        result = self.sleep_data.copy()
        sleep_columns = ['sleep_duration', 'sleep_efficiency', 'rem_sleep', 'deep_sleep']
        available_columns = [col for col in sleep_columns if col in result.columns]
        
        for col in available_columns:
            if method == "zscore":
                z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
                result[f'{col}_anomaly'] = z_scores > threshold
            elif method == "iqr":
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                result[f'{col}_anomaly'] = (result[col] < lower_bound) | (result[col] > upper_bound)
        
        return result
    
    def compare_phases_statistical(self, cycle_phases: pd.Series, 
                                 sleep_metric: str = 'sleep_duration') -> Dict[str, float]:
        """
        Perform statistical comparison of sleep metrics across cycle phases.
        
        Parameters:
        -----------
        cycle_phases : pd.Series
            Series indicating cycle phase for each date
        sleep_metric : str, default='sleep_duration'
            Sleep metric to compare
            
        Returns:
        --------
        Dict[str, float]
            Statistical test results
        """
        if self.sleep_data is None:
            raise ValueError("No sleep data loaded. Call load_sleep_data() first.")
        
        if sleep_metric not in self.sleep_data.columns:
            raise ValueError(f"Sleep metric '{sleep_metric}' not found in data.")
        
        # Prepare data
        analysis_data = self.sleep_data.copy()
        analysis_data['cycle_phase'] = cycle_phases
        
        # Get unique phases
        phases = analysis_data['cycle_phase'].dropna().unique()
        
        if len(phases) < 2:
            raise ValueError("Need at least 2 cycle phases for comparison.")
        
        # If R is available, use ANOVA
        if self.use_r:
            try:
                # Convert to R format
                ro.globalenv['sleep_metric'] = analysis_data[sleep_metric].values
                ro.globalenv['cycle_phase'] = analysis_data['cycle_phase'].values
                
                # Perform ANOVA
                r_code = """
                data_df <- data.frame(metric = sleep_metric, phase = as.factor(cycle_phase))
                data_df <- data_df[complete.cases(data_df), ]
                anova_result <- aov(metric ~ phase, data = data_df)
                summary_result <- summary(anova_result)
                list(
                    f_statistic = summary_result[[1]][["F value"]][1],
                    p_value = summary_result[[1]][["Pr(>F)"]][1]
                )
                """
                result = ro.r(r_code)
                return {
                    'test': 'ANOVA',
                    'f_statistic': float(result[0][0]),
                    'p_value': float(result[1][0])
                }
            except Exception as e:
                warnings.warn(f"R analysis failed: {e}. Using Python alternative.")
        
        # Python alternative using scipy if available
        try:
            from scipy import stats
            
            # Group data by phases
            phase_groups = []
            for phase in phases:
                phase_data = analysis_data[analysis_data['cycle_phase'] == phase][sleep_metric].dropna()
                if len(phase_data) > 0:
                    phase_groups.append(phase_data.values)
            
            if len(phase_groups) >= 2:
                f_stat, p_val = stats.f_oneway(*phase_groups)
                return {
                    'test': 'ANOVA (scipy)',
                    'f_statistic': float(f_stat),
                    'p_value': float(p_val)
                }
        except ImportError:
            pass
        
        # Basic comparison using means
        phase_means = analysis_data.groupby('cycle_phase')[sleep_metric].mean()
        return {
            'test': 'descriptive_comparison',
            'phase_means': phase_means.to_dict(),
            'overall_mean': analysis_data[sleep_metric].mean(),
            'overall_std': analysis_data[sleep_metric].std()
        }