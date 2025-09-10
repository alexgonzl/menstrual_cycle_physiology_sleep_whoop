"""
Cycle Analysis Module

Provides functionality for analyzing menstrual cycle patterns,
variability, and physiological changes throughout the cycle.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class CycleAnalyzer:
    """
    Analyzer for menstrual cycle patterns and physiological changes.
    
    This class provides methods for identifying cycle phases, analyzing
    cycle variability, and examining physiological changes throughout
    the menstrual cycle.
    """
    
    def __init__(self, use_r: bool = False):
        """
        Initialize the CycleAnalyzer.
        
        Parameters:
        -----------
        use_r : bool, default=False
            Whether to enable R integration for advanced statistical analysis.
        """
        self.use_r = use_r and R_AVAILABLE
        self.cycle_data = None
        
        if self.use_r:
            self.r_stats = importr('stats')
    
    def load_cycle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load cycle and physiological data for analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with date and physiological measurements
            
        Returns:
        --------
        pd.DataFrame
            Loaded cycle data
        """
        if 'date' not in data.columns:
            raise ValueError("Data must contain 'date' column.")
        
        self.cycle_data = data.copy()
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.cycle_data['date']):
            self.cycle_data['date'] = pd.to_datetime(self.cycle_data['date'])
        
        # Sort by date
        self.cycle_data = self.cycle_data.sort_values('date').reset_index(drop=True)
        
        return self.cycle_data
    
    def identify_cycle_phases(self, menstruation_dates: List[datetime], 
                            cycle_length: int = 28) -> pd.Series:
        """
        Identify menstrual cycle phases for each date.
        
        Parameters:
        -----------
        menstruation_dates : List[datetime]
            List of menstruation start dates
        cycle_length : int, default=28
            Average cycle length in days
            
        Returns:
        --------
        pd.Series
            Series with cycle phase for each date
        """
        if self.cycle_data is None:
            raise ValueError("No cycle data loaded. Call load_cycle_data() first.")
        
        phases = pd.Series(index=self.cycle_data.index, dtype='object')
        
        for i, date in enumerate(self.cycle_data['date']):
            # Find the most recent menstruation date
            recent_menses = [m for m in menstruation_dates if m <= date.to_pydatetime()]
            
            if not recent_menses:
                phases.iloc[i] = 'unknown'
                continue
            
            last_menses = max(recent_menses)
            days_since_menses = (date.to_pydatetime() - last_menses).days
            
            # Define phases based on days since menstruation
            if days_since_menses <= 5:
                phases.iloc[i] = 'menstrual'
            elif days_since_menses <= 13:
                phases.iloc[i] = 'follicular'
            elif days_since_menses <= 15:
                phases.iloc[i] = 'ovulatory'
            elif days_since_menses <= cycle_length:
                phases.iloc[i] = 'luteal'
            else:
                phases.iloc[i] = 'unknown'
        
        return phases
    
    def calculate_cycle_variability(self, menstruation_dates: List[datetime]) -> Dict[str, float]:
        """
        Calculate menstrual cycle variability metrics.
        
        Parameters:
        -----------
        menstruation_dates : List[datetime]
            List of menstruation start dates
            
        Returns:
        --------
        Dict[str, float]
            Cycle variability metrics
        """
        if len(menstruation_dates) < 2:
            raise ValueError("Need at least 2 menstruation dates to calculate variability.")
        
        # Sort dates
        sorted_dates = sorted(menstruation_dates)
        
        # Calculate cycle lengths
        cycle_lengths = []
        for i in range(1, len(sorted_dates)):
            length = (sorted_dates[i] - sorted_dates[i-1]).days
            cycle_lengths.append(length)
        
        cycle_lengths = np.array(cycle_lengths)
        
        return {
            'mean_cycle_length': float(np.mean(cycle_lengths)),
            'std_cycle_length': float(np.std(cycle_lengths)),
            'min_cycle_length': float(np.min(cycle_lengths)),
            'max_cycle_length': float(np.max(cycle_lengths)),
            'cv_cycle_length': float(np.std(cycle_lengths) / np.mean(cycle_lengths)),
            'number_of_cycles': len(cycle_lengths)
        }
    
    def analyze_physiological_changes(self, cycle_phases: pd.Series, 
                                    metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Analyze physiological changes across cycle phases.
        
        Parameters:
        -----------
        cycle_phases : pd.Series
            Cycle phase for each observation
        metrics : List[str]
            List of physiological metrics to analyze
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Physiological metrics by cycle phase
        """
        if self.cycle_data is None:
            raise ValueError("No cycle data loaded. Call load_cycle_data() first.")
        
        missing_metrics = [m for m in metrics if m not in self.cycle_data.columns]
        if missing_metrics:
            raise ValueError(f"Missing metrics in data: {missing_metrics}")
        
        # Add cycle phases to data
        analysis_data = self.cycle_data.copy()
        analysis_data['cycle_phase'] = cycle_phases
        
        results = {}
        
        for metric in metrics:
            results[metric] = {}
            
            for phase in analysis_data['cycle_phase'].unique():
                if pd.isna(phase):
                    continue
                
                phase_data = analysis_data[analysis_data['cycle_phase'] == phase][metric]
                phase_data = phase_data.dropna()
                
                if len(phase_data) > 0:
                    results[metric][phase] = {
                        'mean': float(phase_data.mean()),
                        'std': float(phase_data.std()),
                        'median': float(phase_data.median()),
                        'count': len(phase_data),
                        'min': float(phase_data.min()),
                        'max': float(phase_data.max())
                    }
        
        return results
    
    def detect_ovulation_patterns(self, temperature_data: Optional[pd.Series] = None,
                                hormone_data: Optional[pd.Series] = None) -> Dict[str, List[datetime]]:
        """
        Detect ovulation patterns using temperature or hormone data.
        
        Parameters:
        -----------
        temperature_data : pd.Series, optional
            Basal body temperature data
        hormone_data : pd.Series, optional
            Hormone level data (e.g., LH)
            
        Returns:
        --------
        Dict[str, List[datetime]]
            Detected ovulation dates by method
        """
        if self.cycle_data is None:
            raise ValueError("No cycle data loaded. Call load_cycle_data() first.")
        
        ovulation_dates = {}
        
        # Temperature-based detection
        if temperature_data is not None:
            temp_dates = self._detect_ovulation_temperature(temperature_data)
            ovulation_dates['temperature'] = temp_dates
        
        # Hormone-based detection
        if hormone_data is not None:
            hormone_dates = self._detect_ovulation_hormone(hormone_data)
            ovulation_dates['hormone'] = hormone_dates
        
        return ovulation_dates
    
    def _detect_ovulation_temperature(self, temperature_data: pd.Series) -> List[datetime]:
        """Detect ovulation using temperature shift method."""
        ovulation_dates = []
        
        # Simple temperature shift detection
        temp_series = temperature_data.dropna()
        if len(temp_series) < 10:
            return ovulation_dates
        
        # Look for sustained temperature increases
        rolling_mean = temp_series.rolling(window=3).mean()
        
        for i in range(3, len(temp_series) - 3):
            # Check if current temp is higher than previous 3 days
            if (temp_series.iloc[i] > rolling_mean.iloc[i-3:i].max() + 0.2 and
                temp_series.iloc[i+1] > rolling_mean.iloc[i-3:i].max() + 0.1 and
                temp_series.iloc[i+2] > rolling_mean.iloc[i-3:i].max() + 0.1):
                
                ovulation_date = self.cycle_data.iloc[i]['date'].to_pydatetime()
                ovulation_dates.append(ovulation_date)
        
        return ovulation_dates
    
    def _detect_ovulation_hormone(self, hormone_data: pd.Series) -> List[datetime]:
        """Detect ovulation using hormone peak detection."""
        ovulation_dates = []
        
        hormone_series = hormone_data.dropna()
        if len(hormone_series) < 5:
            return ovulation_dates
        
        # Simple peak detection
        for i in range(2, len(hormone_series) - 2):
            # Check if current value is a local maximum
            if (hormone_series.iloc[i] > hormone_series.iloc[i-1] and
                hormone_series.iloc[i] > hormone_series.iloc[i+1] and
                hormone_series.iloc[i] > hormone_series.iloc[i-2] and
                hormone_series.iloc[i] > hormone_series.iloc[i+2]):
                
                # Check if it's significantly higher than baseline
                baseline = hormone_series.median()
                if hormone_series.iloc[i] > baseline * 1.5:
                    ovulation_date = self.cycle_data.iloc[i]['date'].to_pydatetime()
                    ovulation_dates.append(ovulation_date)
        
        return ovulation_dates
    
    def correlate_with_external_factors(self, cycle_phases: pd.Series,
                                      external_factors: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Correlate cycle phases with external factors.
        
        Parameters:
        -----------
        cycle_phases : pd.Series
            Cycle phase for each observation
        external_factors : Dict[str, pd.Series]
            External factors to correlate with (e.g., stress, exercise)
            
        Returns:
        --------
        Dict[str, float]
            Correlation coefficients
        """
        if self.cycle_data is None:
            raise ValueError("No cycle data loaded. Call load_cycle_data() first.")
        
        # Create numeric encoding for cycle phases
        phase_mapping = {'menstrual': 1, 'follicular': 2, 'ovulatory': 3, 'luteal': 4}
        numeric_phases = cycle_phases.map(phase_mapping)
        
        correlations = {}
        
        for factor_name, factor_data in external_factors.items():
            if len(factor_data) == len(numeric_phases):
                # Remove NaN values
                valid_indices = ~(numeric_phases.isna() | factor_data.isna())
                
                if valid_indices.sum() > 10:  # Need sufficient data points
                    corr = np.corrcoef(numeric_phases[valid_indices], 
                                     factor_data[valid_indices])[0, 1]
                    correlations[factor_name] = float(corr)
        
        return correlations
    
    def run_r_analysis(self, r_script: str) -> Dict[str, any]:
        """
        Run custom R analysis on cycle data.
        
        Parameters:
        -----------
        r_script : str
            R script to execute
            
        Returns:
        --------
        Dict[str, any]
            Results from R analysis
        """
        if not self.use_r:
            raise RuntimeError("R integration not available. Initialize with use_r=True.")
        
        if self.cycle_data is not None:
            ro.globalenv['cycle_data'] = self.cycle_data
        
        try:
            result = ro.r(r_script)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}