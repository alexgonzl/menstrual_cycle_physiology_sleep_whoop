"""
Data Processing Module

Handles loading, cleaning, and preprocessing of menstrual cycle and sleep data
from wearable devices (particularly Whoop data).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import warnings

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    warnings.warn("rpy2 not available. R integration features will be disabled.")


class DataProcessor:
    """
    Main class for processing menstrual cycle and sleep data.
    
    This class provides methods for loading, cleaning, and preprocessing
    data from wearable devices, with optional R integration for advanced
    statistical analysis.
    """
    
    def __init__(self, use_r: bool = False):
        """
        Initialize the DataProcessor.
        
        Parameters:
        -----------
        use_r : bool, default=False
            Whether to enable R integration for advanced statistical functions.
        """
        self.use_r = use_r and R_AVAILABLE
        self.data = None
        
        if self.use_r:
            pandas2ri.activate()
            self.r_base = importr('base')
            self.r_stats = importr('stats')
    
    def load_data(self, filepath: str, data_type: str = "csv") -> pd.DataFrame:
        """
        Load data from file.
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        data_type : str, default="csv"
            Type of data file ("csv", "excel", "json")
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        if data_type.lower() == "csv":
            self.data = pd.read_csv(filepath)
        elif data_type.lower() == "excel":
            self.data = pd.read_excel(filepath)
        elif data_type.lower() == "json":
            self.data = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
        return self.data
    
    def clean_data(self, remove_outliers: bool = True, outlier_method: str = "iqr") -> pd.DataFrame:
        """
        Clean the loaded data by handling missing values and outliers.
        
        Parameters:
        -----------
        remove_outliers : bool, default=True
            Whether to remove outliers
        outlier_method : str, default="iqr"
            Method for outlier detection ("iqr", "zscore")
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Handle missing values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
        
        # Remove outliers if requested
        if remove_outliers:
            if outlier_method == "iqr":
                self.data = self._remove_outliers_iqr(self.data)
            elif outlier_method == "zscore":
                self.data = self._remove_outliers_zscore(self.data)
        
        return self.data
    
    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= threshold]
        
        return df
    
    def apply_r_function(self, r_code: str, return_result: bool = True) -> Optional[Any]:
        """
        Execute R code with access to the current dataset.
        
        Parameters:
        -----------
        r_code : str
            R code to execute
        return_result : bool, default=True
            Whether to return the result of the R code execution
            
        Returns:
        --------
        Any or None
            Result of R code execution if return_result=True
        """
        if not self.use_r:
            raise RuntimeError("R integration not available. Initialize with use_r=True.")
        
        if self.data is not None:
            ro.globalenv['data'] = self.data
        
        result = ro.r(r_code)
        
        if return_result:
            return result
        
        return None
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset.
        
        Returns:
        --------
        Dict[str, Any]
            Summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return {
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_summary': self.data.describe().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }