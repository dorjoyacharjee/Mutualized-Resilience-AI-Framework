"""
Helper Utilities Module
Common functions for data handling, coordinate transforms, logging

Author: Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import json
from datetime import datetime
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data integrity and quality"""

    @staticmethod
    def check_missing_values(data: Union[pd.DataFrame, np.ndarray], threshold: float = 0.1) -> bool:
        """
        Check if missing values exceed threshold

        Args:
            data: DataFrame or array
            threshold: Maximum acceptable missing fraction (0-1)

        Returns:
            True if data passes validation
        """
        if isinstance(data, pd.DataFrame):
            missing_fraction = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        else:
            missing_fraction = np.isnan(data).sum() / data.size

        if missing_fraction > threshold:
            logger.warning(f"Missing values {missing_fraction:.2%} exceed threshold {threshold:.2%}")
            return False

        return True

    @staticmethod
    def check_data_range(data: np.ndarray, expected_min: float, expected_max: float) -> bool:
        """
        Check if data falls within expected range

        Args:
            data: Input array
            expected_min: Minimum expected value
            expected_max: Maximum expected value

        Returns:
            True if data within range
        """
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)

        if data_min < expected_min or data_max > expected_max:
            logger.warning(f"Data range [{data_min}, {data_max}] outside expected [{expected_min}, {expected_max}]")
            return False

        return True

    @staticmethod
    def detect_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """
        Detect outliers in data

        Args:
            data: Input array (1D)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold

        Returns:
            Boolean mask of outliers
        """
        data_clean = data[~np.isnan(data)]

        if method == 'iqr':
            q1 = np.percentile(data_clean, 25)
            q3 = np.percentile(data_clean, 75)
            iqr = q3 - q1

            outliers = (data < q1 - threshold * iqr) | (data > q3 + threshold * iqr)

        elif method == 'zscore':
            z_scores = np.abs((data - np.nanmean(data)) / (np.nanstd(data) + 1e-10))
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        logger.info(f"Detected {outliers.sum()} outliers using {method} method")

        return outliers


class CoordinateTransforms:
    """Coordinate transformation utilities"""

    @staticmethod
    def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> Dict:
        """
        Convert bounding box to GeoJSON polygon

        Args:
            bbox: [S, W, N, E] format

        Returns:
            GeoJSON polygon dict
        """
        S, W, N, E = bbox

        polygon = {
            'type': 'Polygon',
            'coordinates': [[
                [W, S],
                [E, S],
                [E, N],
                [W, N],
                [W, S]
            ]]
        }

        return polygon

    @staticmethod
    def pixel_to_geographic(row: int, col: int, bbox: Tuple[float, float, float, float], 
                           shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic (lat, lon)

        Args:
            row: Pixel row
            col: Pixel column
            bbox: [S, W, N, E]
            shape: (height, width)

        Returns:
            (latitude, longitude)
        """
        S, W, N, E = bbox
        height, width = shape

        lat = S + (1 - row / height) * (N - S)
        lon = W + col / width * (E - W)

        return lat, lon

    @staticmethod
    def geographic_to_pixel(lat: float, lon: float, bbox: Tuple[float, float, float, float],
                           shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel

        Args:
            lat: Latitude
            lon: Longitude
            bbox: [S, W, N, E]
            shape: (height, width)

        Returns:
            (row, col) pixel coordinates
        """
        S, W, N, E = bbox
        height, width = shape

        row = int((1 - (lat - S) / (N - S)) * height)
        col = int((lon - W) / (E - W) * width)

        return row, col


class TimeSeriesUtils:
    """Time series manipulation utilities"""

    @staticmethod
    def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """
        Compute rolling mean

        Args:
            data: Input array
            window: Window size

        Returns:
            Rolling mean
        """
        if len(data) < window:
            return np.full_like(data, np.mean(data))

        result = np.convolve(data, np.ones(window)/window, mode='same')

        return result

    @staticmethod
    def seasonal_decompose(data: np.ndarray, period: int = 12) -> Dict[str, np.ndarray]:
        """
        Simple seasonal decomposition

        Args:
            data: Input time series
            period: Seasonal period (e.g., 12 for monthly data)

        Returns:
            Dict with 'trend', 'seasonal', 'residual'
        """
        # Trend
        trend = TimeSeriesUtils.rolling_mean(data, period)

        # Detrended
        detrended = data - trend

        # Seasonal (average same season across years)
        seasonal = np.zeros_like(data)
        n_periods = len(data) // period

        for i in range(period):
            seasonal_values = detrended[i::period][:n_periods]
            seasonal_avg = np.nanmean(seasonal_values)
            seasonal[i::period] = seasonal_avg

        # Residual
        residual = data - trend - seasonal

        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }

    @staticmethod
    def autocorrelation(data: np.ndarray, max_lag: int = 20) -> Dict[int, float]:
        """
        Compute autocorrelation

        Args:
            data: Input time series
            max_lag: Maximum lag to compute

        Returns:
            Dict of lag: correlation pairs
        """
        data_normalized = (data - np.mean(data)) / np.std(data)

        acf = {}
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean(data_normalized[:-lag] * data_normalized[lag:])

        return acf


class FileHandlers:
    """File I/O utilities"""

    @staticmethod
    def safe_load_json(filepath: str, default: Dict = None) -> Dict:
        """
        Safely load JSON file with fallback

        Args:
            filepath: Path to JSON file
            default: Default dict if file doesn't exist

        Returns:
            Loaded dict or default
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return default or {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {filepath}")
            return default or {}

    @staticmethod
    def safe_save_json(data: Dict, filepath: str) -> bool:
        """
        Safely save dict as JSON

        Args:
            data: Dictionary to save
            filepath: Output path

        Returns:
            True if successful
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            return False

    @staticmethod
    def get_file_size(filepath: str) -> str:
        """
        Get human-readable file size

        Args:
            filepath: Path to file

        Returns:
            Size string (e.g., "1.5 MB")
        """
        size_bytes = Path(filepath).stat().st_size

        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024

        return f"{size_bytes:.1f} TB"


class PerformanceMetrics:
    """Calculate custom performance metrics"""

    @staticmethod
    def basis_risk_score(satellite_payout: np.ndarray, actual_loss: np.ndarray) -> float:
        """
        Calculate basis risk score (lower = better)

        Args:
            satellite_payout: Binary payout decisions from satellite
            actual_loss: Actual loss occurred

        Returns:
            Basis risk score (0-1)
        """
        # False positive: payout but no loss
        fp = np.sum((satellite_payout == 1) & (actual_loss < 0.1))

        # False negative: loss but no payout
        fn = np.sum((satellite_payout == 0) & (actual_loss > 0.2))

        # Basis risk
        basis_risk = (fp + fn) / len(satellite_payout)

        return basis_risk

    @staticmethod
    def economic_impact(
        farmers_covered: int,
        avg_farm_size_ha: float,
        insurance_multiplier: float = 1.5
    ) -> Dict[str, float]:
        """
        Estimate economic impact

        Args:
            farmers_covered: Number of farmers
            avg_farm_size_ha: Average farm size
            insurance_multiplier: Insurance coverage multiplier on harvest value

        Returns:
            Impact dict with different metrics
        """
        # Assumptions
        avg_crop_value_per_ha = 2500  # USD

        total_area = farmers_covered * avg_farm_size_ha
        total_crop_value = total_area * avg_crop_value_per_ha
        insured_value = total_crop_value * insurance_multiplier

        return {
            'total_area_ha': int(total_area),
            'total_crop_value_usd': int(total_crop_value),
            'insured_value_usd': int(insured_value),
            'value_per_farmer_usd': int(insured_value / farmers_covered)
        }


class ProgressTracker:
    """Track long-running process progress"""

    def __init__(self, total_steps: int, name: str = "Process"):
        """Initialize progress tracker"""
        self.total_steps = total_steps
        self.current_step = 0
        self.name = name
        self.start_time = datetime.now()

    def update(self, increment: int = 1) -> None:
        """Update progress"""
        self.current_step += increment

        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current_step / (elapsed + 1e-10)
        remaining = (self.total_steps - self.current_step) / (rate + 1e-10)

        pct = (self.current_step / self.total_steps) * 100

        logger.info(f"{self.name}: {pct:.1f}% ({self.current_step}/{self.total_steps}) "
                   f"ETA: {int(remaining)}s")

    def finish(self) -> None:
        """Mark as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    # Example usage

    # Data validation
    data = np.random.randn(100)
    validator = DataValidator()
    print(f"Data valid: {validator.check_data_range(data, -5, 5)}")

    # Coordinate transforms
    ct = CoordinateTransforms()
    bbox = (20, 87, 27, 93)  # Bangladesh
    lat, lon = ct.pixel_to_geographic(64, 64, bbox, (128, 128))
    print(f"Pixel (64, 64) -> Geographic ({lat:.2f}, {lon:.2f})")

    # Time series decomposition
    ts = np.sin(np.linspace(0, 4*np.pi, 120)) + np.random.randn(120) * 0.1
    ts_utils = TimeSeriesUtils()
    decomposed = ts_utils.seasonal_decompose(ts, period=12)
    print(f"Seasonal decomposition complete: {list(decomposed.keys())}")

    # Economic impact
    metrics = PerformanceMetrics()
    impact = metrics.economic_impact(farmers_covered=1000000, avg_farm_size_ha=0.5)
    print(f"\nEconomic Impact (1M farmers):")
    for k, v in impact.items():
        print(f"  {k}: {v:,}")
