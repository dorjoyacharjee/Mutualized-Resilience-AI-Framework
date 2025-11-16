"""
Climate Data Preprocessor
Processes CMIP6 and ERA5 NetCDF files for model training

Handles:
- Temporal aggregation (daily -> monthly)
- Spatial interpolation and regridding
- Anomaly calculation
- Feature engineering for LSTM input

Author: Dorjoy Acharjee
"""

import xarray as xr
import numpy as np
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import interpolate
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimatePreprocessor:
    """Preprocess climate data from CMIP6 and ERA5"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.preproc_cfg = self.config['preprocessing']['climate']

        # Input/output paths
        self.input_dir = Path(self.config['paths']['data_raw'])
        self.output_dir = Path(self.config['paths']['data_processed']) / 'climate'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Target grid resolution
        self.target_res = self.preproc_cfg['spatial_resolution']  # degrees

        # Region bbox
        self.bbox = self.config['region']['bbox']  # [S, W, N, E]

        logger.info("ClimatePreprocessor initialized")

    def load_netcdf(self, filepath: str) -> xr.Dataset:
        """
        Load NetCDF file with error handling

        Args:
            filepath: Path to NetCDF file

        Returns:
            xarray Dataset
        """
        try:
            ds = xr.open_dataset(filepath, engine='netcdf4')
            logger.info(f"Loaded: {filepath}")
            logger.info(f"  Variables: {list(ds.data_vars)}")
            logger.info(f"  Dims: {dict(ds.dims)}")
            return ds
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise

    def subset_spatial(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Subset to region of interest (Bangladesh)

        Args:
            ds: Input dataset

        Returns:
            Spatially subsetted dataset
        """
        # Detect lat/lon dimension names (varies by source)
        lat_names = ['lat', 'latitude', 'y']
        lon_names = ['lon', 'longitude', 'x']

        lat_dim = [d for d in ds.dims if d in lat_names][0]
        lon_dim = [d for d in ds.dims if d in lon_names][0]

        S, W, N, E = self.bbox

        # Subset
        ds_subset = ds.sel(
            {lat_dim: slice(S, N), lon_dim: slice(W, E)}
        )

        logger.info(f"Spatial subset: {dict(ds_subset.dims)}")
        return ds_subset

    def regrid(self, ds: xr.Dataset, target_resolution: float = None) -> xr.Dataset:
        """
        Regrid to uniform resolution

        Args:
            ds: Input dataset
            target_resolution: Target grid resolution in degrees

        Returns:
            Regridded dataset
        """
        target_res = target_resolution or self.target_res

        # Detect coordinate names
        lat_names = ['lat', 'latitude', 'y']
        lon_names = ['lon', 'longitude', 'x']

        lat_dim = [d for d in ds.dims if d in lat_names][0]
        lon_dim = [d for d in ds.dims if d in lon_names][0]

        # Create target grid
        S, W, N, E = self.bbox
        new_lats = np.arange(S, N, target_res)
        new_lons = np.arange(W, E, target_res)

        # Regrid using linear interpolation
        ds_regrid = ds.interp(
            {lat_dim: new_lats, lon_dim: new_lons},
            method='linear'
        )

        logger.info(f"Regridded to {target_res}° resolution")
        logger.info(f"  New dims: {dict(ds_regrid.dims)}")

        return ds_regrid

    def temporal_aggregate(
        self, 
        ds: xr.Dataset, 
        freq: str = 'M'
    ) -> xr.Dataset:
        """
        Aggregate temporal resolution

        Args:
            ds: Input dataset
            freq: Aggregation frequency ('M'=monthly, 'Y'=yearly)

        Returns:
            Temporally aggregated dataset
        """
        # Detect time dimension
        time_names = ['time', 'date', 't']
        time_dim = [d for d in ds.dims if d in time_names][0]

        # Resample
        ds_agg = ds.resample({time_dim: freq}).mean()

        logger.info(f"Temporal aggregation to {freq}")
        logger.info(f"  Original time steps: {len(ds[time_dim])}")
        logger.info(f"  Aggregated time steps: {len(ds_agg[time_dim])}")

        return ds_agg

    def calculate_anomalies(
        self,
        ds: xr.Dataset,
        baseline_years: Tuple[int, int] = None,
        variables: List[str] = None
    ) -> xr.Dataset:
        """
        Calculate anomalies relative to baseline period

        Args:
            ds: Input dataset
            baseline_years: (start_year, end_year) for baseline climatology
            variables: Variables to compute anomalies for

        Returns:
            Dataset with anomaly variables added
        """
        baseline_years = baseline_years or tuple(self.preproc_cfg['anomaly_baseline'])
        variables = variables or list(ds.data_vars)

        # Detect time dimension
        time_names = ['time', 'date', 't']
        time_dim = [d for d in ds.dims if d in time_names][0]

        # Extract baseline period
        baseline_start = f"{baseline_years[0]}-01-01"
        baseline_end = f"{baseline_years[1]}-12-31"

        try:
            baseline = ds.sel({time_dim: slice(baseline_start, baseline_end)})
        except:
            # If slice fails, try indexing by year
            baseline = ds.where(
                (ds[time_dim].dt.year >= baseline_years[0]) &
                (ds[time_dim].dt.year <= baseline_years[1]),
                drop=True
            )

        # Compute climatology (mean over baseline)
        climatology = baseline.mean(dim=time_dim)

        # Compute anomalies
        ds_anomaly = ds.copy()
        for var in variables:
            if var in ds.data_vars:
                anomaly_var = f"{var}_anomaly"
                ds_anomaly[anomaly_var] = ds[var] - climatology[var]

        logger.info(f"Calculated anomalies for {len(variables)} variables")
        logger.info(f"  Baseline: {baseline_years[0]}-{baseline_years[1]}")

        return ds_anomaly

    def engineer_features(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Engineer climate features for ML models

        Args:
            ds: Input dataset with climate variables

        Returns:
            Dataset with engineered features
        """
        ds_feat = ds.copy()

        # Detect time dimension
        time_names = ['time', 'date', 't']
        time_dim = [d for d in ds.dims if d in time_names][0]

        # Monthly indicators
        if time_dim in ds.dims:
            ds_feat['month'] = ds[time_dim].dt.month
            ds_feat['season'] = ds[time_dim].dt.month % 12 // 3 + 1  # 1=DJF, 2=MAM, 3=JJA, 4=SON

        # Temperature-based features
        if 'tas' in ds.data_vars or '2m_temperature' in ds.data_vars:
            temp_var = 'tas' if 'tas' in ds.data_vars else '2m_temperature'

            # Growing degree days (base 10°C)
            gdd = (ds[temp_var] - 283.15).clip(min=0)  # Convert K to C, threshold 10C
            ds_feat['gdd'] = gdd

            # Extreme heat days (>35°C)
            ds_feat['extreme_heat_days'] = (ds[temp_var] > 308.15).astype(int)

        # Precipitation-based features
        if 'pr' in ds.data_vars or 'total_precipitation' in ds.data_vars:
            precip_var = 'pr' if 'pr' in ds.data_vars else 'total_precipitation'

            # Dry days (precipitation < 1mm)
            ds_feat['dry_days'] = (ds[precip_var] < 0.001).astype(int)

            # Heavy precip days (>50mm)
            ds_feat['heavy_precip_days'] = (ds[precip_var] > 0.05).astype(int)

            # Consecutive dry days (simple approximation)
            # Note: Full CDD calculation requires rolling window
            ds_feat['precip_intensity'] = ds[precip_var].where(ds[precip_var] > 0, 0)

        logger.info("Engineered climate features")
        logger.info(f"  Total variables: {len(ds_feat.data_vars)}")

        return ds_feat

    def spatial_average(self, ds: xr.Dataset, mask: Optional[xr.DataArray] = None) -> pd.DataFrame:
        """
        Compute spatial averages (area-weighted)

        Args:
            ds: Input dataset
            mask: Optional mask for specific regions

        Returns:
            DataFrame with time series of spatial averages
        """
        # Detect coordinate names
        lat_names = ['lat', 'latitude', 'y']
        lon_names = ['lon', 'longitude', 'x']

        lat_dim = [d for d in ds.dims if d in lat_names][0]
        lon_dim = [d for d in ds.dims if d in lon_names][0]

        # Calculate area weights (cos of latitude)
        weights = np.cos(np.deg2rad(ds[lat_dim]))
        weights.name = "weights"

        # Apply mask if provided
        if mask is not None:
            ds_masked = ds.where(mask)
        else:
            ds_masked = ds

        # Weighted spatial mean
        ds_mean = ds_masked.weighted(weights).mean(dim=[lat_dim, lon_dim])

        # Convert to DataFrame
        df = ds_mean.to_dataframe()

        logger.info(f"Computed spatial averages: {df.shape}")

        return df

    def process_cmip6(
        self,
        input_file: str,
        output_name: str = None
    ) -> Tuple[xr.Dataset, pd.DataFrame]:
        """
        Full CMIP6 preprocessing pipeline

        Args:
            input_file: Path to CMIP6 NetCDF file
            output_name: Name for output files

        Returns:
            Tuple of (processed Dataset, spatial average DataFrame)
        """
        logger.info(f"=== Processing CMIP6: {input_file} ===")

        # Load
        ds = self.load_netcdf(input_file)

        # Subset spatial
        ds = self.subset_spatial(ds)

        # Regrid
        ds = self.regrid(ds)

        # Temporal aggregation
        ds = self.temporal_aggregate(ds, freq='M')

        # Calculate anomalies (if historical data available)
        try:
            ds = self.calculate_anomalies(ds)
        except Exception as e:
            logger.warning(f"Could not calculate anomalies: {e}")

        # Engineer features
        ds = self.engineer_features(ds)

        # Spatial average for time series
        df = self.spatial_average(ds)

        # Save outputs
        if output_name is None:
            output_name = Path(input_file).stem + '_processed'

        output_nc = self.output_dir / f"{output_name}.nc"
        output_csv = self.output_dir / f"{output_name}_timeseries.csv"

        ds.to_netcdf(output_nc)
        df.to_csv(output_csv)

        logger.info(f"Saved: {output_nc}")
        logger.info(f"Saved: {output_csv}")

        return ds, df

    def process_era5(
        self,
        input_file: str,
        output_name: str = None
    ) -> Tuple[xr.Dataset, pd.DataFrame]:
        """
        Full ERA5 preprocessing pipeline

        Args:
            input_file: Path to ERA5 NetCDF file
            output_name: Name for output files

        Returns:
            Tuple of (processed Dataset, spatial average DataFrame)
        """
        logger.info(f"=== Processing ERA5: {input_file} ===")

        # Load
        ds = self.load_netcdf(input_file)

        # ERA5 uses different naming conventions
        # Standardize variable names
        var_mapping = {
            't2m': 'tas',
            '2m_temperature': 'tas',
            'tp': 'pr',
            'total_precipitation': 'pr'
        }

        ds = ds.rename({k: v for k, v in var_mapping.items() if k in ds.data_vars})

        # Subset spatial
        ds = self.subset_spatial(ds)

        # Regrid
        ds = self.regrid(ds)

        # Temporal aggregation (ERA5 may already be monthly)
        if 'time' in ds.dims and len(ds.time) > 1000:  # Likely hourly/daily
            ds = self.temporal_aggregate(ds, freq='M')

        # Calculate anomalies
        ds = self.calculate_anomalies(ds)

        # Engineer features
        ds = self.engineer_features(ds)

        # Spatial average
        df = self.spatial_average(ds)

        # Save
        if output_name is None:
            output_name = Path(input_file).stem + '_processed'

        output_nc = self.output_dir / f"{output_name}.nc"
        output_csv = self.output_dir / f"{output_name}_timeseries.csv"

        ds.to_netcdf(output_nc)
        df.to_csv(output_csv)

        logger.info(f"Saved: {output_nc}")
        logger.info(f"Saved: {output_csv}")

        return ds, df

    def merge_timeseries(
        self,
        era5_df: pd.DataFrame,
        cmip6_df: pd.DataFrame,
        overlap_years: Tuple[int, int] = (2015, 2023)
    ) -> pd.DataFrame:
        """
        Merge ERA5 (historical) and CMIP6 (future) time series

        Args:
            era5_df: ERA5 DataFrame (historical)
            cmip6_df: CMIP6 DataFrame (projections)
            overlap_years: Years for bias correction calibration

        Returns:
            Combined DataFrame
        """
        # Bias correction: adjust CMIP6 to match ERA5 climatology
        overlap_start = f"{overlap_years[0]}-01-01"
        overlap_end = f"{overlap_years[1]}-12-31"

        era5_overlap = era5_df.loc[overlap_start:overlap_end]
        cmip6_overlap = cmip6_df.loc[overlap_start:overlap_end]

        # Calculate bias (mean difference)
        common_vars = set(era5_overlap.columns) & set(cmip6_overlap.columns)

        bias = {}
        for var in common_vars:
            bias[var] = era5_overlap[var].mean() - cmip6_overlap[var].mean()

        # Apply bias correction to full CMIP6 series
        cmip6_corrected = cmip6_df.copy()
        for var, bias_val in bias.items():
            if var in cmip6_corrected.columns:
                cmip6_corrected[var] += bias_val

        # Concatenate (ERA5 up to 2023, CMIP6 from 2024 onwards)
        era5_historical = era5_df.loc[:f"{overlap_years[1]}-12-31"]
        cmip6_future = cmip6_corrected.loc[f"{overlap_years[1]+1}-01-01":]

        combined = pd.concat([era5_historical, cmip6_future])

        logger.info(f"Merged time series: {combined.shape}")
        logger.info(f"  Historical (ERA5): {len(era5_historical)} months")
        logger.info(f"  Future (CMIP6): {len(cmip6_future)} months")

        return combined


if __name__ == "__main__":
    processor = ClimatePreprocessor()

    # Example: Process ERA5 file
    era5_file = "data/raw/era5/era5_monthly_1979-2023.nc"
    if Path(era5_file).exists():
        ds_era5, df_era5 = processor.process_era5(era5_file)
        print(f"\nProcessed ERA5: {df_era5.shape}")

    # Example: Process CMIP6 file
    cmip6_file = "data/raw/cmip6/cmip6_ssp3-7.0_tas_2025-2100.nc"
    if Path(cmip6_file).exists():
        ds_cmip6, df_cmip6 = processor.process_cmip6(cmip6_file)
        print(f"\nProcessed CMIP6: {df_cmip6.shape}")

        # Merge if both available
        if 'df_era5' in locals() and 'df_cmip6' in locals():
            combined = processor.merge_timeseries(df_era5, df_cmip6)
            output_path = processor.output_dir / "combined_climate_timeseries.csv"
            combined.to_csv(output_path)
            print(f"\nMerged time series saved: {output_path}")
