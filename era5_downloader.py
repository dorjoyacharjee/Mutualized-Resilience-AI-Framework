"""
ERA5 Reanalysis Data Downloader
Downloads historical climate reanalysis from ECMWF

Data Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
License: Copernicus License
"""

import cdsapi
import os
import yaml
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERA5Downloader:
    """ERA5 reanalysis data acquisition"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.era5_cfg = self.config['data_sources']['era5']
        self.client = cdsapi.Client()
        
        self.output_dir = Path(self.config['paths']['data_raw']) / 'era5'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_monthly(
        self,
        variables: List[str] = None,
        year_range: Tuple[int, int] = None,
        months: List[int] = None
    ) -> str:
        """
        Download ERA5 monthly aggregated data
        
        Args:
            variables: ERA5 variable names
            year_range: (start_year, end_year)
            months: List of months (1-12)
            
        Returns:
            Path to downloaded file
        """
        variables = variables or self.era5_cfg['variables']
        
        if year_range is None:
            start_str, end_str = self.era5_cfg['temporal_range']
            year_range = (
                int(start_str.split('-')),
                int(end_str.split('-'))
            )
        
        months = months or list(range(1, 13))
        bbox = self.era5_cfg['spatial_bounds']  # [S, W, N, E]
        
        # Convert to CDS format [N, W, S, E]
        cds_bbox = [bbox, bbox, bbox, bbox]
        
        logger.info(f"Downloading ERA5 data for {year_range}-{year_range}")
        
        request = {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': variables,
            'year': [str(y) for y in range(year_range, year_range + 1)],
            'month': [f'{m:02d}' for m in months],
            'time': '00:00',
            'area': cds_bbox,
            'format': 'netcdf',
        }
        
        output_file = self.output_dir / f"era5_monthly_{year_range}-{year_range}.nc"
        
        try:
            self.client.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                request,
                str(output_file)
            )
            logger.info(f"Downloaded: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise


if __name__ == "__main__":
    downloader = ERA5Downloader()
    downloader.download_monthly()
