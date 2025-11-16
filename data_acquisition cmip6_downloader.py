"""
CMIP6 Climate Projection Data Downloader
Downloads climate model projections from Copernicus CDS

Data Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6
License: Copernicus License (free for research)
"""

import cdsapi
import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CMIP6Downloader:
    """Handles CMIP6 climate projection data acquisition"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize downloader with config
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract CMIP6 config
        self.cmip6_cfg = self.config['data_sources']['cmip6']
        self.api_key = os.getenv('CDS_API_KEY')
        self.api_url = self.config['api_credentials']['cds_api_url']
        
        # Init CDS API client
        self.client = cdsapi.Client(url=self.api_url, key=self.api_key)
        
        # Output dir
        self.output_dir = Path(self.config['paths']['data_raw']) / 'cmip6'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("CMIP6Downloader initialized")
    
    def download_scenario(
        self, 
        scenario: str = None,
        models: List[str] = None,
        variables: List[str] = None,
        year_range: Tuple[int, int] = None
    ) -> Dict[str, str]:
        """
        Download CMIP6 data for specified scenario
        
        Args:
            scenario: SSP scenario (e.g., 'ssp3-7.0')
            models: List of climate models
            variables: Climate variables to download
            year_range: (start_year, end_year)
            
        Returns:
            Dict mapping variables to downloaded file paths
        """
        # Use config defaults if not specified
        scenario = scenario or self.cmip6_cfg['scenario']
        models = models or self.cmip6_cfg['models']
        variables = variables or self.cmip6_cfg['variables']
        
        # Parse year range from config if not provided
        if year_range is None:
            start_str, end_str = self.cmip6_cfg['temporal_range']
            year_range = (
                int(start_str.split('-')),
                int(end_str.split('-'))
            )
        
        bbox = self.cmip6_cfg['spatial_bounds']  # [S, W, N, E]
        
        downloaded_files = {}
        
        for var in variables:
            logger.info(f"Downloading CMIP6 variable: {var} for {scenario}")
            
            # CDS API request structure
            request = {
                'format': 'netcdf',
                'temporal_resolution': 'monthly',
                'experiment': scenario,
                'level': 'single_levels',
                'variable': var,
                'model': models,
                'year': [str(y) for y in range(year_range, year_range + 1)],
                'month': [f'{m:02d}' for m in range(1, 13)],
                'area': bbox,  # [N, W, S, E] for CDS
            }
            
            output_file = self.output_dir / f"cmip6_{scenario}_{var}_{year_range}-{year_range}.nc"
            
            try:
                self.client.retrieve(
                    'projections-cmip6',
                    request,
                    str(output_file)
                )
                downloaded_files[var] = str(output_file)
                logger.info(f"Downloaded: {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to download {var}: {e}")
                # Continue with next variable
                
        return downloaded_files
    
    def download_all_scenarios(self) -> None:
        """Download data for all configured scenarios"""
        scenarios = [self.cmip6_cfg['scenario']]  # Extend for multi-scenario
        
        for scenario in scenarios:
            logger.info(f"Starting download for scenario: {scenario}")
            self.download_scenario(scenario=scenario)
            logger.info(f"Completed download for scenario: {scenario}")


if __name__ == "__main__":
    # CLI execution
    downloader = CMIP6Downloader()
    downloader.download_all_scenarios()
