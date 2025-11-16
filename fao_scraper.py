"""
FAOSTAT Agricultural Data Scraper
Downloads crop yield, production, price data

Data Source: https://www.fao.org/faostat/en/#data
License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO (CC BY-NC-SA 3.0 IGO)
"""

import requests
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import List, Dict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAODataScraper:
    """Scrape agricultural statistics from FAOSTAT"""
    
    # FAOSTAT bulk download base URL
    BASE_URL = "https://fenixservices.fao.org/faostat/static/bulkdownloads"
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fao_cfg = self.config['data_sources']['faostat']
        self.output_dir = Path(self.config['paths']['data_raw']) / 'faostat'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_domain(self, domain_code: str) -> str:
        """
        Download entire FAOSTAT domain (dataset)
        
        Args:
            domain_code: Domain code (e.g., 'QCL' for crops)
            
        Returns:
            Path to downloaded file
        """
        # Construct download URL
        url = f"{self.BASE_URL}/{domain_code}_E_All_Data_(Normalized).zip"
        
        output_file = self.output_dir / f"{domain_code}_all_data.zip"
        
        logger.info(f"Downloading FAOSTAT domain: {domain_code}")
        logger.info(f"URL: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Download failed for {domain_code}: {e}")
            raise
    
    def extract_and_filter(
        self,
        domain_code: str,
        countries: List[str],
        elements: List[str] = None,
        years: List[int] = None
    ) -> pd.DataFrame:
        """
        Extract and filter FAOSTAT data
        
        Args:
            domain_code: FAOSTAT domain code
            countries: List of country names
            elements: Data elements to keep (e.g., ['Yield', 'Production'])
            years: Year range to filter
            
        Returns:
            Filtered DataFrame
        """
        # Download if not exists
        zip_path = self.output_dir / f"{domain_code}_all_data.zip"
        if not zip_path.exists():
            self.download_domain(domain_code)
        
        # Read from zip (FAOSTAT provides CSV inside zip)
        logger.info(f"Reading data from {zip_path}")
        
        try:
            # Read all CSVs in zip
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError(f"No CSV files found in {zip_path}")
                
                # Read the main data file (usually ends with _Normalized.csv)
                main_csv = [f for f in csv_files if 'Normalized' in f]
                with zip_ref.open(main_csv) as csv_file:
                    df = pd.read_csv(csv_file, encoding='latin-1')
        
        except Exception as e:
            logger.error(f"Failed to read data: {e}")
            raise
        
        # Filter by country
        df = df[df['Area'].isin(countries)]
        
        # Filter by element if specified
        if elements:
            df = df[df['Element'].isin(elements)]
        
        # Filter by year if specified
        if years:
            df = df[df['Year'].between(years, years)]
        
        logger.info(f"Filtered data shape: {df.shape}")
        
        # Save processed data
        output_csv = self.output_dir / f"{domain_code}_filtered.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved filtered data: {output_csv}")
        
        return df
    
    def get_crop_data(self) -> pd.DataFrame:
        """Get crop production and yield data for configured countries"""
        countries = self.fao_cfg['countries']
        elements = self.fao_cfg['elements']
        years = self.fao_cfg['years']
        
        return self.extract_and_filter(
            domain_code='QCL',  # Crops and livestock products
            countries=countries,
            elements=elements,
            years=years
        )
    
    def get_price_data(self) -> pd.DataFrame:
        """Get agricultural price data"""
        countries = self.fao_cfg['countries']
        years = self.fao_cfg['years']
        
        return self.extract_and_filter(
            domain_code='PP',  # Producer prices
            countries=countries,
            elements=['Producer Price (USD/tonne)'],
            years=years
        )


if __name__ == "__main__":
    scraper = FAODataScraper()
    
    # Download crop data
    crop_data = scraper.get_crop_data()
    print(f"Crop data shape: {crop_data.shape}")
    
    # Download price data
    price_data = scraper.get_price_data()
    print(f"Price data shape: {price_data.shape}")
