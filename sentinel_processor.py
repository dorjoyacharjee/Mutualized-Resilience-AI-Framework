"""
Sentinel-1/2 Satellite Data Processor
Accesses Sentinel data via Google Earth Engine

Data Sources:
- Sentinel-1: https://registry.opendata.aws/sentinel-1/
- Sentinel-2: https://registry.opendata.aws/sentinel-2/
License: Copernicus Open Access
"""

import ee
import os
import yaml
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentinelProcessor:
    """Process Sentinel-1 and Sentinel-2 imagery via GEE"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Init GEE
        try:
            ee.Initialize()
            logger.info("Google Earth Engine initialized")
        except Exception as e:
            logger.warning(f"GEE init failed, attempting service account auth: {e}")
            # Service account auth
            service_account = os.getenv('GEE_SERVICE_ACCOUNT')
            key_file = os.getenv('GEE_PRIVATE_KEY_PATH')
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials)
        
        self.s1_cfg = self.config['data_sources']['sentinel1']
        self.s2_cfg = self.config['data_sources']['sentinel2']
        
        # Region of interest (Bangladesh)
        bbox = self.config['region']['bbox']  # [S, W, N, E]
        self.roi = ee.Geometry.Rectangle(bbox)
        
        self.output_dir = Path(self.config['paths']['data_raw']) / 'sentinel'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sentinel1_collection(
        self,
        start_date: str,
        end_date: str,
        polarization: List[str] = None
    ) -> ee.ImageCollection:
        """
        Retrieve Sentinel-1 SAR imagery
        
        Args:
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            polarization: ['VV', 'VH'] or subset
            
        Returns:
            ee.ImageCollection
        """
        polarization = polarization or self.s1_cfg['polarization']
        
        collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(self.roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.eq('orbitProperties_pass', self.s1_cfg['orbit']))
            .select(polarization)
        )
        
        logger.info(f"S1 collection size: {collection.size().getInfo()}")
        return collection
    
    def get_sentinel2_collection(
        self,
        start_date: str,
        end_date: str,
        cloud_cover_max: int = None
    ) -> ee.ImageCollection:
        """
        Retrieve Sentinel-2 optical imagery
        
        Args:
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            cloud_cover_max: Max cloud cover percentage
            
        Returns:
            ee.ImageCollection
        """
        cloud_cover_max = cloud_cover_max or self.s2_cfg['cloud_cover_max']
        bands = self.s2_cfg['bands']
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(self.roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
            .select(bands)
        )
        
        logger.info(f"S2 collection size: {collection.size().getInfo()}")
        return collection
    
    def calculate_ndvi(self, image: ee.Image) -> ee.Image:
        """Calculate NDVI from Sentinel-2"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def calculate_ndwi(self, image: ee.Image) -> ee.Image:
        """Calculate NDWI from Sentinel-2"""
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return image.addBands(ndwi)
    
    def export_to_drive(
        self,
        image: ee.Image,
        description: str,
        folder: str = 'EarthEngineExports'
    ) -> ee.batch.Task:
        """
        Export image to Google Drive
        
        Args:
            image: Image to export
            description: Export task name
            folder: Drive folder name
            
        Returns:
            Export task
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            region=self.roi,
            scale=10,  # 10m resolution
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
        logger.info(f"Export started: {description}")
        return task
    
    def process_flood_detection(
        self,
        pre_flood_date: str,
        post_flood_date: str,
        output_name: str
    ) -> None:
        """
        Detect flood extent using SAR change detection
        
        Args:
            pre_flood_date: Date before flood 'YYYY-MM-DD'
            post_flood_date: Date after flood 'YYYY-MM-DD'
            output_name: Name for output file
        """
        # Get pre-flood baseline
        pre_flood = self.get_sentinel1_collection(
            (datetime.strptime(pre_flood_date, '%Y-%m-%d') - timedelta(days=15)).strftime('%Y-%m-%d'),
            pre_flood_date
        ).median()
        
        # Get post-flood image
        post_flood = self.get_sentinel1_collection(
            post_flood_date,
            (datetime.strptime(post_flood_date, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d')
        ).median()
        
        # Calc backscatter difference
        diff = post_flood.subtract(pre_flood).select('VH')
        
        # Threshold for water detection (backscatter decrease > 3dB)
        flood_mask = diff.lt(-3)
        
        # Export
        self.export_to_drive(
            flood_mask.toByte(),
            f'flood_detection_{output_name}',
            folder='sentinel_floods'
        )
        
        logger.info(f"Flood detection processed: {output_name}")
    
    def process_drought_monitoring(
        self,
        start_date: str,
        end_date: str,
        output_name: str
    ) -> None:
        """
        Monitor drought using NDVI/NDWI time series
        
        Args:
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            output_name: Name for output
        """
        # Get S2 collection
        collection = self.get_sentinel2_collection(start_date, end_date)
        
        # Calculate indices
        with_indices = collection.map(self.calculate_ndvi).map(self.calculate_ndwi)
        
        # Compute temporal mean
        mean_ndvi = with_indices.select('NDVI').mean()
        mean_ndwi = with_indices.select('NDWI').mean()
        
        # Compute temporal std dev
        std_ndvi = with_indices.select('NDVI').reduce(ee.Reducer.stdDev())
        
        # Combine into single image
        output = ee.Image.cat([mean_ndvi, mean_ndwi, std_ndvi])
        
        # Export
        self.export_to_drive(
            output,
            f'drought_indices_{output_name}',
            folder='sentinel_drought'
        )
        
        logger.info(f"Drought monitoring processed: {output_name}")


if __name__ == "__main__":
    processor = SentinelProcessor()
    
    # Example: Process 2020 monsoon flood
    processor.process_flood_detection(
        pre_flood_date='2020-06-01',
        post_flood_date='2020-07-15',
        output_name='2020_monsoon'
    )
    
    # Example: Monitor 2019 drought season
    processor.process_drought_monitoring(
        start_date='2018-12-01',
        end_date='2019-03-31',
        output_name='2018_2019_dry_season'
    )
