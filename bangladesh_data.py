"""
Bangladesh-Specific Data Handler
Manages local agricultural, economic, and cooperative network data

Data Sources:
- Bangladesh Bureau of Statistics (BBS): http://bbs.gov.bd
- Bangladesh Open Data: http://data.gov.bd
- Ministry of Agriculture
- Directorate of Cooperatives

Author: Dorjoy Acharjee

"""

import pandas as pd
import numpy as np
import requests
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BangladeshDataManager:
    """Handle Bangladesh-specific agricultural and economic data"""

    # BBS domains
    BBS_DOMAINS = {
        'agriculture': 'http://nsds.bbs.gov.bd/en/domains/1/Agriculture',
        'economy': 'http://nsds.bbs.gov.bd/en/domains/5/National-Accounts',
        'population': 'http://nsds.bbs.gov.bd/en/domains/2/Population-and-Housing'
    }

    # Open data portal
    OPEN_DATA_URL = 'http://data.gov.bd/api/3/action'

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Bangladesh data manager"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config['paths']['data_external']) / 'bangladesh'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Division boundaries (approx coordinates for 8 divisions)
        self.divisions = self._load_division_data()

        logger.info("BangladeshDataManager initialized")

    def _load_division_data(self) -> Dict:
        """Load Bangladesh administrative division metadata"""
        divisions_data = {
            'Dhaka': {
                'bbox': [23.5, 89.5, 24.5, 90.5],
                'districts': ['Dhaka', 'Gazipur', 'Kishoreganj', 'Manikganj', 'Munshiganj', 
                             'Narayanganj', 'Narsingdi', 'Tangail', 'Faridpur', 'Gopalganj',
                             'Madaripur', 'Rajbari', 'Shariatpur'],
                'area_km2': 20593.74,
                'rural_population': 26000000,
                'climate_zone': 'humid_subtropical',
                'primary_hazards': ['flood', 'riverbank_erosion']
            },
            'Chittagong': {
                'bbox': [21.5, 91.0, 24.0, 92.5],
                'districts': ['Chittagong', 'Cox\'s Bazar', 'Feni', 'Khagrachhari', 
                             'Lakshmipur', 'Noakhali', 'Rangamati', 'Bandarban',
                             'Brahmanbaria', 'Chandpur', 'Comilla'],
                'area_km2': 33771.18,
                'rural_population': 22000000,
                'climate_zone': 'tropical_monsoon',
                'primary_hazards': ['cyclone', 'flood', 'landslide']
            },
            'Rajshahi': {
                'bbox': [24.0, 88.0, 25.5, 89.5],
                'districts': ['Rajshahi', 'Bogra', 'Joypurhat', 'Naogaon', 'Natore',
                             'Nawabganj', 'Pabna', 'Sirajganj'],
                'area_km2': 18174.43,
                'rural_population': 15000000,
                'climate_zone': 'semi_arid',
                'primary_hazards': ['drought', 'heatwave']
            },
            'Khulna': {
                'bbox': [21.5, 89.0, 23.5, 89.8],
                'districts': ['Khulna', 'Bagerhat', 'Chuadanga', 'Jessore', 'Jhenaidah',
                             'Kushtia', 'Magura', 'Meherpur', 'Narail', 'Satkhira'],
                'area_km2': 22284.22,
                'rural_population': 12000000,
                'climate_zone': 'tropical_monsoon',
                'primary_hazards': ['coastal_flood', 'cyclone', 'salinity']
            },
            'Sylhet': {
                'bbox': [24.0, 91.0, 25.5, 92.5],
                'districts': ['Sylhet', 'Habiganj', 'Moulvibazar', 'Sunamganj'],
                'area_km2': 12596.58,
                'rural_population': 8000000,
                'climate_zone': 'humid_subtropical',
                'primary_hazards': ['flash_flood', 'landslide']
            },
            'Barisal': {
                'bbox': [21.8, 89.8, 23.0, 90.8],
                'districts': ['Barisal', 'Barguna', 'Bhola', 'Jhalokati', 'Patuakhali', 'Pirojpur'],
                'area_km2': 13225.20,
                'rural_population': 7000000,
                'climate_zone': 'tropical_monsoon',
                'primary_hazards': ['coastal_flood', 'cyclone', 'riverbank_erosion']
            },
            'Rangpur': {
                'bbox': [25.0, 88.5, 26.5, 89.8],
                'districts': ['Rangpur', 'Dinajpur', 'Gaibandha', 'Kurigram', 'Lalmonirhat',
                             'Nilphamari', 'Panchagarh', 'Thakurgaon'],
                'area_km2': 16184.99,
                'rural_population': 13000000,
                'climate_zone': 'humid_subtropical',
                'primary_hazards': ['drought', 'flood']
            },
            'Mymensingh': {
                'bbox': [24.0, 89.8, 25.5, 91.0],
                'districts': ['Mymensingh', 'Jamalpur', 'Netrokona', 'Sherpur'],
                'area_km2': 10584.06,
                'rural_population': 9000000,
                'climate_zone': 'humid_subtropical',
                'primary_hazards': ['flood', 'flash_flood']
            }
        }
        return divisions_data

    def load_agricultural_census(self, year: int = 2008) -> pd.DataFrame:
        """
        Load agricultural census data

        Args:
            year: Census year (2008 is most recent publicly available)

        Returns:
            DataFrame with census data
        """
        # Path where user should place manually downloaded files
        census_file = self.output_dir / f'agricultural_census_{year}.csv'

        if not census_file.exists():
            logger.warning(f"Census file not found: {census_file}")
            logger.info("Please download from: http://data.gov.bd/dataset/agriculture-census-2008")
            logger.info(f"Place the CSV file at: {census_file}")

            # Return simulated structure for development
            return self._simulate_census_data()

        df = pd.read_csv(census_file)
        logger.info(f"Loaded agricultural census: {df.shape}")
        return df

    def _simulate_census_data(self) -> pd.DataFrame:
        """Generate simulated census data for testing (replace with real data)"""
        logger.warning("Using SIMULATED census data - replace with real data for production!")

        data = []
        for div_name, div_data in self.divisions.items():
            for district in div_data['districts'][:3]:  # First 3 districts per division
                data.append({
                    'division': div_name,
                    'district': district,
                    'total_farms': np.random.randint(50000, 200000),
                    'farm_households': np.random.randint(40000, 180000),
                    'cultivated_area_ha': np.random.randint(20000, 100000),
                    'rice_area_ha': np.random.randint(15000, 80000),
                    'wheat_area_ha': np.random.randint(1000, 10000),
                    'jute_area_ha': np.random.randint(500, 5000),
                    'avg_farm_size_ha': np.random.uniform(0.3, 1.2),
                    'irrigated_pct': np.random.uniform(30, 80),
                    'mechanized_pct': np.random.uniform(10, 40)
                })

        return pd.DataFrame(data)

    def get_cooperative_registry(self) -> pd.DataFrame:
        """
        Load cooperative (somobay somiti) registry data

        Returns:
            DataFrame with cooperative locations and membership
        """
        coop_file = self.output_dir / 'cooperative_registry.csv'

        if not coop_file.exists():
            logger.warning("Cooperative registry not found")
            logger.info("Contact: Directorate of Cooperatives, Bangladesh")
            logger.info("Website: http://www.lgd.gov.bd/")

            # Generate simulated data
            return self._simulate_cooperative_data()

        df = pd.read_csv(coop_file)
        logger.info(f"Loaded cooperative registry: {df.shape}")
        return df

    def _simulate_cooperative_data(self) -> pd.DataFrame:
        """Simulate cooperative network data"""
        logger.warning("Using SIMULATED cooperative data")

        np.random.seed(42)
        n_cooperatives = 500  # Representing sample of ~30,000 total

        data = []
        for i in range(n_cooperatives):
            div = np.random.choice(list(self.divisions.keys()))
            div_data = self.divisions[div]

            # Random location within division bbox
            lat = np.random.uniform(div_data['bbox'][0], div_data['bbox'][2])
            lon = np.random.uniform(div_data['bbox'][1], div_data['bbox'][3])

            data.append({
                'coop_id': f'COOP_{i:05d}',
                'name': f'{div} Agricultural Cooperative {i}',
                'division': div,
                'district': np.random.choice(div_data['districts']),
                'latitude': lat,
                'longitude': lon,
                'members': np.random.randint(80, 300),
                'registered_farmland_ha': np.random.randint(40, 200),
                'established_year': np.random.randint(1990, 2020),
                'primary_crop': np.random.choice(['rice', 'jute', 'wheat', 'vegetables']),
                'has_irrigation': np.random.choice([True, False], p=[0.6, 0.4]),
                'flood_risk': 'high' if div in ['Khulna', 'Sylhet', 'Barisal'] else 'medium',
                'drought_risk': 'high' if div in ['Rajshahi', 'Rangpur'] else 'low'
            })

        df = pd.DataFrame(data)

        # Save simulated data
        df.to_csv(self.output_dir / 'cooperative_registry_simulated.csv', index=False)
        logger.info(f"Saved simulated cooperative data: {self.output_dir / 'cooperative_registry_simulated.csv'}")

        return df

    def get_disaster_history(self, start_year: int = 2000, end_year: int = 2023) -> pd.DataFrame:
        """
        Compile historical disaster events in Bangladesh

        Args:
            start_year: Start year for disaster records
            end_year: End year

        Returns:
            DataFrame with disaster events
        """
        # Major documented disasters (curated from EM-DAT, ReliefWeb, news sources)
        disasters = [
            {'year': 2020, 'month': 7, 'type': 'flood', 'division': 'Sylhet', 
             'affected_people': 5400000, 'economic_loss_usd': 1200000000},
            {'year': 2019, 'month': 8, 'type': 'flood', 'division': 'Dhaka',
             'affected_people': 7800000, 'economic_loss_usd': 1800000000},
            {'year': 2017, 'month': 8, 'type': 'flood', 'division': 'Multiple',
             'affected_people': 8000000, 'economic_loss_usd': 2500000000},
            {'year': 2019, 'month': 5, 'type': 'cyclone', 'division': 'Khulna',
             'affected_people': 1000000, 'economic_loss_usd': 500000000},
            {'year': 2016, 'month': 5, 'type': 'cyclone', 'division': 'Chittagong',
             'affected_people': 800000, 'economic_loss_usd': 300000000},
            {'year': 2009, 'month': 5, 'type': 'cyclone', 'division': 'Khulna',
             'affected_people': 3900000, 'economic_loss_usd': 2700000000},
            {'year': 2018, 'month': 3, 'type': 'drought', 'division': 'Rajshahi',
             'affected_people': 500000, 'economic_loss_usd': 150000000},
            {'year': 2016, 'month': 3, 'type': 'drought', 'division': 'Rangpur',
             'affected_people': 700000, 'economic_loss_usd': 200000000},
            {'year': 2010, 'month': 8, 'type': 'flood', 'division': 'Multiple',
             'affected_people': 13000000, 'economic_loss_usd': 3200000000},
            {'year': 2007, 'month': 11, 'type': 'cyclone', 'division': 'Barisal',
             'affected_people': 8900000, 'economic_loss_usd': 2300000000},
        ]

        df = pd.DataFrame(disasters)
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

        logger.info(f"Compiled {len(df)} major disaster events ({start_year}-{end_year})")
        return df

    def calculate_risk_scores(self, cooperative_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores for each cooperative based on location

        Args:
            cooperative_data: DataFrame with cooperative info

        Returns:
            DataFrame with added risk scores
        """
        df = cooperative_data.copy()

        # Map division to hazard intensities (0-10 scale)
        flood_intensity = {
            'Khulna': 9, 'Sylhet': 9, 'Barisal': 8, 'Dhaka': 7,
            'Chittagong': 7, 'Mymensingh': 6, 'Rangpur': 5, 'Rajshahi': 3
        }

        drought_intensity = {
            'Rajshahi': 9, 'Rangpur': 8, 'Dhaka': 5, 'Mymensingh': 4,
            'Khulna': 3, 'Chittagong': 2, 'Sylhet': 2, 'Barisal': 2
        }

        cyclone_intensity = {
            'Khulna': 9, 'Barisal': 9, 'Chittagong': 8, 'Sylhet': 2,
            'Dhaka': 1, 'Rajshahi': 1, 'Rangpur': 1, 'Mymensingh': 1
        }

        df['flood_risk_score'] = df['division'].map(flood_intensity)
        df['drought_risk_score'] = df['division'].map(drought_intensity)
        df['cyclone_risk_score'] = df['division'].map(cyclone_intensity)

        # Composite risk (weighted)
        df['composite_risk'] = (
            0.4 * df['flood_risk_score'] +
            0.3 * df['drought_risk_score'] +
            0.3 * df['cyclone_risk_score']
        )

        logger.info("Calculated risk scores for cooperatives")
        return df

    def export_geojson(self, data: pd.DataFrame, output_name: str) -> str:
        """
        Export data as GeoJSON for mapping

        Args:
            data: DataFrame with latitude/longitude columns
            output_name: Output filename (without extension)

        Returns:
            Path to exported GeoJSON file
        """
        features = []

        for idx, row in data.iterrows():
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']]
                },
                'properties': {k: v for k, v in row.items() 
                              if k not in ['latitude', 'longitude']}
            }
            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        output_path = self.output_dir / f'{output_name}.geojson'
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)

        logger.info(f"Exported GeoJSON: {output_path}")
        return str(output_path)


if __name__ == "__main__":
    # Example usage
    manager = BangladeshDataManager()

    # Load census data
    census = manager.load_agricultural_census()
    print(f"\nCensus data shape: {census.shape}")
    print(census.head())

    # Get cooperative registry
    coops = manager.get_cooperative_registry()
    print(f"\nCooperative data shape: {coops.shape}")

    # Calculate risk scores
    coops_with_risk = manager.calculate_risk_scores(coops)
    print(f"\nRisk scores calculated:")
    print(coops_with_risk[['coop_id', 'division', 'flood_risk_score', 
                            'drought_risk_score', 'composite_risk']].head())

    # Export to GeoJSON
    geojson_path = manager.export_geojson(coops_with_risk, 'cooperatives_bangladesh')
    print(f"\nGeoJSON exported: {geojson_path}")

    # Get disaster history
    disasters = manager.get_disaster_history(2015, 2023)
    print(f"\nDisaster events (2015-2023):")
    print(disasters)
