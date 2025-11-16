"""
Satellite Imagery Preprocessor
Processes Sentinel-1/2 imagery for CNN training

Handles:
- SAR speckle filtering
- Cloud masking
- Index calculation (NDVI, NDWI)
- Tile generation for U-Net
- Data augmentation

Author: Dorjoy Acharjee
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import logging
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
from scipy.ndimage import uniform_filter, generic_filter
from skimage import exposure
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatellitePreprocessor:
    """Preprocess Sentinel satellite imagery"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sat_cfg = self.config['preprocessing']['satellite']

        # Paths
        self.input_dir = Path(self.config['paths']['data_raw']) / 'sentinel'
        self.output_dir = Path(self.config['paths']['data_processed']) / 'satellite'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Processing params
        self.tile_size = self.sat_cfg['tile_size']
        self.normalization = self.sat_cfg['normalization']

        logger.info("SatellitePreprocessor initialized")

    def lee_filter(self, img: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Apply Lee speckle filter to SAR imagery

        Args:
            img: Input SAR image (linear scale)
            window_size: Filter window size (must be odd)

        Returns:
            Filtered image
        """
        img_mean = uniform_filter(img, size=window_size)
        img_sqr_mean = uniform_filter(img**2, size=window_size)

        # Variance
        img_variance = img_sqr_mean - img_mean**2
        img_variance = np.maximum(img_variance, 0)  # Ensure non-negative

        # Overall variance
        overall_variance = np.var(img)

        # Lee filter weights
        img_weights = img_variance / (img_variance + overall_variance + 1e-10)

        # Apply filter
        img_filtered = img_mean + img_weights * (img - img_mean)

        return img_filtered

    def calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI

        Args:
            red: Red band
            nir: Near-infrared band

        Returns:
            NDVI array
        """
        # Avoid division by zero
        denominator = nir + red + 1e-10
        ndvi = (nir - red) / denominator

        # Clip to valid range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)

        return ndvi

    def calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate NDWI (Normalized Difference Water Index)

        Args:
            green: Green band
            nir: Near-infrared band

        Returns:
            NDWI array
        """
        denominator = green + nir + 1e-10
        ndwi = (green - nir) / denominator

        ndwi = np.clip(ndwi, -1, 1)

        return ndwi

    def cloud_mask_s2(
        self,
        scl: np.ndarray,
        valid_classes: List[int] = None
    ) -> np.ndarray:
        """
        Create cloud mask from Sentinel-2 Scene Classification Layer

        Args:
            scl: Scene Classification Layer array
            valid_classes: Valid (non-cloud) class IDs
                           Default: [4, 5, 6, 7] = vegetation, bare soil, water, unclassified

        Returns:
            Boolean mask (True = valid, False = cloud/shadow)
        """
        if valid_classes is None:
            valid_classes = [4, 5, 6, 7]  # Valid surface classes

        mask = np.isin(scl, valid_classes)

        return mask

    def normalize_image(
        self,
        img: np.ndarray,
        method: str = None,
        percentile: Tuple[float, float] = (2, 98)
    ) -> np.ndarray:
        """
        Normalize image to [0, 1] range

        Args:
            img: Input image
            method: 'minmax', 'percentile', or 'standardize'
            percentile: (low, high) percentiles for clipping

        Returns:
            Normalized image
        """
        method = method or self.normalization

        if method == 'minmax':
            img_min = np.nanmin(img)
            img_max = np.nanmax(img)
            normalized = (img - img_min) / (img_max - img_min + 1e-10)

        elif method == 'percentile':
            p_low, p_high = percentile
            vmin = np.nanpercentile(img, p_low)
            vmax = np.nanpercentile(img, p_high)
            normalized = np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)

        elif method == 'standardize':
            mean = np.nanmean(img)
            std = np.nanstd(img)
            normalized = (img - mean) / (std + 1e-10)
            # Scale to [0, 1]
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-10)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def generate_tiles(
        self,
        img: np.ndarray,
        tile_size: int = None,
        overlap: int = 0,
        mask: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Generate image tiles for CNN processing

        Args:
            img: Input image (H, W, C) or (H, W)
            tile_size: Size of square tiles
            overlap: Overlap in pixels between adjacent tiles
            mask: Optional mask to skip invalid tiles

        Returns:
            List of (tile, (row, col)) tuples
        """
        tile_size = tile_size or self.tile_size

        if img.ndim == 2:
            h, w = img.shape
            c = 1
            img = img[:, :, np.newaxis]
        else:
            h, w, c = img.shape

        stride = tile_size - overlap

        tiles = []

        for i in range(0, h - tile_size + 1, stride):
            for j in range(0, w - tile_size + 1, stride):
                tile = img[i:i+tile_size, j:j+tile_size, :]

                # Skip if mask indicates invalid region
                if mask is not None:
                    tile_mask = mask[i:i+tile_size, j:j+tile_size]
                    valid_fraction = np.mean(tile_mask)
                    if valid_fraction < 0.5:  # Skip if <50% valid pixels
                        continue

                # Skip if tile contains too many NaN/zeros
                if np.isnan(tile).sum() > 0.1 * tile.size:
                    continue

                tiles.append((tile, (i, j)))

        logger.info(f"Generated {len(tiles)} tiles of size {tile_size}x{tile_size}")

        return tiles

    def augment_tile(self, tile: np.ndarray) -> List[np.ndarray]:
        """
        Data augmentation for training tiles

        Args:
            tile: Input tile (H, W, C)

        Returns:
            List of augmented tiles
        """
        augmented = [tile]  # Original

        # Horizontal flip
        augmented.append(np.fliplr(tile))

        # Vertical flip
        augmented.append(np.flipud(tile))

        # 90° rotation
        augmented.append(np.rot90(tile, k=1))

        # 180° rotation
        augmented.append(np.rot90(tile, k=2))

        # 270° rotation
        augmented.append(np.rot90(tile, k=3))

        # Brightness adjustment (±10%)
        if tile.dtype == np.float32 or tile.dtype == np.float64:
            bright_up = np.clip(tile * 1.1, 0, 1)
            bright_down = np.clip(tile * 0.9, 0, 1)
            augmented.append(bright_up)
            augmented.append(bright_down)

        return augmented

    def process_sentinel1_flood(
        self,
        pre_event_path: str,
        post_event_path: str,
        output_name: str
    ) -> Dict:
        """
        Process Sentinel-1 SAR for flood detection

        Args:
            pre_event_path: Path to pre-flood SAR image
            post_event_path: Path to post-flood SAR image
            output_name: Name for output files

        Returns:
            Dictionary with processing metadata
        """
        logger.info(f"Processing S1 flood detection: {output_name}")

        # Load images (assume GeoTIFF format)
        with rasterio.open(pre_event_path) as src_pre:
            img_pre = src_pre.read(1).astype(np.float32)  # VH polarization
            profile = src_pre.profile

        with rasterio.open(post_event_path) as src_post:
            img_post = src_post.read(1).astype(np.float32)

        # Convert to linear scale (if in dB)
        if img_pre.min() < 0:  # Likely in dB
            img_pre = 10 ** (img_pre / 10)
            img_post = 10 ** (img_post / 10)

        # Apply Lee filter to reduce speckle
        img_pre_filtered = self.lee_filter(img_pre, window_size=5)
        img_post_filtered = self.lee_filter(img_post, window_size=5)

        # Calculate backscatter change
        change = img_post_filtered - img_pre_filtered
        change_db = 10 * np.log10(img_post_filtered / (img_pre_filtered + 1e-10))

        # Flood mask (backscatter decrease > 3dB)
        flood_mask = (change_db < -3).astype(np.uint8)

        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        flood_mask = cv2.morphologyEx(flood_mask, cv2.MORPH_OPEN, kernel)
        flood_mask = cv2.morphologyEx(flood_mask, cv2.MORPH_CLOSE, kernel)

        # Save flood mask
        profile.update(dtype=rasterio.uint8, count=1)
        output_mask = self.output_dir / f"{output_name}_flood_mask.tif"
        with rasterio.open(output_mask, 'w', **profile) as dst:
            dst.write(flood_mask, 1)

        # Generate tiles for CNN training
        # Stack pre, post, change as 3-channel input
        img_stack = np.stack([
            self.normalize_image(img_pre_filtered),
            self.normalize_image(img_post_filtered),
            self.normalize_image(change)
        ], axis=-1)

        tiles = self.generate_tiles(img_stack, mask=flood_mask > 0)

        # Save tiles
        tiles_dir = self.output_dir / f"{output_name}_tiles"
        tiles_dir.mkdir(exist_ok=True)

        for idx, (tile, (row, col)) in enumerate(tiles):
            tile_path = tiles_dir / f"tile_{row}_{col}.npy"
            np.save(tile_path, tile)

        metadata = {
            'output_mask': str(output_mask),
            'tiles_dir': str(tiles_dir),
            'num_tiles': len(tiles),
            'flood_pixels': int(flood_mask.sum()),
            'flood_area_pct': float(flood_mask.mean() * 100)
        }

        # Save metadata
        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Processed S1 flood: {metadata['num_tiles']} tiles, "
                   f"{metadata['flood_area_pct']:.2f}% flooded")

        return metadata

    def process_sentinel2_drought(
        self,
        image_path: str,
        output_name: str,
        scl_path: Optional[str] = None
    ) -> Dict:
        """
        Process Sentinel-2 for drought monitoring

        Args:
            image_path: Path to Sentinel-2 image (multi-band GeoTIFF)
            output_name: Name for output files
            scl_path: Optional path to Scene Classification Layer

        Returns:
            Dictionary with processing metadata
        """
        logger.info(f"Processing S2 drought: {output_name}")

        # Load image (assume bands in order: B2, B3, B4, B8, B11, B12)
        with rasterio.open(image_path) as src:
            bands = src.read().astype(np.float32)
            profile = src.profile

            # Band mapping for Sentinel-2
            blue = bands[0]    # B2
            green = bands[1]   # B3
            red = bands[2]     # B4
            nir = bands[3]     # B8
            swir1 = bands[4]   # B11
            swir2 = bands[5]   # B12

        # Load cloud mask if available
        if scl_path and Path(scl_path).exists():
            with rasterio.open(scl_path) as src_scl:
                scl = src_scl.read(1)
                cloud_mask = self.cloud_mask_s2(scl)
        else:
            cloud_mask = np.ones_like(red, dtype=bool)

        # Calculate indices
        ndvi = self.calculate_ndvi(red, nir)
        ndwi = self.calculate_ndwi(green, nir)

        # Apply cloud mask
        ndvi = np.where(cloud_mask, ndvi, np.nan)
        ndwi = np.where(cloud_mask, ndwi, np.nan)

        # Identify drought stress (low NDVI + low NDWI)
        drought_mask = ((ndvi < 0.3) & (ndwi < 0.2)).astype(np.uint8)

        # Save indices
        profile.update(dtype=rasterio.float32, count=1)

        output_ndvi = self.output_dir / f"{output_name}_ndvi.tif"
        with rasterio.open(output_ndvi, 'w', **profile) as dst:
            dst.write(ndvi.astype(np.float32), 1)

        output_ndwi = self.output_dir / f"{output_name}_ndwi.tif"
        with rasterio.open(output_ndwi, 'w', **profile) as dst:
            dst.write(ndwi.astype(np.float32), 1)

        # Generate tiles (RGB + NDVI + NDWI as 5-channel input)
        img_stack = np.stack([
            self.normalize_image(red),
            self.normalize_image(green),
            self.normalize_image(blue),
            self.normalize_image(ndvi),
            self.normalize_image(ndwi)
        ], axis=-1)

        tiles = self.generate_tiles(img_stack, mask=cloud_mask)

        # Save tiles
        tiles_dir = self.output_dir / f"{output_name}_tiles"
        tiles_dir.mkdir(exist_ok=True)

        for idx, (tile, (row, col)) in enumerate(tiles):
            tile_path = tiles_dir / f"tile_{row}_{col}.npy"
            np.save(tile_path, tile)

        metadata = {
            'output_ndvi': str(output_ndvi),
            'output_ndwi': str(output_ndwi),
            'tiles_dir': str(tiles_dir),
            'num_tiles': len(tiles),
            'drought_pixels': int(drought_mask.sum()),
            'drought_area_pct': float(drought_mask.mean() * 100),
            'mean_ndvi': float(np.nanmean(ndvi)),
            'mean_ndwi': float(np.nanmean(ndwi))
        }

        # Save metadata
        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Processed S2 drought: {metadata['num_tiles']} tiles, "
                   f"mean NDVI={metadata['mean_ndvi']:.3f}")

        return metadata


if __name__ == "__main__":
    processor = SatellitePreprocessor()

    # Example: Process Sentinel-1 flood
    pre_flood = "data/raw/sentinel/s1_pre_flood_VH.tif"
    post_flood = "data/raw/sentinel/s1_post_flood_VH.tif"

    if Path(pre_flood).exists() and Path(post_flood).exists():
        metadata = processor.process_sentinel1_flood(
            pre_flood, post_flood, output_name="2020_monsoon_flood"
        )
        print(f"\nFlood processing complete:")
        print(f"  Tiles: {metadata['num_tiles']}")
        print(f"  Flood area: {metadata['flood_area_pct']:.2f}%")

    # Example: Process Sentinel-2 drought
    s2_image = "data/raw/sentinel/s2_dry_season.tif"

    if Path(s2_image).exists():
        metadata = processor.process_sentinel2_drought(
            s2_image, output_name="2019_dry_season"
        )
        print(f"\nDrought processing complete:")
        print(f"  Tiles: {metadata['num_tiles']}")
        print(f"  Mean NDVI: {metadata['mean_ndvi']:.3f}")
