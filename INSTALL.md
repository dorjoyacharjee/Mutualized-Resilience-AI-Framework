# Installation Guide

Complete setup instructions for the Climate-Adaptive Agricultural Insurance AI Framework.

---

## System Requirements

### Hardware
- **CPU:** Multi-core processor (8+ cores recommended)
- **RAM:** 32GB minimum, 64GB recommended
- **Storage:** 150GB free space (100GB for data, 50GB for models/outputs)
- **GPU:** NVIDIA GPU with 8GB+ VRAM (for CNN training)
  - CUDA 11.8 or higher
  - cuDNN 8.6 or higher

### Software
- **Operating System:** Linux (Ubuntu 20.04+), macOS (12+), Windows 10/11
- **Python:** 3.8, 3.9, or 3.10 (3.11+ may have compatibility issues)
- **Git:** For cloning repository

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/climate-agricultural-insurance-ai.git
cd climate-agricultural-insurance-ai
```

#### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

#### 4. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

#### 5. Install Package

```bash
pip install -e .
```

This installs the package in editable mode, allowing you to modify code and see changes immediately.

---

### Method 2: Conda Installation

If you prefer Conda/Anaconda:

```bash
# Create environment
conda create -n agri-insurance python=3.10
conda activate agri-insurance

# Install dependencies
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn
conda install -c conda-forge xarray netCDF4 rasterio geopandas
pip install tensorflow==2.13.0
pip install xgboost lightgbm

# Install remaining from requirements
pip install -r requirements.txt

# Install package
pip install -e .
```

---

### Method 3: Docker Installation

For containerized deployment:

```bash
# Build Docker image
docker build -t agri-insurance:latest .

# Run container
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs agri-insurance:latest
```

---

## API Key Configuration

Several data sources require API keys or credentials.

### 1. Copernicus Climate Data Store (CDS)

For CMIP6 and ERA5 data:

**Step 1:** Create account at https://cds.climate.copernicus.eu/

**Step 2:** Get API key from https://cds.climate.copernicus.eu/user

**Step 3:** Configure credentials

**Linux/macOS:**
```bash
echo "url: https://cds.climate.copernicus.eu/api/v2" > ~/.cdsapirc
echo "key: YOUR_UID:YOUR_API_KEY" >> ~/.cdsapirc
```

**Windows:**
Create file `%USERPROFILE%\.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

### 2. Google Earth Engine

For Sentinel satellite data:

**Step 1:** Sign up at https://earthengine.google.com/

**Step 2:** Authenticate

```bash
earthengine authenticate
```

Follow the browser authentication flow.

**Step 3 (Optional):** Service Account

For automated access:
1. Create service account in Google Cloud Console
2. Download JSON key file
3. Set environment variable:
```bash
export GEE_SERVICE_ACCOUNT="your-account@project.iam.gserviceaccount.com"
export GEE_PRIVATE_KEY_PATH="/path/to/service-account-key.json"
```

### 3. Environment Variables

Create `.env` file in project root:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# Copernicus CDS
CDS_API_KEY=your_cds_key
CDS_API_URL=https://cds.climate.copernicus.eu/api/v2

# Google Earth Engine
GEE_SERVICE_ACCOUNT=your-account@project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_PATH=/path/to/key.json

# AWS (for Sentinel data on S3, optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-west-2
```

---

## GPU Setup (Optional but Recommended)

For TensorFlow GPU support:

### CUDA Installation

**Ubuntu:**
```bash
# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-11-8

# Install cuDNN
# Download from https://developer.nvidia.com/cudnn
sudo dpkg -i cudnn-local-repo-*.deb
```

### Verify GPU

```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

---

## Verify Installation

Run verification script:

```bash
python -c "
import sys
import tensorflow as tf
import xarray as xr
import rasterio
import earthengine as ee

print('Python:', sys.version)
print('TensorFlow:', tf.__version__)
print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)
print('xarray:', xr.__version__)
print('rasterio:', rasterio.__version__)
print('Installation successful!')
"
```

Expected output:
```
Python: 3.10.x
TensorFlow: 2.13.0
GPU Available: True
xarray: 2023.7.0
rasterio: 1.3.8
Installation successful!
```

---

## Directory Structure Setup

Create necessary directories:

```bash
mkdir -p data/{raw,processed,external,interim}
mkdir -p data/raw/{cmip6,era5,sentinel,faostat}
mkdir -p data/external/{bangladesh,cooperatives}
mkdir -p outputs/{models,figures,reports}
mkdir -p logs
```

Or use the automated script:

```bash
python scripts/setup_directories.py
```

---

## Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Fails

**Solution:** Install TensorFlow separately first:
```bash
pip install tensorflow==2.13.0
```

If GPU version needed:
```bash
pip install tensorflow[and-cuda]==2.13.0
```

#### 2. NetCDF4 or rasterio Installation Error

**Linux:**
```bash
sudo apt-get install libhdf5-dev libnetcdf-dev libgdal-dev
pip install netCDF4 rasterio
```

**macOS:**
```bash
brew install netcdf gdal
pip install netCDF4 rasterio
```

**Windows:** Use conda:
```bash
conda install -c conda-forge netcdf4 rasterio
```

#### 3. Memory Errors During Data Download

Increase available memory or download data in smaller chunks. Edit `config.yaml`:
```yaml
data_sources:
  cmip6:
    temporal_range: ["2025-01-01", "2030-12-31"]  # Shorter range
```

#### 4. CDS API Authentication Error

Check your `~/.cdsapirc` file format. Ensure no extra spaces:
```
url: https://cds.climate.copernicus.eu/api/v2
key: 12345:abcdef-1234-5678-90ab-cdefghijklmn
```

#### 5. Google Earth Engine Timeout

Increase timeout in `config.yaml`:
```yaml
preprocessing:
  satellite:
    gee_timeout: 600  # seconds
```

---

## Next Steps

After successful installation:

1. **Configure settings:** Edit `config.yaml` to match your environment
2. **Test data acquisition:** Run `python src/data_acquisition/cmip6_downloader.py`
3. **Run full pipeline:** `python main_pipeline.py --stage full`
4. **Explore notebooks:** `jupyter notebook notebooks/01_data_exploration.ipynb`

---

## Getting Help

- **Documentation:** See `docs/` directory
- **Issues:** https://github.com/yourusername/climate-agricultural-insurance-ai/issues
- **Discussions:** https://github.com/yourusername/climate-agricultural-insurance-ai/discussions
- **Email:** your-email@institution.edu

---

## Uninstallation

To remove the package:

```bash
# Deactivate environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove cached files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

To keep code but remove downloaded data:
```bash
rm -rf data/raw/*
rm -rf data/processed/*
```
