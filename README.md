# Climate-Adaptive Agricultural Insurance AI Framework

Implementation of the integrated AI framework for agricultural insurance in climate-vulnerable regions, combining Meta-Risk Evaluation and Prediction (MREP) with a dual-module AI architecture (HAM + SFM).

## Overview

This repository contains the complete implementation of the research described in:
**"From Meta-Risk to Mutualized Resilience: An Integrated AI Framework for Climate-Adaptive Agricultural Insurance"**

### Key Components:
1. **MREP Model** - Predicts insurer retreat using LSTM + XGBoost
2. **HAM (Hazard Assessment Module)** - U-Net CNN for satellite-based damage detection
3. **SFM (Solvency Forecasting Module)** - Long-term fund viability prediction

## Data Sources

All data is from public repositories (see `docs/DATA_SOURCES.md`):
- Climate: CMIP6, ERA5 (Copernicus CDS)
- Satellite: Sentinel-1/2 (ESA/AWS)
- Agriculture: FAOSTAT, Bangladesh BBS
- Disaster: Munich Re, Swiss Re reports

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/climate-agricultural-insurance-ai.git
cd climate-agricultural-insurance-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your CDS API key, GEE credentials, etc.

Requirements
Python 3.8+
TensorFlow 2.13+
PyTorch 2.0+ (for U-Net)
XGBoost 1.7+
Google Earth Engine API
CDS API (Copernicus)
32GB+ RAM recommended
GPU with 8GB+ VRAM (CUDA 11.8+)
