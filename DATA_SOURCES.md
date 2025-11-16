# Data Sources Documentation

Complete list of all data sources used in the Climate-Adaptive Agricultural Insurance AI Framework, with access instructions, licensing, and citation requirements.

---

## Climate Data

### 1. CMIP6 Climate Model Projections

**Description:** Global gridded climate projections from coupled model intercomparison project.

**Repository:** Copernicus Climate Data Store (CDS)  
**URL:** https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6  
**DOI:** See individual model DOIs at https://wcrp-cmip.org/cmip6/

**Access:**
- Create free account at CDS
- Obtain API key from account settings
- Install cdsapi: `pip install cdsapi`
- Configure credentials in `~/.cdsapirc`

**Variables Used:**
- tas (near-surface air temperature)
- pr (precipitation)
- tasmax (maximum temperature)
- tasmin (minimum temperature)

**Scenarios:** SSP3-7.0 (high emissions)  
**Temporal Coverage:** 2025-2100  
**Spatial Coverage:** Bangladesh (20-27°N, 87-93°E)  
**Resolution:** 0.25° (~25km)

**License:** Copernicus License - Free for research, attribution required  
**Citation:**
```
WCRP (2024). CMIP6 Climate Projections. World Climate Research Programme. 
Available at: https://wcrp-cmip.org/cmip6/
```

**File Format:** NetCDF4  
**Approximate Size:** 5-10 GB per variable per scenario

---

### 2. ERA5 Reanalysis

**Description:** Hourly global climate reanalysis for historical baseline.

**Repository:** ECMWF Copernicus Climate Data Store  
**URL:** https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

**Access:** Same as CMIP6 (CDS account + API key)

**Variables Used:**
- 2m_temperature
- total_precipitation
- 10m_u_wind_component
- 10m_v_wind_component

**Temporal Coverage:** 1979-2023  
**Spatial Coverage:** Bangladesh (20-27°N, 87-93°E)  
**Resolution:** 0.25° (~25km)

**License:** Copernicus License  
**Citation:**
```
Hersbach, H., et al. (2020). The ERA5 global reanalysis. 
Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
DOI: 10.1002/qj.3803
```

**File Format:** NetCDF4  
**Approximate Size:** 8-12 GB for full time series

---

## Satellite Imagery

### 3. Sentinel-1 SAR

**Description:** C-band synthetic aperture radar for all-weather flood detection.

**Repository:** ESA Copernicus Open Access Hub / AWS Registry of Open Data  
**URL:** 
- https://scihub.copernicus.eu/
- https://registry.opendata.aws/sentinel-1/

**Access:**
- Via Google Earth Engine (recommended): https://earthengine.google.com/
- Direct download via Copernicus Hub (requires registration)
- AWS S3: `s3://sentinel-s1-l1c/` (requester-pays)

**Product Type:** GRD (Ground Range Detected)  
**Polarization:** VV, VH  
**Orbit:** Descending  
**Resolution:** 10 meters

**Temporal Coverage:** 2014-present (6-12 day revisit)  
**Spatial Coverage:** Global

**License:** ESA Copernicus Open Access  
**Citation:**
```
European Space Agency (ESA). Sentinel-1 SAR User Guide. 
Available at: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar
```

**File Format:** GeoTIFF (processed), SAFE (original)  
**Approximate Size:** 500 MB - 1 GB per scene

---

### 4. Sentinel-2 Optical

**Description:** Multispectral optical imagery for vegetation monitoring.

**Repository:** ESA Copernicus / AWS Registry of Open Data  
**URL:**
- https://scihub.copernicus.eu/
- https://registry.opendata.aws/sentinel-2/

**Access:** Same as Sentinel-1

**Product Level:** L2A (Surface Reflectance)  
**Bands Used:** B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)  
**Resolution:** 10-20 meters

**Cloud Cover Threshold:** <20%  
**Temporal Coverage:** 2015-present (5-day revisit)

**License:** ESA Copernicus Open Access  
**Citation:**
```
European Space Agency (ESA). Sentinel-2 User Handbook. 
Available at: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi
```

**File Format:** GeoTIFF / JP2  
**Approximate Size:** 600 MB per scene (L2A)

---

## Agricultural Statistics

### 5. FAOSTAT

**Description:** Global agricultural production, yield, and price statistics.

**Repository:** Food and Agriculture Organization (FAO)  
**URL:** https://www.fao.org/faostat/en/#data

**Access:**
- Bulk download: https://fenixservices.fao.org/faostat/static/bulkdownloads
- API: Available for programmatic access
- No registration required

**Domains Used:**
- QCL: Crops and livestock products (production, yield, area)
- PP: Producer prices
- FBS: Food balance sheets

**Countries:** Bangladesh, India  
**Temporal Coverage:** 1961-2023  
**Update Frequency:** Annual

**License:** CC BY-NC-SA 3.0 IGO  
**Citation:**
```
FAO (2024). FAOSTAT Statistical Database. Food and Agriculture Organization of the United Nations. 
Available at: https://www.fao.org/faostat/
```

**File Format:** CSV (zipped)  
**Approximate Size:** 50-200 MB per domain

---

### 6. Bangladesh Bureau of Statistics (BBS)

**Description:** National agricultural census and economic indicators.

**Repository:** Bangladesh National Statistics Portal  
**URL:**
- Agriculture domain: http://nsds.bbs.gov.bd/en/domains/1/Agriculture
- Open data portal: http://data.gov.bd/

**Access:**
- Manual download from BBS website
- Agricultural Census 2008: http://data.gov.bd/dataset/agriculture-census-2008
- Registration may be required for some datasets

**Data Available:**
- Farm household counts by district
- Cultivated area by crop type
- Mechanization rates
- Irrigation coverage

**Temporal Coverage:** Census years (1977, 1983, 1996, 2008)  
**Spatial Coverage:** District/Upazila level

**License:** Bangladesh Government Open Data License  
**Citation:**
```
Bangladesh Bureau of Statistics (2010). Agricultural Census 2008: National Report. 
Statistics and Informatics Division, Ministry of Planning, Government of Bangladesh.
```

**File Format:** PDF, CSV  
**Approximate Size:** 10-50 MB

---

## Disaster and Insurance Data

### 7. Munich Re NatCatSERVICE

**Description:** Global natural catastrophe loss database.

**Repository:** Munich Reinsurance Company  
**URL:** https://www.munichre.com/en/solutions/for-industry-clients/natcatservice.html

**Access:**
- Summary statistics publicly available
- Full database access requires registration and approval
- Academic research access may be granted upon request

**Data Available:**
- Disaster event dates and locations
- Economic losses
- Insured losses
- Fatalities

**Temporal Coverage:** 1980-present  
**Update Frequency:** Continuous

**License:** Proprietary (summary stats free, full database restricted)  
**Citation:**
```
Munich Re (2024). NatCatSERVICE Database. 
Available at: https://www.munichre.com/natcatservice
```

**File Format:** Excel, PDF reports  
**Note:** For this research, publicly available summary statistics were used

---

### 8. Swiss Re Sigma

**Description:** Insurance industry reports and disaster impact data.

**Repository:** Swiss Re Institute  
**URL:** https://www.swissre.com/institute/research/sigma-research.html

**Access:**
- Public reports freely available
- Registration required for full reports

**Data Available:**
- Annual insurance market statistics
- Natural disaster economic/insured losses
- Regional insurance penetration rates

**Temporal Coverage:** 1970-present  
**Update Frequency:** Annual

**License:** Swiss Re proprietary (reports free for research)  
**Citation:**
```
Swiss Re (2024). Sigma Reports. Swiss Re Institute. 
Available at: https://www.swissre.com/institute/
```

**File Format:** PDF  
**Approximate Size:** 5-15 MB per report

---

### 9. World Bank Open Data

**Description:** Economic indicators and agricultural insurance policy reports.

**Repository:** World Bank Open Knowledge Repository  
**URL:** https://openknowledge.worldbank.org/

**Specific Resources:**
- Agricultural Insurance in Bangladesh: https://openknowledge.worldbank.org/handle/10986/12624
- Policy Options for Agriculture Insurance: https://openknowledge.worldbank.org/handle/10986/30691

**Access:** Free, no registration required

**License:** Creative Commons Attribution 3.0 IGO (CC BY 3.0 IGO)  
**Citation:**
```
World Bank (2010). Agricultural Insurance in Bangladesh: Market Perspectives and Policy Analysis. 
Washington, DC: World Bank. Available at: https://openknowledge.worldbank.org/handle/10986/12624
```

**File Format:** PDF, CSV  
**Approximate Size:** 2-10 MB per report

---

## Data Statement for Journal Submission

All research data used in this study are sourced from public, government, and institutional repositories listed above. The authors do not own or control the raw data or its access policies. Each data source has specific licensing terms that must be followed:

1. **Climate data (CMIP6, ERA5):** Copernicus License - Free for research with attribution
2. **Satellite data (Sentinel-1/2):** ESA Copernicus Open Access - Free with attribution
3. **Agricultural statistics (FAOSTAT):** CC BY-NC-SA 3.0 IGO - Non-commercial with attribution
4. **National statistics (BBS):** Bangladesh Government Open Data - Attribution required
5. **Disaster data (Munich Re, Swiss Re):** Publicly available summaries used

For cross-referencing, persistent repository links are provided for each dataset. Where datasets require registration or manual download, explicit instructions are documented.

---

## Data Availability Statement (for Paper)

**Suggested text for manuscript:**

> All data used in this study are publicly available from third-party repositories. Climate projections (CMIP6) and reanalysis (ERA5) can be accessed from the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/). Satellite imagery (Sentinel-1/2) is available via Google Earth Engine (https://earthengine.google.com/) and AWS Registry of Open Data. Agricultural statistics are from FAOSTAT (https://www.fao.org/faostat/) and Bangladesh Bureau of Statistics (http://bbs.gov.bd/). Disaster loss data are from publicly available Munich Re and Swiss Re reports. Complete data access instructions and code for reproduction are available at [GitHub repository link].

---

## Contact for Data Questions

For questions about data acquisition or preprocessing:
- CMIP6/ERA5: cds.support@copernicus-climate.eu
- Sentinel: https://sentinel.esa.int/web/sentinel/user-guides
- FAOSTAT: FAO-statistics@fao.org
- BBS: info@bbs.gov.bd

For questions about this research implementation:
- dorjoyacharjee@gmail.com
