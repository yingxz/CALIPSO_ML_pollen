#Pollen Prediction from CALIPSO / MPL Lidar Optical Properties

This contains two set of script:

calipso_regrid_pipeline.py
---------------

Reads CALIPSO L2 05km Aerosol Profile (Standard-V4) HDF files, applies
quality-control screening, extracts the near-surface aerosol layer,
subsets to a target region, and produces a gap-filled, bi-weekly
1-degree gridded product (saved to NetCDF and Excel).

Pipeline stages
1. Read raw HDF granules and extract variables of interest.
2. Apply quality control: CAD score, extinction/depolarization QC flags,
   and physically valid-range checks.
3. Extract the near-surface layer (0.5-1.5 km above local surface height)
   and average over that layer.
4. Subset to the target region and flatten across all granules into a
   single long-format table.
5. Bin onto a regular lat/lon grid at bi-weekly resolution.
6. Fill spatial gaps with RBF interpolation, then fill any remaining
   gaps (e.g. edge cells) with linear interpolation + forward/backward
   fill.
7. Save the gridded product to NetCDF and Excel.

   

ML_pollen_CALIPSO_cleaned_v2.ipynb
---------------
Goal: Predict ground-observed pollen concentrations (by species) using satellite/ground-based lidar-derived aerosol optical properties (CALIPSO, MPLNET) combined with meteorological and land-cover covariates, via a Random Forest regression model.

Structure
Imports
Station metadata (lat/lon for each monitoring site)
Load per-site predictor (lidar) and target (pollen) data — generalized loop, not one block per city
Load and attach meteorological/land-cover covariates (CDD, day length, land-cover fractions)
Combine all sites into one training table
Train/test split
Train Random Forest
Evaluate + feature importance
Per-species performance metrics

