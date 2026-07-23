#Pollen Prediction from CALIPSO / MPL Lidar Optical Properties

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
