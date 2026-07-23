"""
CALIPSO L2 Aerosol Profile QC and Spatiotemporal Regridding Pipeline
=====================================================================

"""

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from pyhdf.HDF import HDF
from pyhdf.SD import SD, SDC
from pyhdf.VS import VS
from scipy.interpolate import Rbf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Paths, region bounds, and QC thresholds for a single processing run."""

    data_dir: str = "/glade/derecho/scratch/yingxiao/FROM_CHEYENNE/CALIPSO/US_ALL/"
    output_dir: str = "/glade/derecho/scratch/yingxiao/FROM_CHEYENNE/CALIPSO/US_ALL/"

    # Target region (longitude in [-180, 180]; converted to [0, 360] internally)
    lat_bounds: tuple = (25, 50)
    lon_bounds: tuple = (-125.0, -67.0)

    # Grid resolution (degrees) and averaging window
    grid_resolution: float = 1.0

    # Near-surface layer window (km above local surface height)
    surface_layer_bounds: tuple = (0.5, 1.5)

    # Valid physical ranges for each variable: (min, max)
    valid_ranges: dict = field(default_factory=lambda: {
        "depo": (0.0, 0.6),
        "extinc": (0.0, 1.25),
        "back": (0.0, 0.05),
        "aod": (0.0, 3.0),
    })

    # CALIPSO HDF field names for each variable of interest
    field_names: dict = field(default_factory=lambda: {
        "depo": "Particulate_Depolarization_Ratio_Profile_532",
        "extinc": "Extinction_Coefficient_532",
        "back": "Total_Backscatter_Coefficient_532",
        "aod": "Column_Optical_Depth_Tropospheric_Aerosols_532",
    })


# ---------------------------------------------------------------------------
# Stage 1: Read raw granules
# ---------------------------------------------------------------------------

def read_lidar_altitude(hdf_path: str) -> np.ndarray:
    """Read the fixed lidar altitude grid (metadata) from a CALIPSO granule."""
    hdf = HDF(hdf_path)
    vs = VS(hdf)
    xid = vs.find("metadata")
    altitude_field = vs.attach(xid)
    altitude_field.setfields("Lidar_Data_Altitudes")
    nrecs, *_ = altitude_field.inquire()
    altitude = altitude_field.read(nRec=nrecs)
    altitude_field.detach()
    # Altitudes are stored top-to-bottom; flip to bottom-to-top for indexing by height.
    return np.array(altitude[0][0])[::-1]


def read_granule(file_path: str) -> dict:
    """Read one CALIPSO L2 granule and return the raw variables needed downstream."""
    sd = SD(file_path, SDC.READ)

    lat = sd.select("Latitude")[:, 0]
    lon = sd.select("Longitude")[:, 0]
    surface = sd.select("Surface_Top_Altitude_532")[:]

    depo = np.rot90(sd.select("Particulate_Depolarization_Ratio_Profile_532")[:], 1)
    extinc = np.rot90(sd.select("Extinction_Coefficient_532")[:], 1)
    back = np.rot90(sd.select("Total_Backscatter_Coefficient_532")[:], 1)
    aod = np.rot90(sd.select("Column_Optical_Depth_Tropospheric_Aerosols_532")[:], 1)

    cad_score = np.rot90(sd.select("CAD_Score")[:, :, 0], 1)
    extinction_qc = np.rot90(sd.select("Extinction_QC_Flag_532")[:, :, 0], 1)
    depo_uncertainty = np.rot90(sd.select("Particulate_Depolarization_Ratio_Uncertainty_532")[:], 1)

    profile_time = sd.select("Profile_UTC_Time")[0]
    day_night_flag = sd.select("Day_Night_Flag")[0]

    return {
        "lat": lat,
        "lon": lon,
        "surface": surface,
        "depo": depo,
        "extinc": extinc,
        "back": back,
        "aod": aod,
        "cad_score": cad_score,
        "extinction_qc": extinction_qc,
        "depo_uncertainty": depo_uncertainty,
        "profile_time": profile_time,
        "day_night_flag": day_night_flag,
    }


def read_all_granules(data_dir: str) -> list:
    """Read every granule in a directory, in alphabetical (i.e. time) order."""
    filenames = sorted(os.listdir(data_dir))
    granules = []
    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        granules.append(read_granule(file_path))
    return granules


def granule_date(granule: dict) -> pd.Timestamp:
    """Convert a granule's CALIPSO profile time (YYMMDD.frac) to a calendar date."""
    yymmdd = int(granule["profile_time"][0]) + 20_000_000
    return pd.to_datetime(yymmdd, format="%Y%m%d")


# ---------------------------------------------------------------------------
# Stage 2: Quality control
# ---------------------------------------------------------------------------

def apply_quality_control(granule: dict, valid_ranges: dict) -> dict:
    """Screen a granule's aerosol variables using CAD score, QC flags, and
    physically valid ranges. Values that fail any check are set to NaN.
    """
    depo, extinc, back, aod = granule["depo"], granule["extinc"], granule["back"], granule["aod"]

    # CAD score: retain only high-confidence aerosol classifications.
    low_confidence = granule["cad_score"] >= -70
    depo[low_confidence] = np.nan
    extinc[low_confidence] = np.nan
    back[low_confidence] = np.nan

    # Extinction QC flag: keep only flags 0 ("good") and 1 ("adjusted, good").
    bad_qc = np.logical_and(granule["extinction_qc"] != 0.0, granule["extinction_qc"] != 1.0)
    depo[bad_qc] = np.nan
    extinc[bad_qc] = np.nan
    back[bad_qc] = np.nan

    # Depolarization ratio uncertainty: drop physically implausible uncertainty values.
    bad_uncertainty = np.logical_or(granule["depo_uncertainty"] >= 1.0, granule["depo_uncertainty"] < 0.0)
    depo[bad_uncertainty] = np.nan

    # Fill value and physical-range screening, applied per variable.
    for arr, (lo, hi) in zip(
        (depo, extinc, back, aod),
        (valid_ranges["depo"], valid_ranges["extinc"], valid_ranges["back"], valid_ranges["aod"]),
    ):
        arr[arr == -9999] = np.nan
        out_of_range = np.logical_or(arr < lo, arr > hi)
        arr[out_of_range] = np.nan

    granule["depo"] = np.ma.masked_invalid(depo)
    granule["extinc"] = np.ma.masked_invalid(extinc)
    granule["back"] = np.ma.masked_invalid(back)
    granule["aod"] = np.ma.masked_invalid(aod)
    return granule


# ---------------------------------------------------------------------------
# Stage 3: Near-surface layer extraction
# ---------------------------------------------------------------------------

def extract_near_surface_layer(granule: dict, height: np.ndarray, layer_bounds: tuple) -> dict:
    """Mask each profile to the near-surface layer (height - surface within
    layer_bounds, in km) and average the remaining vertical levels.
    """
    lo, hi = layer_bounds
    height_col = height[:, np.newaxis]
    surface_row = granule["surface"][:, 0][np.newaxis, :]
    height_above_surface = height_col - surface_row
    layer_mask = (height_above_surface >= lo) & (height_above_surface <= hi)

    for var in ("depo", "extinc", "back", "aod"):
        granule[var] = np.where(layer_mask, granule[var], np.nan)

    granule["depo_avg"] = np.nanmean(granule["depo"], axis=0)
    granule["extinc_avg"] = np.nanmean(granule["extinc"], axis=0)
    granule["back_avg"] = np.nanmean(granule["back"], axis=0)
    granule["aod_avg"] = np.nanmean(granule["aod"], axis=0)
    return granule


# ---------------------------------------------------------------------------
# Stage 4: Region subsetting and long-format table construction
# ---------------------------------------------------------------------------

def subset_region(granule: dict, date: pd.Timestamp, lat_bounds: tuple, lon_bounds: tuple) -> pd.DataFrame:
    """Subset one granule's near-surface averages to the target region and
    return a long-format DataFrame of (lat, lon, date, variable values).
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    in_region = np.where(
        (granule["lat"] >= lat_min) & (granule["lat"] <= lat_max)
        & (granule["lon"] >= lon_min) & (granule["lon"] <= lon_max)
    )[0]

    if in_region.size == 0:
        return pd.DataFrame(columns=["latitude", "longitude", "date", "depo", "extinc", "back", "aod"])

    return pd.DataFrame({
        "latitude": granule["lat"][in_region],
        "longitude": granule["lon"][in_region],
        "date": date,
        "depo": granule["depo_avg"][in_region],
        "extinc": granule["extinc_avg"][in_region],
        "back": granule["back_avg"][in_region],
        "aod": granule["aod_avg"][in_region],
    })


def build_long_table(granules: list, dates: list, cfg: Config) -> pd.DataFrame:
    """Subset every granule to the target region and concatenate into one table."""
    frames = [
        subset_region(g, d, cfg.lat_bounds, cfg.lon_bounds)
        for g, d in zip(granules, dates)
    ]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["longitude"] = df["longitude"] % 360
    df["year"] = df["date"].dt.isocalendar().year
    df["week"] = df["date"].dt.isocalendar().week
    df["bi_week"] = ((df["week"] - 1) // 2) + 1
    return df


# ---------------------------------------------------------------------------
# Stage 5-6: Gridding and gap-filling
# ---------------------------------------------------------------------------

def build_grid_bins(cfg: Config) -> tuple:
    """Construct the target lat/lon bin edges for the region and resolution."""
    lat_min, lat_max = cfg.lat_bounds
    lon_min, lon_max = cfg.lon_bounds
    lat_bins = np.arange(lat_min, lat_max + cfg.grid_resolution, cfg.grid_resolution)
    lon_bins = np.arange(lon_min % 360, (lon_max % 360) + cfg.grid_resolution, cfg.grid_resolution)
    return lat_bins, lon_bins


def grid_and_interpolate(df: pd.DataFrame, cfg: Config) -> xr.Dataset:
    """Bin observations onto the target grid at bi-weekly resolution, then
    fill spatial gaps first with RBF interpolation and then, for any
    remaining gaps, with linear interpolation plus forward/backward fill.
    """
    variables = ["depo", "extinc", "back", "aod"]
    lat_bins, lon_bins = build_grid_bins(cfg)

    df = df.copy()
    df["lat_bin"] = np.floor(df["latitude"] / cfg.grid_resolution) * cfg.grid_resolution
    df["lon_bin"] = np.floor(df["longitude"] / cfg.grid_resolution) * cfg.grid_resolution

    bi_weekly_avg = df.groupby(["year", "bi_week", "lat_bin", "lon_bin"]).mean(numeric_only=True).reset_index()
    grid_lat, grid_lon = np.meshgrid(lat_bins, lon_bins, indexing="ij")

    interpolated = {}
    for (year, bi_week), group in bi_weekly_avg.groupby(["year", "bi_week"]):
        for var in variables:
            valid = group[["lat_bin", "lon_bin", var]].dropna()
            if valid.empty:
                continue
            rbf = Rbf(valid["lat_bin"], valid["lon_bin"], valid[var], function="linear", smooth=0)
            interpolated[(year, bi_week, var)] = rbf(grid_lat, grid_lon)

    bi_week_keys = list(bi_weekly_avg.groupby(["year", "bi_week"]).groups.keys())
    data_vars = {}
    for var in variables:
        grids = [interpolated[key + (var,)] for key in bi_week_keys if key + (var,) in interpolated]
        if grids:
            data_vars[var] = np.stack(grids, axis=0)

    ds = xr.Dataset(
        {var: (["bi_week", "lat_bin", "lon_bin"], data_vars[var]) for var in data_vars},
        coords={"bi_week": range(len(bi_week_keys)), "lat_bin": lat_bins, "lon_bin": lon_bins},
    )

    # Fill any remaining gaps (e.g. bi-weeks/cells the RBF fit couldn't reach)
    # with linear interpolation, then forward/backward fill.
    df_gridded = ds.to_dataframe().reset_index()
    for var in variables:
        if var in df_gridded:
            df_gridded[var] = (
                df_gridded[var]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )

    return df_gridded.set_index(["bi_week", "lat_bin", "lon_bin"]).to_xarray()


# ---------------------------------------------------------------------------
# Stage 7: Save outputs
# ---------------------------------------------------------------------------

def save_outputs(ds: xr.Dataset, cfg: Config, name: str = "bi_weekly_regridded_CALIPSO") -> None:
    """Save the gridded, gap-filled dataset to both NetCDF and Excel."""
    ds.to_dataframe().reset_index().to_excel(
        os.path.join(cfg.output_dir, f"{name}.xlsx"), index=False
    )
    ds.to_netcdf(os.path.join(cfg.output_dir, f"{name}.nc"))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config) -> xr.Dataset:
    granules = read_all_granules(cfg.data_dir)
    dates = [granule_date(g) for g in granules]

    # Lidar altitude grid is fixed across granules; read once from the first file.
    first_file = os.path.join(cfg.data_dir, sorted(os.listdir(cfg.data_dir))[0])
    height = read_lidar_altitude(first_file)

    granules = [apply_quality_control(g, cfg.valid_ranges) for g in granules]
    granules = [extract_near_surface_layer(g, height, cfg.surface_layer_bounds) for g in granules]

    long_table = build_long_table(granules, dates, cfg)
    gridded = grid_and_interpolate(long_table, cfg)
    save_outputs(gridded, cfg)
    return gridded


if __name__ == "__main__":
    run_pipeline(Config())
