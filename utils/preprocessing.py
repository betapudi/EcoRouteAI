import os
import math
import json
import joblib
import numpy as np
import pandas as pd  # ← ADD THIS IMPORT
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from itertools import product
from scipy.spatial import cKDTree
import rasterio
from rasterio.errors import RasterioIOError
import lightgbm as lgb
from tqdm import tqdm
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
IDW_NEIGHBORS = 5
IDW_POWER = 2.0
HOUR_TOL = 1
MATCH_BY_WEEKDAY = False
np.random.seed(42)
tqdm.pandas()

# -----------------------------
# UTILITIES
# -----------------------------
def _open_raster(path):
    try:
        return rasterio.open(path)
    except RasterioIOError as e:
        print(f"⚠ Could not open raster: {path} ({e})")
        return None

def sample_rasters(lat, lon, ndvi_src, elev_src):
    ndvi_val, elev_val = np.nan, np.nan
    xy = [(lon, lat)]
    if ndvi_src:
        try:
            arr = list(ndvi_src.sample(xy))[0]
            ndvi_val = float(arr[0])
            if ndvi_val > 1.0:
                ndvi_val /= 10000.0
        except:
            ndvi_val = np.nan
    if elev_src:
        try:
            arr = list(elev_src.sample(xy))[0]
            elev_val = float(arr[0])
        except:
            elev_val = np.nan
    if not math.isnan(ndvi_val):
        ndvi_val = np.clip(ndvi_val, 0.0, 1.0)
    return ndvi_val, elev_val

def _subset_for_time(df, hour, weekday, month):
    hmin, hmax = max(0, hour - HOUR_TOL), min(23, hour + HOUR_TOL)
    cond = (df["month"] == month) & (df["hour"].between(hmin, hmax))
    if MATCH_BY_WEEKDAY:
        cond &= (df["weekday"] == weekday)
    return df.loc[cond].copy()

def idw_interpolate(df_air, lat, lon, hour, weekday, month, variables, k=IDW_NEIGHBORS, power=IDW_POWER):
    subset = _subset_for_time(df_air, hour, weekday, month)
    if subset.empty:
        return {f"{v}_idw": np.nan for v in variables}
    
    coords = subset[["latitude", "longitude"]].to_numpy()
    tree = cKDTree(coords)
    k_use = int(min(k, len(subset)))
    
    try:
        dist, idx = tree.query([[lat, lon]], k=k_use)
        if k_use == 1:
            d = np.array([dist]).astype(float)
            ii = np.array([idx]).astype(int)
        else:
            d = dist[0].astype(float)
            ii = idx[0].astype(int)
        
        w = 1.0 / (np.power(d, power) + 1e-6)
        out = {}
        for v in variables:
            vals = subset.iloc[ii][v].to_numpy(dtype=float)
            out[f"{v}_idw"] = float(np.dot(w, vals) / w.sum())
        return out
    except:
        return {f"{v}_idw": np.nan for v in variables}

def normalize(series):
    series_min = series.min()
    series_max = series.max()
    if series_max - series_min == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series_min) / (series_max - series_min)

# -----------------------------
# PREPROCESSING FUNCTION
# -----------------------------
def preprocess_live_data(city_name, live_air_df, ndvi_tif, elev_tif, eco_sites_df,                          
                          aggregate_hours=False):
    # Check if air data is provided
    if live_air_df is None:
        raise ValueError("Air quality data must be provided.")

    # Filter city if column exists
    if "city" in live_air_df.columns:
        air_df = live_air_df[live_air_df["city"] == city_name].copy()
    else:
        air_df = live_air_df.copy()

    # Parse time
    if "local_time" in air_df.columns:
        air_df["local_time"] = pd.to_datetime(air_df["local_time"], dayfirst=True, errors="coerce")
        air_df = air_df.dropna(subset=["local_time"]).copy()
    else:
        # Add current time if not available
        air_df["local_time"] = pd.Timestamp.now()

    pollutants = ["PM2_5", "PM10", "CO2", "AT", "RH"]

    if aggregate_hours:
        # Average values over all available hours before interpolation
        air_df = (
            air_df.groupby(["station_name", "latitude", "longitude"], as_index=False)[pollutants]
            .mean()
        )
    
        # Use latest available record time
        if "local_time" in live_air_df.columns:
            latest_time = live_air_df["local_time"].max()
        else:
            latest_time = pd.Timestamp.now()
            
        # Ensure it's a Timestamp
        latest_time = pd.to_datetime(latest_time)
        
        air_df["hour"] = latest_time.hour
        air_df["weekday"] = latest_time.weekday()
        air_df["month"] = latest_time.month
    
        # Cyclical encodings
        air_df["hour_sin"] = np.sin(2 * np.pi * air_df["hour"] / 24)
        air_df["hour_cos"] = np.cos(2 * np.pi * air_df["hour"] / 24)
        air_df["month_sin"] = np.sin(2 * np.pi * air_df["month"] / 12)
        air_df["month_cos"] = np.cos(2 * np.pi * air_df["month"] / 12)
        air_df["weekday_sin"] = np.sin(2 * np.pi * air_df["weekday"] / 7)
        air_df["weekday_cos"] = np.cos(2 * np.pi * air_df["weekday"] / 7)
    
        # Drop local_time to avoid accidental filtering
        if "local_time" in air_df.columns:
            air_df = air_df.drop(columns=["local_time"])
    else:
        # Per-hour prediction mode
        air_df["hour"] = air_df["local_time"].dt.hour
        air_df["weekday"] = air_df["local_time"].dt.weekday
        air_df["month"] = air_df["local_time"].dt.month
        
        # Add cyclical encodings
        air_df["hour_sin"] = np.sin(2 * np.pi * air_df["hour"] / 24)
        air_df["hour_cos"] = np.cos(2 * np.pi * air_df["hour"] / 24)
        air_df["month_sin"] = np.sin(2 * np.pi * air_df["month"] / 12)
        air_df["month_cos"] = np.cos(2 * np.pi * air_df["month"] / 12)
        air_df["weekday_sin"] = np.sin(2 * np.pi * air_df["weekday"] / 7)
        air_df["weekday_cos"] = np.cos(2 * np.pi * air_df["weekday"] / 7)
        
    try:
        # Try load tiff files
        ndvi_src = _open_raster(ndvi_tif)
        elev_src = _open_raster(elev_tif)
    except FileNotFoundError:
        st.warning("Elevation or NDVI data not available. Using default values.")
    # Prepare eco sites data
    if aggregate_hours or live_air_df is not None:
        eco_df = eco_sites_df.copy()
        eco_df["hour"] = air_df["hour"].iloc[0] if len(air_df) > 0 else 12
        eco_df["weekday"] = air_df["weekday"].iloc[0] if len(air_df) > 0 else 0
        eco_df["month"] = air_df["month"].iloc[0] if len(air_df) > 0 else 1
    else:
        # Synthetic time expansion for training
        hours = range(6, 19)
        weekdays = range(0, 7)
        months = range(1, 13)
        synthetic_rows = []
        for _, r in eco_sites_df.iterrows():
            for h, wd, m in product(hours, weekdays, months):
                synthetic_rows.append({
                    "site_name": r["site_name"],
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "hour": h,
                    "weekday": wd,
                    "month": m
                })
        eco_df = pd.DataFrame(synthetic_rows)
        
    eco_df = gpd.GeoDataFrame(eco_df, geometry="geometry", crs="EPSG:4326")
    
    # Optional temporal helpers
    eco_df["is_weekend"] = eco_df["weekday"].isin([5, 6]).astype(int)
    eco_df["season"] = pd.cut(
        eco_df["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon", "Winter2"],
        right=True
    )
    # Cyclical encodings
    eco_df["hour_sin"] = np.sin(2 * np.pi * eco_df["hour"] / 24)
    eco_df["hour_cos"] = np.cos(2 * np.pi * eco_df["hour"] / 24)
    eco_df["month_sin"] = np.sin(2 * np.pi * eco_df["month"] / 12)
    eco_df["month_cos"] = np.cos(2 * np.pi * eco_df["month"] / 12)
    eco_df["weekday_sin"] = np.sin(2 * np.pi * eco_df["weekday"] / 7)
    eco_df["weekday_cos"] = np.cos(2 * np.pi * eco_df["weekday"] / 7)

    # IDW + Raster sampling
    rows = []
    for _, row in tqdm(eco_df.iterrows(), total=len(eco_df), desc="Processing eco sites"):
        lat, lon = float(row["latitude"]), float(row["longitude"])
        idw_vals = idw_interpolate(air_df, lat, lon, row["hour"], row["weekday"], row["month"], pollutants)
        ndvi_val, elev_val = sample_rasters(lat, lon, ndvi_src, elev_src)
        
        # Simple landcover proxy from NDVI thresholds
        if not np.isnan(ndvi_val):
            if ndvi_val >= 0.6:
                lc = 41  # forest-like
            elif ndvi_val >= 0.3:
                lc = 31  # shrub/grass
            else:
                lc = 11  # built/bare proxy
        else:
            lc = 31

        rows.append({
            **row.to_dict(),
            **idw_vals,
            "ndvi_mean": ndvi_val,
            "elev_mean": elev_val,
            "landcover_mode": lc
            })

    eco_full = pd.DataFrame(rows)

    # Calculate derived features for ML model
    # Eco score (same formula)
    for pollutant in pollutants:
        if f"{pollutant}_idw" in eco_full.columns:
            eco_full[f"{pollutant}_idw_norm"] = normalize(eco_full[f"{pollutant}_idw"])
    
    if "ndvi_mean" in eco_full.columns:
        eco_full["ndvi_mean_norm"] = normalize(eco_full["ndvi_mean"])
    if "elev_mean" in eco_full.columns:
        eco_full["elev_mean_norm"] = normalize(eco_full["elev_mean"])
    
    # Interaction Features
    if "ndvi_mean" in eco_full.columns and "elev_mean" in eco_full.columns:
        eco_full["ndvi_elev"] = eco_full["ndvi_mean"] * eco_full["elev_mean"]
    
    if "PM2_5_idw" in eco_full.columns and "PM10_idw" in eco_full.columns:
        eco_full["pm_ratio"] = eco_full["PM2_5_idw"] / (eco_full["PM10_idw"] + 1e-6)
    
    if "AT_idw" in eco_full.columns and "RH_idw" in eco_full.columns:
        eco_full["heat_humidity"] = eco_full["AT_idw"] * eco_full["RH_idw"]
    
    return eco_full