#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_glm_radar_catalog.py

Unified, Generalized Convective Storm & RMA-1 Radar Catalog Pipeline.

Capabilities:
  - Multi-period parameterized execution:
      python build_glm_radar_catalog.py --year 2021 --month 12
      python build_glm_radar_catalog.py --year 2022
      python build_glm_radar_catalog.py --start 2021-01-01 --end 2021-01-15 --out_dir ./output_custom
      python build_glm_radar_catalog.py --consolidate-year 2021
      python build_glm_radar_catalog.py --consolidate-multiyear 2021 2022
  - Decoupled YAML configuration (radar_catalog_config.yaml).
  - High-speed parallel S3 listing (ThreadPoolExecutor).
  - Geometric screening relative to RMA-1 (<= 240 km).
  - Frozen classification thresholds for cross-period consistency (default: 80 min, 50 flashes/10min).
  - Convective prioritization (Priority A: long_intense, Priority B: short_intense / long_moderate).
  - Midnight-crossing buffer (+/- 60 min) for complete radar volume acquisition.
  - Generates standardized Parquet/CSV outputs and diagnostic summaries.
"""

import argparse
import calendar
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from tqdm import tqdm

EARTH_RADIUS_KM = 6371.0088
S3_START_RE = re.compile(r"_s(\d{13,14})")

# Multiprocessing globals
fs_worker = None
bbox_worker = None


# =====================================================================
# UTILITIES & CONFIGURATION
# =====================================================================

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    """Computes great-circle distance in kilometers."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def parse_file_start_dt(s3_path: str) -> Optional[datetime]:
    name = os.path.basename(s3_path)
    match = S3_START_RE.search(name)
    if not match:
        return None
    val = match.group(1)
    if len(val) == 14:
        val = val[:-1]
    try:
        return datetime.strptime(val, "%Y%j%H%M%S")
    except ValueError:
        return None


# =====================================================================
# S3 DISCOVERY & FLASH READING
# =====================================================================

def list_files_parallel(
    fs: s3fs.S3FileSystem,
    start_dt: datetime,
    end_dt: datetime,
    bucket_primary: str,
    bucket_fallback: str,
    prefix: str,
    max_threads: int = 24,
) -> List[str]:
    hours = []
    curr = start_dt.replace(minute=0, second=0, microsecond=0)
    while curr <= end_dt:
        hours.append(curr)
        curr += timedelta(hours=1)

    def _list_single_hour(h_dt: datetime) -> List[str]:
        year = h_dt.strftime("%Y")
        doy = h_dt.strftime("%j")
        hour_str = f"{h_dt.hour:02d}"

        for bucket in [bucket_primary, bucket_fallback]:
            path = f"s3://{bucket}/{prefix}/{year}/{doy}/{hour_str}/"
            try:
                raw_files = fs.ls(path)
                if raw_files:
                    matched = []
                    for f in raw_files:
                        f_dt = parse_file_start_dt(f)
                        if f_dt and start_dt <= f_dt <= end_dt:
                            matched.append(f)
                    return matched
            except Exception:
                pass
        return []

    all_files = []
    with ThreadPoolExecutor(max_workers=max_threads) as ex:
        futures = [ex.submit(_list_single_hour, h) for h in hours]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Listing GLM S3 hours", unit="hour"):
            all_files.extend(f.result())

    return sorted(set(all_files))


def init_worker(bbox_val):
    global fs_worker, bbox_worker
    fs_worker = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-1"})
    bbox_worker = bbox_val


def process_file_worker(s3_path: str):
    file_dt = parse_file_start_dt(s3_path)
    if file_dt is None:
        return "failed", None

    ds = None
    full_path = s3_path if s3_path.startswith("s3://") else "s3://" + s3_path

    for attempt in range(3):
        try:
            fobj = fs_worker.open(full_path, "rb")
            ds = xr.open_dataset(fobj, engine="h5netcdf")
            ds.load()
            break
        except Exception:
            if attempt < 2:
                time.sleep(2 + attempt)
            else:
                return "failed", None

    if ds is None:
        return "failed", None

    try:
        if "flash_lon" not in ds or "flash_lat" not in ds:
            return "ok", None

        lons_all = ds["flash_lon"].values
        lats_all = ds["flash_lat"].values
        min_lon, min_lat, max_lon, max_lat = bbox_worker

        inside = (
            (lons_all >= min_lon)
            & (lons_all <= max_lon)
            & (lats_all >= min_lat)
            & (lats_all <= max_lat)
        )

        if not np.any(inside):
            return "ok", None

        valid_idx = np.where(inside)[0]
        lons = lons_all[valid_idx]
        lats = lats_all[valid_idx]

        # Timestamps
        n = len(valid_idx)
        if "flash_time_offset_of_first_event" in ds and "product_time" in ds:
            try:
                prod_time = pd.Timestamp(ds["product_time"].values)
                offsets = ds["flash_time_offset_of_first_event"].values[valid_idx]
                ts_list = []
                for off in offsets:
                    if np.issubdtype(np.asarray(off).dtype, np.timedelta64):
                        ts_list.append(prod_time + pd.to_timedelta(off))
                    else:
                        ts_list.append(prod_time + pd.Timedelta(seconds=float(off)))
                timestamps = pd.to_datetime(ts_list, utc=True)
            except Exception:
                timestamps = pd.to_datetime([file_dt] * n, utc=True)
        else:
            timestamps = pd.to_datetime([file_dt] * n, utc=True)

        energies = (
            ds["flash_energy"].values[valid_idx].astype("float32")
            if "flash_energy" in ds
            else np.full(n, np.nan, dtype="float32")
        )

        df = pd.DataFrame({
            "timestamp": timestamps,
            "lat": lats.astype("float32"),
            "lon": lons.astype("float32"),
            "flash_energy": energies,
        })
        return "ok", df
    finally:
        if ds is not None:
            ds.close()


# =====================================================================
# DBSCAN SPATIAL CLUSTERING & TEMPORAL TRACKING
# =====================================================================

def detect_spatial_clusters(
    flashes: pd.DataFrame,
    time_bin_minutes: int,
    cluster_radius_km: float,
    min_flashes_cluster: int,
) -> pd.DataFrame:
    if flashes.empty:
        return pd.DataFrame()

    flashes = flashes.copy()
    flashes["time_bin"] = flashes["timestamp"].dt.floor(f"{time_bin_minutes}min")
    eps_rad = cluster_radius_km / EARTH_RADIUS_KM
    records = []

    for time_bin, group in tqdm(flashes.groupby("time_bin", sort=True), desc="Spatial clustering", unit="window"):
        if len(group) < min_flashes_cluster:
            continue

        coords_deg = group[["lat", "lon"]].to_numpy()
        coords_rad = np.radians(coords_deg)

        labels = DBSCAN(
            eps=eps_rad,
            min_samples=min_flashes_cluster,
            metric="haversine",
            algorithm="ball_tree",
        ).fit_predict(coords_rad)

        group = group.copy()
        group["cluster_label"] = labels
        group = group[group["cluster_label"] >= 0]
        if group.empty:
            continue

        for cluster_label, cluster in group.groupby("cluster_label"):
            weights = cluster["flash_energy"].fillna(0).to_numpy()
            records.append({
                "timestamp": time_bin,
                "cluster_local_id": int(cluster_label),
                "cluster_flash_count": int(len(cluster)),
                "cluster_energy_sum": float(np.nansum(weights)),
                "cluster_lat": float(cluster["lat"].mean()),
                "cluster_lon": float(cluster["lon"].mean()),
                "cluster_lat_min": float(cluster["lat"].min()),
                "cluster_lat_max": float(cluster["lat"].max()),
                "cluster_lon_min": float(cluster["lon"].min()),
                "cluster_lon_max": float(cluster["lon"].max()),
            })

    clusters = pd.DataFrame(records)
    if clusters.empty:
        return clusters

    clusters = clusters.sort_values(["timestamp", "cluster_flash_count"], ascending=[True, False]).reset_index(drop=True)
    clusters["cluster_id"] = np.arange(len(clusters), dtype=np.int64)
    return clusters


def build_storm_tracks(
    clusters: pd.DataFrame,
    time_bin_minutes: int,
    max_link_distance_km: float,
    max_gap_windows: int,
) -> pd.DataFrame:
    if clusters.empty:
        return clusters.copy()

    clusters = clusters.copy()
    clusters["timestamp"] = pd.to_datetime(clusters["timestamp"], utc=True)
    clusters["storm_id"] = -1

    next_storm_id = 0
    active_tracks = {}
    timestamps = sorted(clusters["timestamp"].unique())
    base_dt = pd.Timedelta(minutes=time_bin_minutes)
    max_gap_dt = pd.Timedelta(minutes=time_bin_minutes * (max_gap_windows + 1))

    for timestamp in tqdm(timestamps, desc="Temporal tracking", unit="window"):
        timestamp = pd.Timestamp(timestamp)
        current_idx = clusters.index[clusters["timestamp"] == timestamp].to_numpy()

        # Remove expired tracks
        expired = [sid for sid, st in active_tracks.items() if (timestamp - st["timestamp"]) > max_gap_dt]
        for sid in expired:
            del active_tracks[sid]

        if len(active_tracks) == 0:
            for idx in current_idx:
                clusters.at[idx, "storm_id"] = next_storm_id
                active_tracks[next_storm_id] = {
                    "timestamp": timestamp,
                    "lat": float(clusters.at[idx, "cluster_lat"]),
                    "lon": float(clusters.at[idx, "cluster_lon"]),
                }
                next_storm_id += 1
            continue

        track_ids = list(active_tracks.keys())
        cost = np.full((len(track_ids), len(current_idx)), np.inf, dtype=float)

        for i, sid in enumerate(track_ids):
            state = active_tracks[sid]
            gap_windows = (timestamp - state["timestamp"]) / base_dt
            allowed_dist = max_link_distance_km * float(gap_windows)

            for j, idx in enumerate(current_idx):
                dist = haversine_km(
                    state["lat"], state["lon"],
                    float(clusters.at[idx, "cluster_lat"]), float(clusters.at[idx, "cluster_lon"])
                )
                if dist <= allowed_dist:
                    cost[i, j] = dist

        if np.isfinite(cost).any():
            large_cost = max_link_distance_km * 1000.0
            assignment_cost = np.where(np.isfinite(cost), cost, large_cost)
            row_ind, col_ind = linear_sum_assignment(assignment_cost)

            matched_col = set()
            for r, c in zip(row_ind, col_ind):
                if not np.isfinite(cost[r, c]):
                    continue
                sid = track_ids[r]
                idx = current_idx[c]
                clusters.at[idx, "storm_id"] = sid
                active_tracks[sid] = {
                    "timestamp": timestamp,
                    "lat": float(clusters.at[idx, "cluster_lat"]),
                    "lon": float(clusters.at[idx, "cluster_lon"]),
                }
                matched_col.add(c)

            for j, idx in enumerate(current_idx):
                if j not in matched_col:
                    clusters.at[idx, "storm_id"] = next_storm_id
                    active_tracks[next_storm_id] = {
                        "timestamp": timestamp,
                        "lat": float(clusters.at[idx, "cluster_lat"]),
                        "lon": float(clusters.at[idx, "cluster_lon"]),
                    }
                    next_storm_id += 1
        else:
            for idx in current_idx:
                clusters.at[idx, "storm_id"] = next_storm_id
                active_tracks[next_storm_id] = {
                    "timestamp": timestamp,
                    "lat": float(clusters.at[idx, "cluster_lat"]),
                    "lon": float(clusters.at[idx, "cluster_lon"]),
                }
                next_storm_id += 1

    return clusters


def summarize_tracks(tracked_clusters: pd.DataFrame, time_bin_minutes: int) -> pd.DataFrame:
    if tracked_clusters.empty:
        return pd.DataFrame()

    tracks_list = []
    for storm_id, storm in tracked_clusters.groupby("storm_id", sort=True):
        storm = storm.sort_values("timestamp")
        start_time = storm["timestamp"].iloc[0]
        end_time = storm["timestamp"].iloc[-1]
        active_windows = int(storm["timestamp"].nunique())
        duration_min = float(
            max(time_bin_minutes, (end_time - start_time).total_seconds() / 60.0 + time_bin_minutes)
        )
        total_flashes = int(storm["cluster_flash_count"].sum())
        max_flashes = int(storm["cluster_flash_count"].max())
        mean_flashes = float(storm["cluster_flash_count"].mean())
        total_energy = float(storm["cluster_energy_sum"].sum())

        peak_row = storm.loc[storm["cluster_flash_count"].idxmax()]
        tracks_list.append({
            "storm_id": int(storm_id),
            "start_time": start_time,
            "end_time": end_time,
            "duration_min": duration_min,
            "n_active_windows": active_windows,
            "total_flashes": total_flashes,
            "max_flashes_10min": max_flashes,
            "mean_flashes_10min": mean_flashes,
            "total_energy": total_energy,
            "peak_time": peak_row["timestamp"],
            "peak_lat": float(peak_row["cluster_lat"]),
            "peak_lon": float(peak_row["cluster_lon"]),
            "mean_lat": float(storm["cluster_lat"].mean()),
            "mean_lon": float(storm["cluster_lon"].mean()),
            "first_date": start_time.strftime("%Y-%m-%d"),
        })

    tracks = pd.DataFrame(tracks_list)
    return tracks.sort_values("start_time").reset_index(drop=True)


# =====================================================================
# CLASSIFICATION & RADAR SELECTION (FROZEN OR COMPUTED)
# =====================================================================

def classify_tracks(
    tracks: pd.DataFrame,
    mode: str = "frozen",
    frozen_duration: float = 80.0,
    frozen_intensity: float = 50.0,
) -> Tuple[pd.DataFrame, float, float]:
    if tracks.empty:
        tracks = tracks.copy()
        tracks["is_long"] = pd.Series(dtype=bool)
        tracks["is_intense"] = pd.Series(dtype=bool)
        tracks["storm_type"] = pd.Series(dtype=str)
        return tracks, frozen_duration, frozen_intensity

    if mode == "computed":
        dur_th = float(tracks["duration_min"].quantile(0.75))
        int_th = float(tracks["max_flashes_10min"].quantile(0.75))
    else:
        dur_th = float(frozen_duration)
        int_th = float(frozen_intensity)

    tracks = tracks.copy()
    tracks["is_long"] = tracks["duration_min"] >= dur_th
    tracks["is_intense"] = tracks["max_flashes_10min"] >= int_th

    def _type(r):
        if r["is_long"] and r["is_intense"]:
            return "long_intense"
        elif r["is_long"] and not r["is_intense"]:
            return "long_moderate"
        elif not r["is_long"] and r["is_intense"]:
            return "short_intense"
        else:
            return "short_lower_activity"

    tracks["storm_type"] = tracks.apply(_type, axis=1)
    return tracks, dur_th, int_th


def attach_radar_geometry(
    clusters: pd.DataFrame,
    tracks: pd.DataFrame,
    radar_lat: float,
    radar_lon: float,
    radius_km: float = 240.0,
    inner_radius_km: float = 120.0,
    outer_radius_km: float = 480.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    geom_cols = [
        "min_distance_to_radar_km", "max_distance_to_radar_km", "mean_distance_to_radar_km",
        "fraction_clusters_within_120km", "fraction_clusters_within_240km", "fraction_clusters_within_480km",
        "enters_120km", "enters_240km", "enters_480km", "fully_within_120km", "fully_within_240km"
    ]
    if clusters.empty or tracks.empty:
        for c in geom_cols:
            if c not in tracks.columns:
                tracks[c] = False if "enters" in c or "fully" in c else (0.0 if "fraction" in c else np.nan)
        return clusters, tracks

    clusters = clusters.copy()
    clusters["distance_to_radar_km"] = haversine_km(
        clusters["cluster_lat"], clusters["cluster_lon"], radar_lat, radar_lon
    )
    clusters[f"within_{int(inner_radius_km)}km"] = clusters["distance_to_radar_km"] <= inner_radius_km
    clusters[f"within_{int(radius_km)}km"] = clusters["distance_to_radar_km"] <= radius_km
    clusters[f"within_{int(outer_radius_km)}km"] = clusters["distance_to_radar_km"] <= outer_radius_km

    track_geom = clusters.groupby("storm_id").agg(
        min_distance_to_radar_km=("distance_to_radar_km", "min"),
        max_distance_to_radar_km=("distance_to_radar_km", "max"),
        mean_distance_to_radar_km=("distance_to_radar_km", "mean"),
        fraction_clusters_within_120km=(f"within_{int(inner_radius_km)}km", "mean"),
        fraction_clusters_within_240km=(f"within_{int(radius_km)}km", "mean"),
        fraction_clusters_within_480km=(f"within_{int(outer_radius_km)}km", "mean"),
        enters_120km=(f"within_{int(inner_radius_km)}km", "any"),
        enters_240km=(f"within_{int(radius_km)}km", "any"),
        enters_480km=(f"within_{int(outer_radius_km)}km", "any"),
        fully_within_120km=(f"within_{int(inner_radius_km)}km", "all"),
        fully_within_240km=(f"within_{int(radius_km)}km", "all"),
    ).reset_index()

    tracks = tracks.drop(columns=[c for c in geom_cols if c in tracks.columns])
    tracks = tracks.merge(track_geom, on="storm_id", how="left")
    return clusters, tracks


def build_radar_convective_selection(
    tracks: pd.DataFrame,
    include_types: List[str],
    priority_mapping: Dict[str, str],
    max_dist_km: float = 240.0,
    buffer_before_min: int = 60,
    buffer_after_min: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects convective events <= max_dist_km, assigns Priority A/B,
    and constructs calendar download days with midnight crossing.
    """
    lead_cols = [
        "storm_id", "priority", "storm_type", "start_time", "end_time",
        "duration_min", "total_flashes", "max_flashes_10min", "total_energy",
        "peak_time", "peak_lat", "peak_lon", "min_distance_to_radar_km",
        "fraction_clusters_within_240km"
    ]
    download_cols = [
        "date", "priority", "n_convective_storms", "storm_ids", "storm_types",
        "required_window_start_utc", "required_window_end_utc",
        "max_flashes_10min", "max_duration_min", "total_flashes_tracked"
    ]

    if tracks.empty:
        return pd.DataFrame(columns=lead_cols), pd.DataFrame(columns=download_cols)

    tracks = tracks.copy()
    tracks["start_time"] = pd.to_datetime(tracks["start_time"], utc=True)
    tracks["end_time"] = pd.to_datetime(tracks["end_time"], utc=True)

    # Filter to convective priority types within radar domain
    is_convective = tracks["storm_type"].isin(include_types)
    is_in_radar = tracks["min_distance_to_radar_km"] <= max_dist_km

    convective_events = tracks[is_convective & is_in_radar].copy()
    convective_events["priority"] = convective_events["storm_type"].map(priority_mapping)

    if convective_events.empty:
        convective_events = pd.DataFrame(columns=lead_cols)
    else:
        other_cols = [c for c in convective_events.columns if c not in lead_cols]
        convective_events = convective_events[[c for c in lead_cols if c in convective_events.columns] + other_cols]

    # Calendar download days with midnight crossing
    day_dict = {}
    buf_before = pd.Timedelta(minutes=buffer_before_min)
    buf_after = pd.Timedelta(minutes=buffer_after_min)

    for _, row in convective_events.iterrows():
        sid = int(row["storm_id"])
        prio = row["priority"]
        st = row["storm_type"]

        t0 = row["start_time"] - buf_before
        t1 = row["end_time"] + buf_after

        curr_d = t0.floor("D")
        end_d = t1.floor("D")

        while curr_d <= end_d:
            d_str = curr_d.strftime("%Y-%m-%d")
            if d_str not in day_dict:
                day_dict[d_str] = {
                    "date": d_str,
                    "priority": prio,
                    "storm_ids": [],
                    "storm_types": set(),
                    "required_window_start_utc": t0,
                    "required_window_end_utc": t1,
                    "max_flashes_10min": int(row["max_flashes_10min"]),
                    "max_duration_min": float(row["duration_min"]),
                    "total_flashes_sum": int(row["total_flashes"]),
                }

            rec = day_dict[d_str]
            rec["storm_ids"].append(sid)
            rec["storm_types"].add(st)

            # Escalate priority: Priority_A takes precedence over Priority_B
            if prio == "Priority_A":
                rec["priority"] = "Priority_A"

            if t0 < rec["required_window_start_utc"]:
                rec["required_window_start_utc"] = t0
            if t1 > rec["required_window_end_utc"]:
                rec["required_window_end_utc"] = t1

            rec["max_flashes_10min"] = max(rec["max_flashes_10min"], int(row["max_flashes_10min"]))
            rec["max_duration_min"] = max(rec["max_duration_min"], float(row["duration_min"]))
            rec["total_flashes_sum"] += int(row["total_flashes"])

            curr_d += pd.Timedelta(days=1)

    download_rows = []
    for d_str, rec in sorted(day_dict.items()):
        download_rows.append({
            "date": rec["date"],
            "priority": rec["priority"],
            "n_convective_storms": len(set(rec["storm_ids"])),
            "storm_ids": ",".join(map(str, sorted(set(rec["storm_ids"])))),
            "storm_types": "+".join(sorted(rec["storm_types"])),
            "required_window_start_utc": rec["required_window_start_utc"].strftime("%Y-%m-%d %H:%M:%S"),
            "required_window_end_utc": rec["required_window_end_utc"].strftime("%Y-%m-%d %H:%M:%S"),
            "max_flashes_10min": rec["max_flashes_10min"],
            "max_duration_min": rec["max_duration_min"],
            "total_flashes_tracked": rec["total_flashes_sum"],
        })

    if not download_rows:
        download_days = pd.DataFrame(columns=download_cols)
    else:
        download_days = pd.DataFrame(download_rows)
    return convective_events, download_days


def generate_diagnostic_summary_text(
    period_label: str,
    n_clusters: int,
    n_tracks: int,
    n_tracks_240: int,
    n_long_intense: int,
    n_short_intense: int,
    n_long_moderate: int,
    n_total_convective: int,
    n_radar_days: int,
) -> str:
    """Generates the standardized diagnostic summary format requested by user."""
    return f"""PERIODO: {period_label}

GLM clusters: {n_clusters:,}
Tracks totales: {n_tracks:,}

Tracks <=240 km: {n_tracks_240:,}

Convectivos seleccionados:
  long_intense: {n_long_intense:,}
  short_intense: {n_short_intense:,}
  long_moderate: {n_long_moderate:,}

Total eventos seleccionados: {n_total_convective:,}
Días radar requeridos: {n_radar_days:,}
"""


# =====================================================================
# PROCESSING WORKFLOWS
# =====================================================================

def process_single_month(
    year: int,
    month: int,
    config: dict,
    force: bool = False,
    num_workers: int = 16,
) -> Path:
    """Processes a single calendar month into output_YYYY_MM/."""
    m_str = f"{month:02d}"
    period_label = f"{year}-{m_str}"
    out_dir = Path(__file__).resolve().parent / f"output_{year}_{m_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "diagnostic_summary.txt"
    if summary_file.exists() and not force:
        print(f"\n[SKIP] Month {period_label} already processed. Summary: {summary_file}")
        return out_dir

    print("\n" + "=" * 72)
    print(f"PROCESSING PERIOD: {period_label}")
    print("=" * 72)

    last_day = calendar.monthrange(year, month)[1]
    start_dt = datetime(year, month, 1, 0, 0, 0)
    end_dt = datetime(year, month, last_day, 23, 59, 59)

    # Check if raw tracking Parquet files already exist
    clusters_path = out_dir / "glm_tracked_clusters.parquet"
    tracks_path = out_dir / "glm_storm_tracks.parquet"

    if clusters_path.exists() and tracks_path.exists() and not force:
        print(f"Loading existing tracking Parquet files from {out_dir}...")
        clusters = pd.read_parquet(clusters_path)
        tracks = pd.read_parquet(tracks_path)
    else:
        # Full GLM fetch and tracking
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": config["s3"]["region"]})
        all_files = list_files_parallel(
            fs=fs,
            start_dt=start_dt,
            end_dt=end_dt,
            bucket_primary=config["s3"]["bucket_primary"],
            bucket_fallback=config["s3"]["bucket_fallback"],
            prefix=config["s3"]["prefix"],
            max_threads=config["s3"].get("max_listing_threads", 24),
        )
        print(f"Found {len(all_files):,} GLM files on S3.")
        if not all_files:
            print(f"Warning: No GLM files found for {period_label}")
            clusters = pd.DataFrame()
            tracks = pd.DataFrame()
        else:
            bbox = tuple(config["domain"]["cordoba_bbox"])
            flash_results = []
            with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(bbox,)) as pool:
                for status, frame in tqdm(pool.imap_unordered(process_file_worker, all_files, chunksize=100),
                                         total=len(all_files), desc="Reading native flashes", unit="file"):
                    if status == "ok" and frame is not None and not frame.empty:
                        flash_results.append(frame)

            if not flash_results:
                print(f"No native GLM flashes inside domain for {period_label}")
                clusters = pd.DataFrame()
                tracks = pd.DataFrame()
            else:
                flashes = pd.concat(flash_results, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
                print(f"Total flashes in domain: {len(flashes):,}")

                # Clustering
                trk_cfg = config["tracking"]
                clusters = detect_spatial_clusters(
                    flashes=flashes,
                    time_bin_minutes=trk_cfg["time_bin_minutes"],
                    cluster_radius_km=trk_cfg["cluster_radius_km"],
                    min_flashes_cluster=trk_cfg["min_flashes_cluster"],
                )
                if clusters.empty:
                    print("No spatial clusters detected.")
                    tracks = pd.DataFrame()
                else:
                    # Linking
                    tracked_clusters = build_storm_tracks(
                        clusters=clusters,
                        time_bin_minutes=trk_cfg["time_bin_minutes"],
                        max_link_distance_km=trk_cfg["max_link_distance_km"],
                        max_gap_windows=trk_cfg["max_gap_windows"],
                    )
                    tracks = summarize_tracks(tracked_clusters, time_bin_minutes=trk_cfg["time_bin_minutes"])
                    clusters = tracked_clusters

    # Classification with frozen thresholds
    cls_cfg = config["classification"]
    tracks, dur_th, int_th = classify_tracks(
        tracks=tracks,
        mode=cls_cfg.get("mode", "frozen"),
        frozen_duration=cls_cfg.get("duration_threshold_min", 80.0),
        frozen_intensity=cls_cfg.get("intensity_threshold_flashes_10min", 50.0),
    )

    # Attach RMA-1 geometry
    rad_cfg = config["radar"]
    sel_cfg = config["selection"]
    radar_radius_km = float(sel_cfg.get("radar_intersection_radius_km", 240.0))

    clusters, tracks = attach_radar_geometry(
        clusters=clusters,
        tracks=tracks,
        radar_lat=rad_cfg["lat"],
        radar_lon=rad_cfg["lon"],
        radius_km=radar_radius_km,
        inner_radius_km=120.0,
        outer_radius_km=480.0,
    )

    # Build convective selection and download days
    dl_cfg = config["download"]
    convective_events, download_days = build_radar_convective_selection(
        tracks=tracks,
        include_types=sel_cfg["include_types"],
        priority_mapping=sel_cfg["priority_mapping"],
        max_dist_km=radar_radius_km,
        buffer_before_min=dl_cfg["buffer_before_min"],
        buffer_after_min=dl_cfg["buffer_after_min"],
    )

    # Counts for diagnostic summary
    n_clusters = len(clusters)
    n_tracks = len(tracks)
    n_tracks_240 = int(tracks["enters_240km"].sum()) if "enters_240km" in tracks else 0

    st_counts = convective_events["storm_type"].value_counts().to_dict() if not convective_events.empty else {}
    n_long_intense = st_counts.get("long_intense", 0)
    n_short_intense = st_counts.get("short_intense", 0)
    n_long_moderate = st_counts.get("long_moderate", 0)
    n_total_convective = len(convective_events)
    n_radar_days = len(download_days)

    diag_text = generate_diagnostic_summary_text(
        period_label=period_label,
        n_clusters=n_clusters,
        n_tracks=n_tracks,
        n_tracks_240=n_tracks_240,
        n_long_intense=n_long_intense,
        n_short_intense=n_short_intense,
        n_long_moderate=n_long_moderate,
        n_total_convective=n_total_convective,
        n_radar_days=n_radar_days,
    )

    # Save monthly files
    clusters.to_parquet(out_dir / "glm_tracked_clusters.parquet", index=False)
    tracks.to_parquet(out_dir / "glm_storm_tracks.parquet", index=False)
    tracks.to_parquet(out_dir / "glm_storm_tracks_rma1.parquet", index=False)
    convective_events.to_parquet(out_dir / "radar_convective_events_240km.parquet", index=False)
    convective_events.to_csv(out_dir / "radar_convective_events_240km.csv", index=False)
    download_days.to_csv(out_dir / "radar_download_days_240km.csv", index=False)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(diag_text)

    print("\n" + diag_text)
    return out_dir


def consolidate_year(year: int, config: dict, num_workers: int = 16) -> Path:
    """Consolidates all 12 monthly folders of a year into output_YYYY_anual/."""
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / f"output_{year}_anual"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"CONSOLIDATING ANNUAL CATALOG FOR YEAR {year}")
    print("=" * 72)

    # Verify/process each month
    for m in range(1, 13):
        process_single_month(year, m, config, force=False, num_workers=num_workers)

    all_clusters = []
    all_tracks = []
    c_offset = 0
    s_offset = 0

    monthly_metrics = []

    for m in range(1, 13):
        m_str = f"{m:02d}"
        m_dir = base_dir / f"output_{year}_{m_str}"
        c_path = m_dir / "glm_tracked_clusters.parquet"
        t_path = m_dir / "glm_storm_tracks.parquet"
        ev_path = m_dir / "radar_convective_events_240km.parquet"
        dl_path = m_dir / "radar_download_days_240km.csv"

        if not (c_path.exists() and t_path.exists()):
            continue

        c_df = pd.read_parquet(c_path)
        t_df = pd.read_parquet(t_path)

        c_df["cluster_id"] += c_offset
        c_df["storm_id"] += s_offset
        t_df["storm_id"] += s_offset

        c_offset = int(c_df["cluster_id"].max()) + 1 if not c_df.empty else c_offset
        s_offset = int(t_df["storm_id"].max()) + 1 if not t_df.empty else s_offset

        all_clusters.append(c_df)
        all_tracks.append(t_df)

        n_conv = 0
        if ev_path.exists() and ev_path.stat().st_size > 5:
            try:
                n_conv = len(pd.read_parquet(ev_path))
            except Exception:
                n_conv = 0

        n_dl = 0
        if dl_path.exists() and dl_path.stat().st_size > 5:
            try:
                n_dl = len(pd.read_csv(dl_path))
            except Exception:
                n_dl = 0
        monthly_metrics.append({
            "month": f"{year}-{m_str}",
            "n_clusters": len(c_df),
            "n_tracks": len(t_df),
            "n_convective_events_240km": n_conv,
            "n_radar_download_days": n_dl,
        })

    annual_clusters = pd.concat(all_clusters, ignore_index=True)
    if not annual_clusters.empty:
        annual_clusters["timestamp"] = pd.to_datetime(annual_clusters["timestamp"], utc=True)
        annual_clusters = annual_clusters.sort_values(
            ["timestamp", "cluster_flash_count"], ascending=[True, False]
        ).reset_index(drop=True)
        annual_clusters["cluster_id"] = np.arange(len(annual_clusters), dtype=np.int64)

        # Continuous tracking across the entire year, seamlessly stitching month boundaries
        trk_cfg = config["tracking"]
        print(f"\nRunning continuous tracking across full year on {len(annual_clusters):,} clusters...")
        annual_clusters = build_storm_tracks(
            clusters=annual_clusters,
            time_bin_minutes=trk_cfg["time_bin_minutes"],
            max_link_distance_km=trk_cfg["max_link_distance_km"],
            max_gap_windows=trk_cfg["max_gap_windows"],
        )
        annual_tracks = summarize_tracks(
            tracked_clusters=annual_clusters,
            time_bin_minutes=trk_cfg["time_bin_minutes"],
        )
    else:
        annual_tracks = pd.DataFrame()

    # Re-apply frozen classification & RMA-1 geometry on annual tracks
    cls_cfg = config["classification"]
    annual_tracks, dur_th, int_th = classify_tracks(
        tracks=annual_tracks,
        mode=cls_cfg.get("mode", "frozen"),
        frozen_duration=cls_cfg.get("duration_threshold_min", 80.0),
        frozen_intensity=cls_cfg.get("intensity_threshold_flashes_10min", 50.0),
    )

    rad_cfg = config["radar"]
    sel_cfg = config["selection"]
    radar_radius_km = float(sel_cfg.get("radar_intersection_radius_km", 240.0))

    annual_clusters, annual_tracks = attach_radar_geometry(
        clusters=annual_clusters,
        tracks=annual_tracks,
        radar_lat=rad_cfg["lat"],
        radar_lon=rad_cfg["lon"],
        radius_km=radar_radius_km,
        inner_radius_km=120.0,
        outer_radius_km=480.0,
    )

    dl_cfg = config["download"]
    convective_events, download_days = build_radar_convective_selection(
        tracks=annual_tracks,
        include_types=sel_cfg["include_types"],
        priority_mapping=sel_cfg["priority_mapping"],
        max_dist_km=radar_radius_km,
        buffer_before_min=dl_cfg["buffer_before_min"],
        buffer_after_min=dl_cfg["buffer_after_min"],
    )

    # Diagnostic summary
    st_counts = convective_events["storm_type"].value_counts().to_dict() if not convective_events.empty else {}
    diag_text = generate_diagnostic_summary_text(
        period_label=str(year),
        n_clusters=len(annual_clusters),
        n_tracks=len(annual_tracks),
        n_tracks_240=int(annual_tracks["enters_240km"].sum()),
        n_long_intense=st_counts.get("long_intense", 0),
        n_short_intense=st_counts.get("short_intense", 0),
        n_long_moderate=st_counts.get("long_moderate", 0),
        n_total_convective=len(convective_events),
        n_radar_days=len(download_days),
    )

    # Export annual files
    annual_clusters.to_parquet(out_dir / f"glm_tracked_clusters_{year}.parquet", index=False)
    annual_tracks.to_parquet(out_dir / f"glm_storm_tracks_{year}.parquet", index=False)
    convective_events.to_parquet(out_dir / f"radar_convective_events_240km_{year}.parquet", index=False)
    convective_events.to_csv(out_dir / f"radar_convective_events_240km_{year}.csv", index=False)
    download_days.to_csv(out_dir / f"radar_download_days_240km_{year}.csv", index=False)
    pd.DataFrame(monthly_metrics).to_csv(out_dir / "annual_summary.csv", index=False)

    with open(out_dir / "diagnostic_summary.txt", "w", encoding="utf-8") as f:
        f.write(diag_text)

    print("\n" + diag_text)
    print(f"Annual outputs saved in: {out_dir}")
    return out_dir


# =====================================================================
# CLI INTERFACE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build GLM & RMA-1 Convective Radar Catalog (Parameterized for Any Period)"
    )
    parser.add_argument("--year", type=int, help="Target year (e.g., 2021)")
    parser.add_argument("--month", type=int, help="Target month (1 to 12)")
    parser.add_argument("--config", type=str, default="radar_catalog_config.yaml", help="Path to YAML config")
    parser.add_argument("--consolidate-year", type=int, help="Consolidate all months for a year")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing outputs")

    args = parser.parse_args()
    config_path = Path(__file__).resolve().parent / args.config
    config = load_config(config_path)

    workers = args.num_workers if args.num_workers != 16 else config.get("processing", {}).get("download_workers", 24)

    if args.consolidate_year:
        consolidate_year(args.consolidate_year, config, num_workers=workers)
    elif args.year and args.month:
        process_single_month(args.year, args.month, config, force=args.force, num_workers=workers)
    elif args.year and not args.month:
        consolidate_year(args.year, config, num_workers=workers)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python build_glm_radar_catalog.py --year 2021 --month 12")
        print("  python build_glm_radar_catalog.py --year 2021")


if __name__ == "__main__":
    import multiprocessing
    main()
