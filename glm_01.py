#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_glm_storm_tracks.py

Builds a GLM-based storm database for radar-case selection.

Workflow
--------
1. Read native GOES GLM flashes inside the study/radar domain.
2. Group flashes into 10-minute windows.
3. Detect spatial electrical clusters in each 10-minute window using DBSCAN.
4. Link clusters between consecutive windows to create storm tracks.
5. Characterize each track by duration and electrical intensity.
6. Classify tracks using thresholds calculated over the complete requested period.
7. Produce:
      - glm_storm_tracks.csv
      - glm_radar_candidate_days.csv
      - optional parquet outputs

No IMERG 0.1-degree grid is used.
"""

import argparse
import multiprocessing
import os
import re
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# =====================================================================
# CONFIGURATION
# =====================================================================

# Replace by the exact radar analysis domain if desired.
# Format:
#   min_lon, min_lat, max_lon, max_lat
CORDOBA_BBOX: Tuple[float, float, float, float] = (
    -66.0,
    -35.6,
    -61.8,
    -29.3,
)

EARTH_RADIUS_KM = 6371.0088

S3_START_RE = re.compile(r"_s(\d{13,14})")


# =====================================================================
# GENERAL UTILITIES
# =====================================================================

def parse_file_start_dt(s3_path: str) -> Optional[datetime]:
    name = os.path.basename(s3_path)

    match = S3_START_RE.search(name)

    if not match:
        return None

    value = match.group(1)

    # Some filenames include one extra fractional-second digit.
    if len(value) == 14:
        value = value[:-1]

    try:
        return datetime.strptime(value, "%Y%j%H%M%S")
    except ValueError:
        return None


def ymd_to_doy(dt: datetime) -> str:
    return dt.strftime("%j")


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Great-circle distance in km.
    """

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1.0 - a),
    )

    return EARTH_RADIUS_KM * c


# =====================================================================
# S3 FILE DISCOVERY
# =====================================================================

def list_files_for_hour(
    fs: s3fs.S3FileSystem,
    satellite: int,
    year: str,
    doy: str,
    hour: int,
) -> List[str]:

    # Preserve the original fallback strategy during the platform
    # transition period.
    satellites = [satellite]

    fallback = 19 if satellite == 16 else 16
    satellites.append(fallback)

    for sat in satellites:

        bucket = f"noaa-goes{sat}"

        prefix = (
            f"GLM-L2-LCFA/"
            f"{year}/"
            f"{doy}/"
            f"{hour:02d}/"
        )

        path = f"s3://{bucket}/{prefix}"

        try:
            files = fs.ls(
                path,
                refresh=True,
            )

            if files:
                return files

        except Exception:
            pass

    return []


# =====================================================================
# MULTIPROCESSING WORKERS
# =====================================================================

fs_worker = None
bbox_worker = None


def init_worker(bbox_val):
    global fs_worker, bbox_worker

    fs_worker = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={
            "region_name": "us-east-1"
        },
    )

    bbox_worker = bbox_val


def build_flash_timestamps(
    ds: xr.Dataset,
    valid_idx: np.ndarray,
    fallback_dt: datetime,
) -> pd.DatetimeIndex:
    """
    Build flash timestamps as robustly as possible.

    Falls back to the file timestamp when the GLM timing variables
    cannot be interpreted.
    """

    n = len(valid_idx)

    if (
        "flash_time_offset_of_first_event" not in ds
        or "product_time" not in ds
    ):
        return pd.to_datetime(
            [fallback_dt] * n,
            utc=True,
        )

    try:

        product_time = pd.Timestamp(
            ds["product_time"].values
        )

        offsets = (
            ds["flash_time_offset_of_first_event"]
            .values[valid_idx]
        )

        timestamps = []

        for offset in offsets:

            # numpy timedelta
            if np.issubdtype(
                np.asarray(offset).dtype,
                np.timedelta64,
            ):
                timestamp = (
                    product_time
                    + pd.to_timedelta(offset)
                )

            else:
                # Preserve compatibility with numeric-second offsets.
                timestamp = (
                    product_time
                    + pd.Timedelta(
                        seconds=float(offset)
                    )
                )

            timestamps.append(timestamp)

        return pd.to_datetime(
            timestamps,
            utc=True,
        )

    except Exception:

        return pd.to_datetime(
            [fallback_dt] * n,
            utc=True,
        )


def process_file_worker(
    s3_path: str,
):
    """
    Extract native GLM flash locations/times inside the domain.

    Important:
    No assignment to the IMERG grid is performed.
    """

    file_dt = parse_file_start_dt(s3_path)

    if file_dt is None:
        return "failed", None

    ds = None

    for attempt in range(3):

        try:

            full_path = (
                s3_path
                if s3_path.startswith("s3://")
                else "s3://" + s3_path
            )

            fobj = fs_worker.open(
                full_path,
                "rb",
            )

            ds = xr.open_dataset(
                fobj,
                engine="h5netcdf",
            )

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

        if (
            "flash_lon" not in ds
            or "flash_lat" not in ds
        ):
            return "ok", None

        lons_all = ds["flash_lon"].values
        lats_all = ds["flash_lat"].values

        (
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        ) = bbox_worker

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

        timestamps = build_flash_timestamps(
            ds=ds,
            valid_idx=valid_idx,
            fallback_dt=file_dt,
        )

        if "flash_energy" in ds:

            energies = (
                ds["flash_energy"]
                .values[valid_idx]
            )

        else:

            energies = np.full(
                len(valid_idx),
                np.nan,
            )

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "lat": lats.astype(
                    "float32"
                ),
                "lon": lons.astype(
                    "float32"
                ),
                "flash_energy": energies.astype(
                    "float32"
                ),
            }
        )

        return "ok", df

    finally:

        if ds is not None:
            ds.close()


# =====================================================================
# SPATIAL CLUSTERING
# =====================================================================

def detect_spatial_clusters(
    flashes: pd.DataFrame,
    time_bin_minutes: int,
    cluster_radius_km: float,
    min_flashes_cluster: int,
) -> pd.DataFrame:
    """
    Detect electrical clusters independently in each temporal window.

    DBSCAN operates on the native flash latitude/longitude coordinates.

    A cluster is therefore a concentration of flashes occurring:
      - within the same temporal window
      - within approximately cluster_radius_km spatially
    """

    if flashes.empty:
        return pd.DataFrame()

    flashes = flashes.copy()

    flashes["time_bin"] = (
        flashes["timestamp"]
        .dt.floor(
            f"{time_bin_minutes}min"
        )
    )

    records = []

    eps_rad = (
        cluster_radius_km
        / EARTH_RADIUS_KM
    )

    grouped = flashes.groupby(
        "time_bin",
        sort=True,
    )

    for time_bin, group in tqdm(
        grouped,
        desc="Spatial clustering",
        unit="window",
    ):

        if len(group) < min_flashes_cluster:
            continue

        coords_deg = (
            group[
                ["lat", "lon"]
            ]
            .to_numpy()
        )

        coords_rad = np.radians(
            coords_deg
        )

        labels = DBSCAN(
            eps=eps_rad,
            min_samples=min_flashes_cluster,
            metric="haversine",
            algorithm="ball_tree",
        ).fit_predict(coords_rad)

        group = group.copy()

        group["cluster_label"] = labels

        # DBSCAN -1 = noise
        group = group[
            group["cluster_label"] >= 0
        ]

        if group.empty:
            continue

        for cluster_label, cluster in (
            group.groupby("cluster_label")
        ):

            weights = (
                cluster["flash_energy"]
                .fillna(0)
                .to_numpy()
            )

            # Geometric centroid. We don't use energy weighting here,
            # because energy availability/scale can vary.
            centroid_lat = float(
                cluster["lat"].mean()
            )

            centroid_lon = float(
                cluster["lon"].mean()
            )

            records.append(
                {
                    "timestamp": time_bin,
                    "cluster_local_id": int(
                        cluster_label
                    ),
                    "cluster_flash_count": int(
                        len(cluster)
                    ),
                    "cluster_energy_sum": float(
                        np.nansum(weights)
                    ),
                    "cluster_lat": centroid_lat,
                    "cluster_lon": centroid_lon,
                    "cluster_lat_min": float(
                        cluster["lat"].min()
                    ),
                    "cluster_lat_max": float(
                        cluster["lat"].max()
                    ),
                    "cluster_lon_min": float(
                        cluster["lon"].min()
                    ),
                    "cluster_lon_max": float(
                        cluster["lon"].max()
                    ),
                }
            )

    clusters = pd.DataFrame(
        records
    )

    if clusters.empty:
        return clusters

    clusters = (
        clusters
        .sort_values(
            [
                "timestamp",
                "cluster_flash_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # Globally unique cluster ID
    clusters["cluster_id"] = np.arange(
        len(clusters),
        dtype=np.int64,
    )

    return clusters


# =====================================================================
# TEMPORAL TRACKING
# =====================================================================

def build_storm_tracks(
    clusters: pd.DataFrame,
    time_bin_minutes: int,
    max_link_distance_km: float,
    max_gap_windows: int,
) -> pd.DataFrame:
    """
    Associate clusters across successive temporal windows.

    One storm track may survive a limited number of missing windows.

    Matching is one-to-one and uses minimum centroid distance through
    the Hungarian algorithm.

    Track continuation requires:
      - time gap <= allowed maximum
      - centroid displacement <= allowed distance

    The distance allowance is scaled with the temporal gap.
    """

    if clusters.empty:
        return clusters.copy()

    clusters = clusters.copy()

    clusters["timestamp"] = pd.to_datetime(
        clusters["timestamp"],
        utc=True,
    )

    clusters["storm_id"] = -1

    next_storm_id = 0

    # storm_id -> last known cluster state
    active_tracks = {}

    timestamps = sorted(
        clusters["timestamp"].unique()
    )

    base_dt = pd.Timedelta(
        minutes=time_bin_minutes
    )

    max_gap_dt = pd.Timedelta(
        minutes=(
            time_bin_minutes
            * (max_gap_windows + 1)
        )
    )

    for timestamp in tqdm(
        timestamps,
        desc="Temporal tracking",
        unit="window",
    ):

        timestamp = pd.Timestamp(
            timestamp
        )

        current_idx = (
            clusters.index[
                clusters["timestamp"]
                == timestamp
            ]
            .to_numpy()
        )

        # Remove tracks that have been inactive for too long.
        expired = []

        for storm_id, state in (
            active_tracks.items()
        ):

            gap = (
                timestamp
                - state["timestamp"]
            )

            if gap > max_gap_dt:
                expired.append(
                    storm_id
                )

        for storm_id in expired:
            del active_tracks[
                storm_id
            ]

        if len(active_tracks) == 0:

            for idx in current_idx:

                clusters.at[
                    idx,
                    "storm_id",
                ] = next_storm_id

                active_tracks[
                    next_storm_id
                ] = {
                    "timestamp": timestamp,
                    "lat": float(
                        clusters.at[
                            idx,
                            "cluster_lat",
                        ]
                    ),
                    "lon": float(
                        clusters.at[
                            idx,
                            "cluster_lon",
                        ]
                    ),
                }

                next_storm_id += 1

            continue

        track_ids = list(
            active_tracks.keys()
        )

        n_tracks = len(
            track_ids
        )

        n_clusters = len(
            current_idx
        )

        cost = np.full(
            (
                n_tracks,
                n_clusters,
            ),
            np.inf,
            dtype=float,
        )

        for i, storm_id in enumerate(
            track_ids
        ):

            state = active_tracks[
                storm_id
            ]

            time_gap = (
                timestamp
                - state["timestamp"]
            )

            gap_windows = (
                time_gap
                / base_dt
            )

            # Allow a proportionally larger displacement
            # if one temporal window was missed.
            allowed_distance = (
                max_link_distance_km
                * float(gap_windows)
            )

            for j, idx in enumerate(
                current_idx
            ):

                distance = haversine_km(
                    state["lat"],
                    state["lon"],
                    float(
                        clusters.at[
                            idx,
                            "cluster_lat",
                        ]
                    ),
                    float(
                        clusters.at[
                            idx,
                            "cluster_lon",
                        ]
                    ),
                )

                if (
                    distance
                    <= allowed_distance
                ):
                    cost[
                        i,
                        j,
                    ] = distance

        matched_clusters = set()

        if np.isfinite(
            cost
        ).any():

            # scipy cannot use an all-inf matrix directly in all cases.
            large_cost = (
                max_link_distance_km
                * 1000.0
            )

            assignment_cost = np.where(
                np.isfinite(cost),
                cost,
                large_cost,
            )

            row_ind, col_ind = (
                linear_sum_assignment(
                    assignment_cost
                )
            )

            for row, col in zip(
                row_ind,
                col_ind,
            ):

                if not np.isfinite(
                    cost[row, col]
                ):
                    continue

                storm_id = track_ids[
                    row
                ]

                idx = current_idx[
                    col
                ]

                clusters.at[
                    idx,
                    "storm_id",
                ] = storm_id

                active_tracks[
                    storm_id
                ] = {
                    "timestamp": timestamp,
                    "lat": float(
                        clusters.at[
                            idx,
                            "cluster_lat",
                        ]
                    ),
                    "lon": float(
                        clusters.at[
                            idx,
                            "cluster_lon",
                        ]
                    ),
                }

                matched_clusters.add(
                    idx
                )

        # Any cluster not matched starts a new storm track.
        for idx in current_idx:

            if idx in matched_clusters:
                continue

            clusters.at[
                idx,
                "storm_id",
            ] = next_storm_id

            active_tracks[
                next_storm_id
            ] = {
                "timestamp": timestamp,
                "lat": float(
                    clusters.at[
                        idx,
                        "cluster_lat",
                    ]
                ),
                "lon": float(
                    clusters.at[
                        idx,
                        "cluster_lon",
                    ]
                ),
            }

            next_storm_id += 1

    clusters["storm_id"] = (
        clusters["storm_id"]
        .astype("int64")
    )

    return clusters


# =====================================================================
# TRACK CHARACTERIZATION
# =====================================================================

def summarize_tracks(
    tracked_clusters: pd.DataFrame,
    time_bin_minutes: int,
) -> pd.DataFrame:
    """
    Generate one row per tracked electrical storm.
    """

    if tracked_clusters.empty:
        return pd.DataFrame()

    records = []

    for storm_id, storm in (
        tracked_clusters.groupby(
            "storm_id"
        )
    ):

        storm = storm.sort_values(
            "timestamp"
        )

        start_time = storm[
            "timestamp"
        ].min()

        end_time = storm[
            "timestamp"
        ].max()

        n_windows = storm[
            "timestamp"
        ].nunique()

        # Observed-span duration.
        # + one bin because a storm seen in one 10-min window has
        # a 10-minute observed duration rather than zero.
        duration_min = (
            (
                end_time
                - start_time
            ).total_seconds()
            / 60.0
            + time_bin_minutes
        )

        total_flashes = int(
            storm[
                "cluster_flash_count"
            ].sum()
        )

        max_flashes = int(
            storm[
                "cluster_flash_count"
            ].max()
        )

        mean_flashes = float(
            storm[
                "cluster_flash_count"
            ].mean()
        )

        peak_idx = (
            storm[
                "cluster_flash_count"
            ]
            .idxmax()
        )

        peak = (
            tracked_clusters.loc[
                peak_idx
            ]
        )

        # Track centroid weighted by flash count.
        counts = storm[
            "cluster_flash_count"
        ].to_numpy(
            dtype=float
        )

        mean_lat = float(
            np.average(
                storm[
                    "cluster_lat"
                ],
                weights=counts,
            )
        )

        mean_lon = float(
            np.average(
                storm[
                    "cluster_lon"
                ],
                weights=counts,
            )
        )

        records.append(
            {
                "storm_id": int(
                    storm_id
                ),
                "start_time": start_time,
                "end_time": end_time,
                "duration_min": float(
                    duration_min
                ),
                "n_active_windows": int(
                    n_windows
                ),
                "total_flashes": total_flashes,
                "max_flashes_10min": max_flashes,
                "mean_flashes_10min": mean_flashes,
                "total_energy": float(
                    storm[
                        "cluster_energy_sum"
                    ].sum()
                ),
                "peak_time": peak[
                    "timestamp"
                ],
                "peak_lat": float(
                    peak[
                        "cluster_lat"
                    ]
                ),
                "peak_lon": float(
                    peak[
                        "cluster_lon"
                    ]
                ),
                "mean_lat": mean_lat,
                "mean_lon": mean_lon,
                "first_date": (
                    start_time.floor(
                        "D"
                    )
                ),
            }
        )

    tracks = pd.DataFrame(
        records
    )

    return (
        tracks
        .sort_values(
            [
                "start_time",
                "max_flashes_10min",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# =====================================================================
# TRACK CLASSIFICATION
# =====================================================================

def classify_tracks(
    tracks: pd.DataFrame,
    duration_quantile: float,
    intensity_quantile: float,
) -> Tuple[
    pd.DataFrame,
    float,
    float,
]:
    """
    Classify storms on two independent axes:

      persistence -> duration_min
      intensity   -> max_flashes_10min

    Thresholds are calculated globally across the complete requested
    period, not separately for each month.
    """

    if tracks.empty:
        return (
            tracks.copy(),
            np.nan,
            np.nan,
        )

    tracks = tracks.copy()

    duration_threshold = float(
        tracks[
            "duration_min"
        ].quantile(
            duration_quantile
        )
    )

    intensity_threshold = float(
        tracks[
            "max_flashes_10min"
        ].quantile(
            intensity_quantile
        )
    )

    tracks["is_long"] = (
        tracks[
            "duration_min"
        ]
        >= duration_threshold
    )

    tracks["is_intense"] = (
        tracks[
            "max_flashes_10min"
        ]
        >= intensity_threshold
    )

    def classify(row):

        if (
            row["is_long"]
            and row["is_intense"]
        ):
            return "long_intense"

        if (
            row["is_long"]
            and not row["is_intense"]
        ):
            return "long_moderate"

        if (
            not row["is_long"]
            and row["is_intense"]
        ):
            return "short_intense"

        return "short_lower_activity"

    tracks["storm_type"] = (
        tracks.apply(
            classify,
            axis=1,
        )
    )

    return (
        tracks,
        duration_threshold,
        intensity_threshold,
    )


# =====================================================================
# DAILY RADAR-CANDIDATE DATABASE
# =====================================================================

def build_daily_database(
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per calendar day containing at least one tracked
    electrical storm.

    A storm spanning midnight can therefore contribute to both dates
    only if a future implementation explicitly splits tracks by date.

    Here the day is associated with the storm start date.
    """

    if tracks.empty:
        return pd.DataFrame()

    tracks = tracks.copy()

    tracks["date"] = (
        tracks[
            "start_time"
        ]
        .dt.floor("D")
    )

    type_counts = (
        tracks
        .pivot_table(
            index="date",
            columns="storm_type",
            values="storm_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    expected_types = [
        "short_intense",
        "long_moderate",
        "long_intense",
        "short_lower_activity",
    ]

    for col in expected_types:

        if col not in type_counts:
            type_counts[col] = 0

    daily = (
        tracks.groupby(
            "date",
            as_index=False,
        )
        .agg(
            n_storm_tracks=(
                "storm_id",
                "nunique",
            ),
            max_track_duration_min=(
                "duration_min",
                "max",
            ),
            max_track_peak_flashes_10min=(
                "max_flashes_10min",
                "max",
            ),
            max_track_total_flashes=(
                "total_flashes",
                "max",
            ),
            total_daily_tracked_flashes=(
                "total_flashes",
                "sum",
            ),
        )
    )

    daily = daily.merge(
        type_counts,
        on="date",
        how="left",
    )

    # -------------------------------------------------------------
    # Representative storm for radar inspection
    #
    # Priority:
    #   1. largest peak electrical activity
    #   2. largest total flash count
    #   3. longest duration
    # -------------------------------------------------------------

    representative = (
        tracks
        .sort_values(
            [
                "date",
                "max_flashes_10min",
                "total_flashes",
                "duration_min",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset="date",
            keep="first",
        )
        [
            [
                "date",
                "storm_id",
                "storm_type",
                "start_time",
                "end_time",
                "peak_time",
                "peak_lat",
                "peak_lon",
            ]
        ]
        .rename(
            columns={
                "storm_id":
                    "representative_storm_id",
                "storm_type":
                    "representative_storm_type",
                "start_time":
                    "representative_start_time",
                "end_time":
                    "representative_end_time",
                "peak_time":
                    "representative_peak_time",
                "peak_lat":
                    "representative_peak_lat",
                "peak_lon":
                    "representative_peak_lon",
            }
        )
    )

    daily = daily.merge(
        representative,
        on="date",
        how="left",
    )

    # -------------------------------------------------------------
    # Radar download recommendation
    #
    # Keep a day if at least one storm is:
    #   - short and intense
    #   - long and moderate
    #   - long and intense
    #
    # Days containing only low-activity short tracks are not selected.
    # -------------------------------------------------------------

    daily["radar_candidate"] = (
        (
            daily["short_intense"]
            + daily["long_moderate"]
            + daily["long_intense"]
        )
        > 0
    )

    def candidate_reason(row):

        reasons = []

        if row["short_intense"] > 0:
            reasons.append(
                "short_intense"
            )

        if row["long_moderate"] > 0:
            reasons.append(
                "long_moderate"
            )

        if row["long_intense"] > 0:
            reasons.append(
                "long_intense"
            )

        if not reasons:
            return (
                "short_lower_activity_only"
            )

        return "+".join(
            reasons
        )

    daily["selection_reason"] = (
        daily.apply(
            candidate_reason,
            axis=1,
        )
    )

    return (
        daily
        .sort_values("date")
        .reset_index(drop=True)
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        required=True,
        help="Start date YYYY-MM-DD",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date YYYY-MM-DD",
    )

    parser.add_argument(
        "--satellite",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--out_dir",
        required=True,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--time_bin_minutes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--cluster_radius_km",
        type=float,
        default=15.0,
        help=(
            "DBSCAN spatial radius for individual "
            "10-min electrical clusters."
        ),
    )

    parser.add_argument(
        "--min_flashes_cluster",
        type=int,
        default=5,
        help=(
            "Minimum number of flashes required to "
            "form a spatial cluster."
        ),
    )

    parser.add_argument(
        "--max_link_distance_km",
        type=float,
        default=30.0,
        help=(
            "Maximum centroid displacement between "
            "consecutive time windows."
        ),
    )

    parser.add_argument(
        "--max_gap_windows",
        type=int,
        default=1,
        help=(
            "Maximum number of missing temporal windows "
            "allowed inside a storm track."
        ),
    )

    parser.add_argument(
        "--duration_quantile",
        type=float,
        default=0.75,
        help=(
            "Quantile defining a long-lived electrical "
            "storm track."
        ),
    )

    parser.add_argument(
        "--intensity_quantile",
        type=float,
        default=0.75,
        help=(
            "Quantile of max_flashes_10min defining "
            "an intense electrical storm track."
        ),
    )

    args = parser.parse_args()

    start_dt = datetime.strptime(
        args.start,
        "%Y-%m-%d",
    )

    end_dt = (
        datetime.strptime(
            args.end,
            "%Y-%m-%d",
        )
        + timedelta(days=1)
        - timedelta(seconds=1)
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "GLM STORM TRACK DATABASE"
    )

    print(
        "=" * 72
    )

    print(
        f"Period: {args.start} -> {args.end}"
    )

    print(
        f"Temporal window: "
        f"{args.time_bin_minutes} min"
    )

    print(
        f"DBSCAN radius: "
        f"{args.cluster_radius_km:.1f} km"
    )

    print(
        f"Minimum flashes per cluster: "
        f"{args.min_flashes_cluster}"
    )

    print(
        f"Track linking distance: "
        f"{args.max_link_distance_km:.1f} km/window"
    )

    print(
        f"Allowed missing windows: "
        f"{args.max_gap_windows}"
    )

    # =============================================================
    # FILE DISCOVERY
    # =============================================================

    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={
            "region_name": "us-east-1"
        },
    )

    current = start_dt.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    hours = []
    while current <= end_dt:
        hours.append(current)
        current += timedelta(hours=1)

    all_files = []

    def _list_single_hour(h_dt: datetime) -> List[str]:
        raw_files = list_files_for_hour(
            fs=fs,
            satellite=args.satellite,
            year=h_dt.strftime("%Y"),
            doy=ymd_to_doy(h_dt),
            hour=h_dt.hour,
        )
        matched = []
        for file_path in raw_files:
            file_dt = parse_file_start_dt(file_path)
            if file_dt and start_dt <= file_dt <= end_dt:
                matched.append(file_path)
        return matched

    max_threads = min(32, max(args.num_workers * 2, 16))
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(_list_single_hour, h)
            for h in hours
        ]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Listing GLM hours",
            unit="hour",
        ):
            all_files.extend(f.result())

    all_files = sorted(
        set(all_files)
    )

    print(
        f"\nFound {len(all_files):,} GLM files."
    )

    if not all_files:
        sys.exit(
            "No GLM files found."
        )

    # =============================================================
    # READ NATIVE FLASHES
    # =============================================================

    flash_results = []

    failed_files = 0

    print(
        "\nReading native GLM flashes..."
    )

    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(
            CORDOBA_BBOX,
        ),
    ) as pool:

        iterator = pool.imap_unordered(
            process_file_worker,
            all_files,
            chunksize=100,
        )

        for status, frame in tqdm(
            iterator,
            total=len(all_files),
            desc="GLM files",
            unit="file",
        ):

            if status == "failed":
                failed_files += 1

            elif (
                frame is not None
                and not frame.empty
            ):
                flash_results.append(
                    frame
                )

    print(
        f"\nFailed files: "
        f"{failed_files:,}"
    )

    if not flash_results:
        sys.exit(
            "No GLM flashes found inside domain."
        )

    flashes = pd.concat(
        flash_results,
        ignore_index=True,
    )

    flashes = (
        flashes
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Native flashes in domain: "
        f"{len(flashes):,}"
    )

    # =============================================================
    # SPATIAL CLUSTERS
    # =============================================================

    print(
        "\nDetecting 10-min electrical clusters..."
    )

    clusters = detect_spatial_clusters(
        flashes=flashes,
        time_bin_minutes=(
            args.time_bin_minutes
        ),
        cluster_radius_km=(
            args.cluster_radius_km
        ),
        min_flashes_cluster=(
            args.min_flashes_cluster
        ),
    )

    if clusters.empty:
        sys.exit(
            "No spatial electrical clusters detected."
        )

    print(
        f"Electrical clusters: "
        f"{len(clusters):,}"
    )

    # =============================================================
    # TEMPORAL TRACKING
    # =============================================================

    print(
        "\nLinking electrical clusters through time..."
    )

    tracked_clusters = (
        build_storm_tracks(
            clusters=clusters,
            time_bin_minutes=(
                args.time_bin_minutes
            ),
            max_link_distance_km=(
                args.max_link_distance_km
            ),
            max_gap_windows=(
                args.max_gap_windows
            ),
        )
    )

    # =============================================================
    # TRACK SUMMARY
    # =============================================================

    tracks = summarize_tracks(
        tracked_clusters=(
            tracked_clusters
        ),
        time_bin_minutes=(
            args.time_bin_minutes
        ),
    )

    print(
        f"Storm tracks detected: "
        f"{len(tracks):,}"
    )

    # =============================================================
    # CLASSIFICATION
    # =============================================================

    (
        tracks,
        duration_threshold,
        intensity_threshold,
    ) = classify_tracks(
        tracks=tracks,
        duration_quantile=(
            args.duration_quantile
        ),
        intensity_quantile=(
            args.intensity_quantile
        ),
    )

    print()
    print(
        "Classification thresholds:"
    )

    print(
        f"  Long duration threshold "
        f"(P{args.duration_quantile * 100:.0f}): "
        f"{duration_threshold:.1f} min"
    )

    print(
        f"  High intensity threshold "
        f"(P{args.intensity_quantile * 100:.0f}): "
        f"{intensity_threshold:.1f} flashes/10 min"
    )

    print()
    print(
        "Storm types:"
    )

    print(
        tracks[
            "storm_type"
        ]
        .value_counts()
        .to_string()
    )

    # =============================================================
    # DAILY RADAR DATABASE
    # =============================================================

    daily = build_daily_database(
        tracks
    )

    candidates = daily[
        daily[
            "radar_candidate"
        ]
    ].copy()

    # =============================================================
    # OUTPUTS
    # =============================================================

    tracked_clusters.to_parquet(
        out_dir
        / "glm_tracked_clusters.parquet",
        index=False,
    )

    tracks.to_parquet(
        out_dir
        / "glm_storm_tracks.parquet",
        index=False,
    )

    tracks.to_csv(
        out_dir
        / "glm_storm_tracks.csv",
        index=False,
    )

    daily.to_csv(
        out_dir
        / "glm_all_active_days.csv",
        index=False,
    )

    candidates.to_csv(
        out_dir
        / "glm_radar_candidate_days.csv",
        index=False,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL DATABASE"
    )

    print(
        "=" * 72
    )

    print(
        f"Tracked storms: "
        f"{len(tracks):,}"
    )

    print(
        f"Days containing tracked storms: "
        f"{len(daily):,}"
    )

    print(
        f"Radar candidate days: "
        f"{len(candidates):,}"
    )

    if len(daily) > 0:

        print(
            f"Selected fraction: "
            f"{len(candidates) / len(daily) * 100:.1f}%"
        )

    print()
    print(
        f"Storm catalogue: "
        f"{out_dir / 'glm_storm_tracks.csv'}"
    )

    print(
        f"Radar download list: "
        f"{out_dir / 'glm_radar_candidate_days.csv'}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()