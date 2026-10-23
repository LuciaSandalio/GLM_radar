#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_radar_geometry_and_priorities.py

Implements Phases 1, 2, 4, and 5 of the RMA-1 Radar Database Construction Plan:
1. Adds RMA-1 geometry (distances, masks within 120km, 240km, 480km, enters_*, fully_within_*)
   to all 42,915 clusters and 7,925 tracks from the 2021 annual catalog.
2. Calculates radar domain coverage statistics (N_120, N_240, N_480).
3. Assigns independent Radar Priorities:
     - Priority A: long_intense within radar domain
     - Priority B: short_intense or long_moderate within radar domain
     - Priority C: short_lower_activity within radar domain
4. Corrects for midnight-crossing events using a +/- 1 hour temporal buffer [t_start - 1h, t_end + 1h].
5. Exports:
     - output_2021_anual/glm_tracked_clusters_rma1.parquet
     - output_2021_anual/glm_storm_tracks_rma1.parquet
     - output_2021_anual/glm_storm_tracks_rma1.csv
     - output_2021_anual/radar_download_days.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ANUAL_DIR = BASE_DIR / "output_2021_anual"

# Coordinates of RMA-1 (Ciudad Universitaria, UNC, Córdoba)
RMA1_LAT = -31.4413
RMA1_LON = -64.1919
EARTH_RADIUS_KM = 6371.0088

BUFFER_HOURS = 1.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Computes great-circle distance in km."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def main():
    print("=" * 72)
    print("RMA-1 RADAR GEOMETRY & PRIORITY SELECTION (2021)")
    print("=" * 72)

    clusters_path = ANUAL_DIR / "glm_tracked_clusters.parquet"
    tracks_path = ANUAL_DIR / "glm_storm_tracks.parquet"

    if not clusters_path.exists() or not tracks_path.exists():
        raise FileNotFoundError(
            f"Required annual catalog files not found in {ANUAL_DIR}"
        )

    print("Loading 2021 annual clusters and tracks...")
    clusters = pd.read_parquet(clusters_path)
    tracks = pd.read_parquet(tracks_path)

    # -------------------------------------------------------------
    # PHASE 1: CLUSTER-LEVEL GEOMETRY
    # -------------------------------------------------------------
    print("Calculating RMA-1 geometric distances for each cluster...")
    clusters["distance_to_radar_km"] = haversine_km(
        clusters["cluster_lat"], clusters["cluster_lon"], RMA1_LAT, RMA1_LON
    )
    clusters["within_120km"] = clusters["distance_to_radar_km"] <= 120.0
    clusters["within_240km"] = clusters["distance_to_radar_km"] <= 240.0
    clusters["within_480km"] = clusters["distance_to_radar_km"] <= 480.0

    # -------------------------------------------------------------
    # PHASE 1 & 2: TRACK-LEVEL GEOMETRY
    # -------------------------------------------------------------
    print("Aggregating geometric properties for each storm track...")
    track_geom = (
        clusters.groupby("storm_id")
        .agg(
            min_distance_to_radar_km=("distance_to_radar_km", "min"),
            max_distance_to_radar_km=("distance_to_radar_km", "max"),
            mean_distance_to_radar_km=("distance_to_radar_km", "mean"),
            fraction_clusters_within_120km=("within_120km", "mean"),
            fraction_clusters_within_240km=("within_240km", "mean"),
            fraction_clusters_within_480km=("within_480km", "mean"),
            enters_120km=("within_120km", "any"),
            enters_240km=("within_240km", "any"),
            enters_480km=("within_480km", "any"),
            fully_within_120km=("within_120km", "all"),
            fully_within_240km=("within_240km", "all"),
        )
        .reset_index()
    )

    tracks = tracks.merge(track_geom, on="storm_id", how="left")

    # -------------------------------------------------------------
    # PHASE 4: RADAR EVENT PRIORITIZATION
    # -------------------------------------------------------------
    # Priority definition:
    #   A: long_intense within chosen radar domain (default: 240 km)
    #   B: short_intense or long_moderate within radar domain
    #   C: short_lower_activity within radar domain
    #   None: outside radar domain
    def assign_track_priority(row, max_rad_km=240.0):
        if not row["enters_240km"]:
            return "OUTSIDE_RADAR"
        st = row["storm_type"]
        if st == "long_intense":
            return "Priority_A"
        elif st in ["short_intense", "long_moderate"]:
            return "Priority_B"
        else:
            return "Priority_C"

    tracks["radar_priority_240km"] = tracks.apply(
        lambda r: assign_track_priority(r, 240.0), axis=1
    )
    tracks["radar_priority_120km"] = tracks.apply(
        lambda r: (
            "Priority_A"
            if r["enters_120km"] and r["storm_type"] == "long_intense"
            else (
                "Priority_B"
                if r["enters_120km"]
                and r["storm_type"] in ["short_intense", "long_moderate"]
                else (
                    "Priority_C"
                    if r["enters_120km"]
                    else "OUTSIDE_RADAR"
                )
            )
        ),
        axis=1,
    )

    # -------------------------------------------------------------
    # PHASE 5: MIDNIGHT CROSSING & RADAR DOWNLOAD DAYS
    # -------------------------------------------------------------
    print("Computing midnight-crossing download windows (+/- 1 hour buffer)...")
    tracks["start_time"] = pd.to_datetime(tracks["start_time"], utc=True)
    tracks["end_time"] = pd.to_datetime(tracks["end_time"], utc=True)

    # Daily aggregation with midnight expansion
    calendar_days_records = {}

    for _, row in tracks.iterrows():
        prio_240 = row["radar_priority_240km"]
        if prio_240 == "OUTSIDE_RADAR":
            continue

        storm_id = int(row["storm_id"])
        st_type = row["storm_type"]

        t_start_buffered = row["start_time"] - pd.Timedelta(hours=BUFFER_HOURS)
        t_end_buffered = row["end_time"] + pd.Timedelta(hours=BUFFER_HOURS)

        day_cursor = t_start_buffered.floor("D")
        day_end = t_end_buffered.floor("D")

        while day_cursor <= day_end:
            day_str = day_cursor.strftime("%Y-%m-%d")

            if day_str not in calendar_days_records:
                calendar_days_records[day_str] = {
                    "date": day_str,
                    "priority_rank": 3,  # 1: A, 2: B, 3: C
                    "highest_priority": prio_240,
                    "storm_ids": [],
                    "storm_types": set(),
                    "enters_120km": False,
                    "enters_240km": True,
                    "earliest_required_time": t_start_buffered,
                    "latest_required_time": t_end_buffered,
                    "max_flashes_10min": int(row["max_flashes_10min"]),
                    "max_duration_min": float(row["duration_min"]),
                    "total_flashes_sum": int(row["total_flashes"]),
                }

            rec = calendar_days_records[day_str]
            rec["storm_ids"].append(storm_id)
            rec["storm_types"].add(st_type)

            if row["enters_120km"]:
                rec["enters_120km"] = True

            # Update highest priority
            curr_rank = (
                1
                if prio_240 == "Priority_A"
                else (2 if prio_240 == "Priority_B" else 3)
            )
            if curr_rank < rec["priority_rank"]:
                rec["priority_rank"] = curr_rank
                rec["highest_priority"] = prio_240

            # Expand time windows
            if t_start_buffered < rec["earliest_required_time"]:
                rec["earliest_required_time"] = t_start_buffered
            if t_end_buffered > rec["latest_required_time"]:
                rec["latest_required_time"] = t_end_buffered

            # Update descriptors
            rec["max_flashes_10min"] = max(
                rec["max_flashes_10min"], int(row["max_flashes_10min"])
            )
            rec["max_duration_min"] = max(
                rec["max_duration_min"], float(row["duration_min"])
            )
            rec["total_flashes_sum"] += int(row["total_flashes"])

            day_cursor += pd.Timedelta(days=1)

    # Convert to DataFrame
    download_rows = []
    for d, rec in sorted(calendar_days_records.items()):
        download_rows.append(
            {
                "date": rec["date"],
                "highest_priority": rec["highest_priority"],
                "n_storms_in_day": len(rec["storm_ids"]),
                "storm_ids": ",".join(map(str, sorted(set(rec["storm_ids"])))),
                "storm_types_present": "+".join(sorted(rec["storm_types"])),
                "enters_120km": rec["enters_120km"],
                "enters_240km": rec["enters_240km"],
                "required_window_start_utc": rec[
                    "earliest_required_time"
                ].strftime("%Y-%m-%d %H:%M:%S"),
                "required_window_end_utc": rec[
                    "latest_required_time"
                ].strftime("%Y-%m-%d %H:%M:%S"),
                "max_flashes_10min": rec["max_flashes_10min"],
                "max_duration_min": rec["max_duration_min"],
                "total_daily_tracked_flashes": rec["total_flashes_sum"],
            }
        )

    download_days_df = pd.DataFrame(download_rows)

    # -------------------------------------------------------------
    # EXPORT RESULTS
    # -------------------------------------------------------------
    print("\nExporting enriched catalogs...")
    clusters.to_parquet(
        ANUAL_DIR / "glm_tracked_clusters_rma1.parquet", index=False
    )
    tracks.to_parquet(ANUAL_DIR / "glm_storm_tracks_rma1.parquet", index=False)
    tracks.to_csv(ANUAL_DIR / "glm_storm_tracks_rma1.csv", index=False)
    download_days_df.to_csv(
        ANUAL_DIR / "radar_download_days.csv", index=False
    )

    print("=" * 72)
    print("SUMMARY OF GEOMETRIC & PRIORITY SCREENING")
    print("=" * 72)
    print(f"Total storm tracks in Córdoba: {len(tracks):,}")
    print(
        f"Tracks entering 120 km (N_120): {tracks['enters_120km'].sum():,} ({tracks['enters_120km'].mean()*100:.1f}%)"
    )
    print(
        f"Tracks entering 240 km (N_240): {tracks['enters_240km'].sum():,} ({tracks['enters_240km'].mean()*100:.1f}%)"
    )
    print(
        f"Tracks entering 480 km (N_480): {tracks['enters_480km'].sum():,} ({tracks['enters_480km'].mean()*100:.1f}%)"
    )
    print("-" * 72)
    print(
        "Radar Download Days (with +/- 1h buffer, 240 km domain):",
        len(download_days_df),
    )
    print(download_days_df["highest_priority"].value_counts().to_string())
    print("-" * 72)
    print("Saved files in:", ANUAL_DIR)
    print("  - glm_tracked_clusters_rma1.parquet")
    print("  - glm_storm_tracks_rma1.parquet")
    print("  - glm_storm_tracks_rma1.csv")
    print("  - radar_download_days.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()

