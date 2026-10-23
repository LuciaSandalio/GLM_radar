#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidate_2021.py

Consolidates the 12 monthly outputs of 2021 into a single unified annual database:
- Merges clusters and tracks with globally unique IDs.
- Computes global annual thresholds (P75 duration, P75 intensity).
- Re-classifies storm types and builds the annual radar candidate days list.
- Exports to CSV and Parquet in output_2021_anual/.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output_2021_anual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def candidate_reason(row):
    reasons = []
    if row["short_intense"] > 0:
        reasons.append("short_intense")
    if row["long_moderate"] > 0:
        reasons.append("long_moderate")
    if row["long_intense"] > 0:
        reasons.append("long_intense")
    if not reasons:
        return "short_lower_activity_only"
    return "+".join(reasons)

def main():
    print("Consolidating 2021 monthly outputs...")
    all_clusters = []
    all_tracks = []
    
    storm_id_offset = 0
    cluster_id_offset = 0
    
    for month in range(1, 13):
        m_str = f"{month:02d}"
        month_dir = BASE_DIR / f"output_2021_{m_str}"
        clusters_file = month_dir / "glm_tracked_clusters.parquet"
        tracks_file = month_dir / "glm_storm_tracks.parquet"
        
        if not clusters_file.exists() or not tracks_file.exists():
            print(f"Warning: Missing files for {m_str}")
            continue
            
        c_df = pd.read_parquet(clusters_file)
        t_df = pd.read_parquet(tracks_file)
        
        # Remap IDs to ensure global uniqueness across the full year
        c_df["cluster_id"] = c_df["cluster_id"] + cluster_id_offset
        c_df["storm_id"] = c_df["storm_id"] + storm_id_offset
        t_df["storm_id"] = t_df["storm_id"] + storm_id_offset
        
        cluster_id_offset = int(c_df["cluster_id"].max()) + 1
        storm_id_offset = int(t_df["storm_id"].max()) + 1
        
        all_clusters.append(c_df)
        all_tracks.append(t_df)

    clusters = pd.concat(all_clusters, ignore_index=True)
    tracks = pd.concat(all_tracks, ignore_index=True)
    
    # Global annual classification thresholds (P75)
    duration_threshold = float(tracks["duration_min"].quantile(0.75))
    intensity_threshold = float(tracks["max_flashes_10min"].quantile(0.75))
    
    def classify_row(row):
        is_long = row["duration_min"] >= duration_threshold
        is_intense = row["max_flashes_10min"] >= intensity_threshold
        if is_long and is_intense:
            return "long_intense"
        elif is_long and not is_intense:
            return "long_moderate"
        elif not is_long and is_intense:
            return "short_intense"
        else:
            return "short_lower_activity"

    tracks["storm_type"] = tracks.apply(classify_row, axis=1)
    tracks["is_long"] = tracks["duration_min"] >= duration_threshold
    tracks["is_intense"] = tracks["max_flashes_10min"] >= intensity_threshold
    
    # Date column
    tracks["date"] = pd.to_datetime(tracks["start_time"], utc=True).dt.floor("D")
    
    # Build daily database exactly like glm_01.py
    type_counts = (
        tracks.pivot_table(
            index="date",
            columns="storm_type",
            values="storm_id",
            aggfunc="count",
            fill_value=0,
        ).reset_index()
    )
    for col in ["short_intense", "long_moderate", "long_intense", "short_lower_activity"]:
        if col not in type_counts.columns:
            type_counts[col] = 0

    daily = (
        tracks.groupby("date", as_index=False)
        .agg(
            n_storm_tracks=("storm_id", "nunique"),
            max_track_duration_min=("duration_min", "max"),
            max_track_peak_flashes_10min=("max_flashes_10min", "max"),
            max_track_total_flashes=("total_flashes", "max"),
            total_daily_tracked_flashes=("total_flashes", "sum"),
        )
    )
    daily = daily.merge(type_counts, on="date", how="left")
    
    representative = (
        tracks.sort_values(
            ["date", "max_flashes_10min", "total_flashes", "duration_min"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates(subset="date", keep="first")[
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
                "storm_id": "representative_storm_id",
                "storm_type": "representative_storm_type",
                "start_time": "representative_start_time",
                "end_time": "representative_end_time",
                "peak_time": "representative_peak_time",
                "peak_lat": "representative_peak_lat",
                "peak_lon": "representative_peak_lon",
            }
        )
    )
    
    daily = daily.merge(representative, on="date", how="left")
    daily["radar_candidate"] = (
        daily["short_intense"] + daily["long_moderate"] + daily["long_intense"]
    ) > 0
    daily["selection_reason"] = daily.apply(candidate_reason, axis=1)
    daily = daily.sort_values("date").reset_index(drop=True)
    candidates = daily[daily["radar_candidate"]].copy()

    # Save outputs
    clusters.to_parquet(OUT_DIR / "glm_tracked_clusters.parquet", index=False)
    tracks.to_parquet(OUT_DIR / "glm_storm_tracks.parquet", index=False)
    tracks.to_csv(OUT_DIR / "glm_storm_tracks.csv", index=False)
    daily.to_csv(OUT_DIR / "glm_all_active_days.csv", index=False)
    candidates.to_csv(OUT_DIR / "glm_radar_candidate_days.csv", index=False)
    
    print("=" * 72)
    print("2021 ANNUAL CONSOLIDATION COMPLETE")
    print("=" * 72)
    print(f"Total tracked storms: {len(tracks):,}")
    print(f"Total clusters: {len(clusters):,}")
    print(f"Annual Long duration threshold (P75): {duration_threshold:.1f} min")
    print(f"Annual High intensity threshold (P75): {intensity_threshold:.1f} flashes/10 min")
    print("\nStorm types (Annual):")
    print(tracks["storm_type"].value_counts().to_string())
    print("-" * 72)
    print(f"Total active days: {len(daily)}")
    print(f"Radar candidate days: {len(candidates)} ({len(candidates)/len(daily)*100:.1f}%)")
    print(f"Saved into: {OUT_DIR}")

if __name__ == "__main__":
    main()

