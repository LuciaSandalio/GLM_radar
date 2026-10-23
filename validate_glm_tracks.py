#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_glm_tracks.py

Visual validation tool for GLM storm tracking:
  - Samples ~25 stratified tracks:
      10 long_intense
      5 short_intense
      5 long_moderate
      5 short_lower_activity
  - For each track, generates a 2-panel diagnostic plot:
      1. Spatial map showing RMA-1 (center), 120km and 240km range rings,
         cluster trajectory with directional arrows and flash intensity.
      2. Time series of flash count N_flash(t) across successive 10-min windows.
  - Generates validation_plots/ and validation_summary.csv with kinematic metrics
    (speed km/h, max centroid jump km, missing window gaps).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
ANUAL_DIR = BASE_DIR / "output_2021_anual"
PLOTS_DIR = BASE_DIR / "validation_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RMA1_LAT = -31.4413
RMA1_LON = -64.1919
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def get_range_ring(center_lat, center_lon, radius_km, n_points=100):
    angles = np.linspace(0, 2 * np.pi, n_points)
    d_rad = radius_km / EARTH_RADIUS_KM
    lat_r = np.radians(center_lat)
    lon_r = np.radians(center_lon)

    ring_lats = np.arcsin(
        np.sin(lat_r) * np.cos(d_rad)
        + np.cos(lat_r) * np.sin(d_rad) * np.cos(angles)
    )
    ring_lons = lon_r + np.arctan2(
        np.sin(angles) * np.sin(d_rad) * np.cos(lat_r),
        np.cos(d_rad) - np.sin(lat_r) * np.sin(ring_lats),
    )
    return np.degrees(ring_lats), np.degrees(ring_lons)


def main():
    print("=" * 72)
    print("GLM TRACKING VISUAL & KINEMATIC VALIDATION")
    print("=" * 72)

    clusters = pd.read_parquet(ANUAL_DIR / "glm_tracked_clusters_2021.parquet")
    tracks = pd.read_parquet(ANUAL_DIR / "glm_storm_tracks_2021.parquet")

    # Sample definition
    sample_spec = {
        "long_intense": 10,
        "short_intense": 5,
        "long_moderate": 5,
        "short_lower_activity": 5,
    }

    selected_tracks = []
    np.random.seed(42)

    for st_type, count in sample_spec.items():
        sub = tracks[(tracks["storm_type"] == st_type) & (tracks["enters_240km"])]
        if len(sub) == 0:
            sub = tracks[tracks["storm_type"] == st_type]
        n_sample = min(count, len(sub))
        sampled = sub.sample(n=n_sample, random_state=42)
        selected_tracks.append(sampled)

    sample_df = pd.concat(selected_tracks, ignore_index=True)
    print(f"Sampled {len(sample_df)} tracks for detailed validation.")

    summary_records = []
    ring_120_lat, ring_120_lon = get_range_ring(RMA1_LAT, RMA1_LON, 120.0)
    ring_240_lat, ring_240_lon = get_range_ring(RMA1_LAT, RMA1_LON, 240.0)

    for idx, tr in sample_df.iterrows():
        sid = int(tr["storm_id"])
        st_type = tr["storm_type"]
        c_sub = clusters[clusters["storm_id"] == sid].sort_values("timestamp").reset_index(drop=True)

        if len(c_sub) == 0:
            continue

        c_sub["timestamp"] = pd.to_datetime(c_sub["timestamp"], utc=True)
        n_pts = len(c_sub)

        # Kinematic calculations
        jumps_km = []
        speed_kmh = []
        time_gaps_min = []

        for k in range(len(c_sub) - 1):
            d_km = haversine_km(
                c_sub.at[k, "cluster_lat"], c_sub.at[k, "cluster_lon"],
                c_sub.at[k + 1, "cluster_lat"], c_sub.at[k + 1, "cluster_lon"]
            )
            dt_min = (c_sub.at[k + 1, "timestamp"] - c_sub.at[k, "timestamp"]).total_seconds() / 60.0
            jumps_km.append(d_km)
            time_gaps_min.append(dt_min)
            if dt_min > 0:
                speed_kmh.append(d_km / (dt_min / 60.0))

        max_jump = float(np.max(jumps_km)) if jumps_km else 0.0
        mean_speed = float(np.mean(speed_kmh)) if speed_kmh else 0.0
        max_gap = float(np.max(time_gaps_min)) if time_gaps_min else 0.0

        summary_records.append({
            "storm_id": sid,
            "storm_type": st_type,
            "start_time": tr["start_time"],
            "end_time": tr["end_time"],
            "duration_min": tr["duration_min"],
            "total_flashes": tr["total_flashes"],
            "max_flashes_10min": tr["max_flashes_10min"],
            "n_clusters": n_pts,
            "max_jump_km": round(max_jump, 1),
            "mean_speed_kmh": round(mean_speed, 1),
            "max_time_gap_min": round(max_gap, 1),
            "min_distance_to_radar_km": round(tr["min_distance_to_radar_km"], 1),
        })

        # --- FIGURE GENERATION ---
        fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Spatial Map
        ax_map.plot(ring_240_lon, ring_240_lat, "k--", alpha=0.5, label="RMA-1 240 km")
        ax_map.plot(ring_120_lon, ring_120_lat, "b:", alpha=0.6, label="RMA-1 120 km")
        ax_map.plot(RMA1_LON, RMA1_LAT, "r^", markersize=10, label="RMA-1 Radar")

        sc = ax_map.scatter(
            c_sub["cluster_lon"], c_sub["cluster_lat"],
            c=c_sub["cluster_flash_count"], cmap="plasma",
            s=np.clip(c_sub["cluster_flash_count"] * 1.5, 30, 300),
            edgecolor="black", zorder=4
        )
        cbar = plt.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_label("Flashes / 10 min")

        # Trajectory line and arrows
        ax_map.plot(c_sub["cluster_lon"], c_sub["cluster_lat"], "gray", alpha=0.7, lw=1.5, zorder=3)
        for k in range(len(c_sub) - 1):
            dx = c_sub.at[k + 1, "cluster_lon"] - c_sub.at[k, "cluster_lon"]
            dy = c_sub.at[k + 1, "cluster_lat"] - c_sub.at[k, "cluster_lat"]
            ax_map.annotate(
                "",
                xy=(c_sub.at[k + 1, "cluster_lon"], c_sub.at[k + 1, "cluster_lat"]),
                xytext=(c_sub.at[k, "cluster_lon"], c_sub.at[k, "cluster_lat"]),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2, shrinkA=3, shrinkB=3),
                zorder=5,
            )

        # Start and End labels
        ax_map.text(c_sub["cluster_lon"].iloc[0], c_sub["cluster_lat"].iloc[0], " Start",
                    fontsize=9, fontweight="bold", color="green", zorder=6)
        ax_map.text(c_sub["cluster_lon"].iloc[-1], c_sub["cluster_lat"].iloc[-1], " End",
                    fontsize=9, fontweight="bold", color="black", zorder=6)

        t_start_str = str(tr["start_time"])[:16]
        t_end_str = str(tr["end_time"])[11:16]
        ax_map.set_title(f"Track {sid}: {st_type}\n({t_start_str} to {t_end_str} UTC)")
        ax_map.set_xlabel("Longitud (°)")
        ax_map.set_ylabel("Latitud (°)")
        ax_map.grid(True, linestyle=":", alpha=0.6)
        ax_map.legend(loc="upper right", fontsize=8)

        # Panel 2: Temporal Evolution
        ax_ts.plot(c_sub["timestamp"], c_sub["cluster_flash_count"], "o-", color="purple", lw=2, markersize=6)
        ax_ts.set_title(f"Evolución Eléctrica: {tr['total_flashes']} destellos totales")
        ax_ts.set_xlabel("Hora UTC")
        ax_ts.set_ylabel("Destellos / 10 min")
        ax_ts.grid(True, linestyle=":", alpha=0.6)
        ax_ts.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plot_fname = PLOTS_DIR / f"track_{sid:05d}_{st_type}.png"
        fig.savefig(plot_fname, dpi=120)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(BASE_DIR / "validation_summary.csv", index=False)
    print(f"\nGenerated {len(summary_records)} diagnostic plots in: {PLOTS_DIR}")
    print("Kinematic metrics saved in: validation_summary.csv")
    print("\nSample preview:")
    print(summary_df[["storm_id", "storm_type", "duration_min", "total_flashes", "max_jump_km", "mean_speed_kmh"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
