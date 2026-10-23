#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_monthly_2021.py

Batch runner to process GLM-16 data month by month for the remainder of 2021.
Saves intermediate monthly results into separate output directories, ensuring
progress is safe from interruptions.
"""

import calendar
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON_EXE = sys.executable
SCRIPT_PATH = Path(__file__).resolve().parent / "glm_01.py"
BASE_DIR = Path(__file__).resolve().parent

MONTHS_2021 = [
    (2021, m) for m in range(2, 13)
]

def main():
    print("=" * 72)
    print("BATCH PROCESSING GLM 2021 (MONTH BY MONTH: FEB -> DEC)")
    print("=" * 72)
    start_total = time.time()

    for year, month in MONTHS_2021:
        month_str = f"{month:02d}"
        last_day = calendar.monthrange(year, month)[1]
        start_date = f"{year}-{month_str}-01"
        end_date = f"{year}-{month_str}-{last_day:02d}"
        out_dir = BASE_DIR / f"output_{year}_{month_str}"

        done_marker = out_dir / "glm_radar_candidate_days.csv"
        if done_marker.exists():
            print(f"\n[SKIP] Month {year}-{month_str} already completed. Output: {out_dir}")
            continue

        print("\n" + "#" * 72)
        print(f"STARTING MONTH: {year}-{month_str} ({start_date} to {end_date})")
        print(f"Output directory: {out_dir}")
        print("#" * 72)

        cmd = [
            PYTHON_EXE,
            str(SCRIPT_PATH),
            "--start", start_date,
            "--end", end_date,
            "--out_dir", str(out_dir),
            "--num_workers", "16",
        ]

        t0 = time.time()
        res = subprocess.run(cmd)

        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"\n[ERROR] Failed processing {year}-{month_str} (exit code: {res.returncode})")
            sys.exit(res.returncode)

        print(f"\n[SUCCESS] Month {year}-{month_str} finished in {elapsed / 60:.1f} minutes.")

    total_min = (time.time() - start_total) / 60
    print("\n" + "=" * 72)
    print(f"ALL MONTHS COMPLETED SUCCESSFULLY in {total_min:.1f} minutes.")
    print("=" * 72)

if __name__ == "__main__":
    main()

