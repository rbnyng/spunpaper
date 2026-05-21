#!/usr/bin/env python3
"""End-to-end demo of the modeling pipeline.

`sample_data/` has
  - ECM_richness_europe.csv : the full European ECM richness table.
  - representations/*.npy   : 100 real Tessera per-sample embedding patches
                              (shape 3 x 3 x 128, float32).

This script filters the CSV to the 100 samples that have matching
representations and runs modeling/spun_train_patch.py on them with
satellite embeddings as the only feature source. It is a pipeline test only.
With only 100 samples the metrics are not scientifically meaningful, but
every stage of the downstream training code (loader, PCA, RF, metrics,
plotting) is exercised end-to-end.

Usage:
    python demo/run_demo.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = REPO_ROOT / "sample_data"
MODELING_DIR = REPO_ROOT / "modeling"

SOURCE_CSV = SAMPLE_DATA / "ECM_richness_europe.csv"
REPR_DIR = SAMPLE_DATA / "representations"

DEMO_OUT = REPO_ROOT / "demo" / "demo_run"
SUBSET_CSV = DEMO_OUT / "ecm_subset.csv"
RESULTS_DIR = DEMO_OUT / "results"

CLIMATE_DIR = DEMO_OUT / "climate_unused"
CLIMATE_CACHE = DEMO_OUT / "climate_cache_unused"

def make_subset_csv():
    repr_ids = {p.stem.split("_")[1] for p in REPR_DIR.glob("*.npy")}
    df = pd.read_csv(SOURCE_CSV)
    df["sample_id"] = df["sample_id"].astype(str)
    sub = df[df["sample_id"].isin(repr_ids)].copy()
    sub = sub.dropna(subset=["latitude", "longitude", "rarefied"])
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBSET_CSV, index=False)
    print(f"  {len(sub)} samples written to {SUBSET_CSV}")
    print(f"  lat {sub['latitude'].min():.2f}..{sub['latitude'].max():.2f}, "
          f"lon {sub['longitude'].min():.2f}..{sub['longitude'].max():.2f}")
    print(f"  rarefied mean {sub['rarefied'].mean():.2f}, "
          f"range {sub['rarefied'].min():.1f}..{sub['rarefied'].max():.1f}")


def run_trainer():
    CLIMATE_DIR.mkdir(parents=True, exist_ok=True)
    CLIMATE_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    unused_wc = DEMO_OUT / "worldcover_unused.csv"
    cmd = [
        sys.executable, "spun_train_patch.py",
        "--model", "rf",
        "--num_runs", "3",
        "--use-satellite",
        "--no-use-climate",
        "--no-use-soil",
        "--no-use-worldcover",
        "--dim_reduction", "pca",
        "--dim_reduction_components", "16",
        "--biodiversity_csvs", str(SUBSET_CSV),
        "--representations_dir", str(REPR_DIR),
        "--climate_data_dir", str(CLIMATE_DIR),
        "--climate_cache_dir", str(CLIMATE_CACHE),
        "--soil_cache_dir", str(DEMO_OUT / "soil_unused"),
        "--worldcover_path", str(unused_wc),
        "--results_dir", str(RESULTS_DIR),
    ]
    print("  exec:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(MODELING_DIR))

def main():
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    make_subset_csv()
    run_trainer()
    print("Demo complete")
    print(f"  results under: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
