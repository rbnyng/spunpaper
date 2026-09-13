#!/usr/bin/env python
"""
python data_generation/prepare_fetch.py --workers 4 --year 2024 --patch-size 7
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths  # noqa: E402
from geotessera_embeddings import GeoTesseraPatchExtractor  # noqa: E402


def load_points(csv_paths, id_col, lat_col, lon_col):
    frames = []
    for p in csv_paths:
        d = pd.read_csv(p)
        missing = {id_col, lat_col, lon_col} - set(d.columns)
        if missing:
            raise SystemExit(f"{p} is missing columns: {sorted(missing)}")
        frames.append(d[[id_col, lat_col, lon_col]])
    df = pd.concat(frames, ignore_index=True).dropna(subset=[id_col, lat_col, lon_col])
    df[id_col] = df[id_col].astype(str)
    return df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)


def shard_by_tile(df, n_shards, lat_col="latitude", lon_col="longitude"):
    """Assign whole tiles to shards, balancing sample counts. Returns a list of DataFrames.

    Greedy longest-processing-time: place the busiest tile on the lightest shard.
    Exact balance is not the point -- keeping tiles intact is -- but this stops one
    worker inheriting every dense tile and running long after the others finish.
    """
    tiles = defaultdict(list)
    for i, lat, lon in df[[lat_col, lon_col]].itertuples(index=True):
        tiles[GeoTesseraPatchExtractor.tile_of(float(lon), float(lat))].append(i)
    loads = [0] * n_shards
    buckets = [[] for _ in range(n_shards)]
    for _, rows in sorted(tiles.items(), key=lambda kv: -len(kv[1])):
        k = min(range(n_shards), key=lambda j: loads[j])
        buckets[k].extend(rows)
        loads[k] += len(rows)
    return [df.loc[sorted(b)].reset_index(drop=True) for b in buckets], len(tiles)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords-csv", nargs="+", default=paths.ECM_CSVS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--output-dir", default=None,
                   help="Patch cache to write. Default: REPRESENTATIONS_DIR from config.env.")
    p.add_argument("--plan-dir", default=None,
                   help="Where to put shard CSVs, the filtered registry and logs. "
                        "Default: <output-dir>_fetchplan.")
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--lat-col", default="latitude")
    p.add_argument("--lon-col", default="longitude")
    p.add_argument("--no-registry", action="store_true",
                   help="Skip building the filtered registry (needs the full manifest "
                        "already in geotessera's cache, i.e. one fetch must have run).")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    out = Path(args.output_dir or paths.REPRESENTATIONS_DIR)
    plan = Path(args.plan_dir or f"{out}_fetchplan")
    plan.mkdir(parents=True, exist_ok=True)

    df = load_points(args.coords_csv, args.id_col, args.lat_col, args.lon_col)
    shards, n_tiles = shard_by_tile(df, args.workers, args.lat_col, args.lon_col)
    print(f"{len(df):,} unique samples across {n_tiles:,} tiles "
          f"-> {args.workers} shards of {[len(s) for s in shards]}")

    paths_out = []
    for i, s in enumerate(shards):
        f = plan / f"shard_{i}.csv"
        s.to_csv(f, index=False)
        paths_out.append(f)

    registry = plan / "registry"
    reg_arg = ""
    if not args.no_registry:
        try:
            from geotessera_embeddings import build_filtered_registry
            build_filtered_registry(
                zip(df[args.lon_col].astype(float), df[args.lat_col].astype(float)),
                registry, year=args.year)
            reg_arg = f" \\\n      --registry-dir {registry}"
            print(f"filtered registry written to {registry}")
        except Exception as e:                                    # noqa: BLE001
            print(f"could not build the filtered registry ({type(e).__name__}: {e}).")

    src = Path(__file__).resolve()
    print(f"\nRun these {args.workers} commands in parallel (each owns a disjoint set of "
          f"tiles, so\nthey never contend, and each is independently resumable):\n")
    for i, f in enumerate(paths_out):
        print(f"  {args.python} {src.parent / 'geotessera_embeddings.py'} \\\n"
              f"      --coords-csv {f} \\\n"
              f"      --output-dir {out} \\\n"
              f"      --embeddings-dir {plan / f'tiles_w{i}'} \\\n"
              f"      --year {args.year} --patch-size {args.patch_size} "
              f"--prune-tiles{reg_arg} \\\n"
              f"      > {plan / f'fetch_{i}.log'} 2>&1 &")
    print(f"\n  wait\n")


if __name__ == "__main__":
    main()
