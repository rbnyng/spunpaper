#!/usr/bin/env python
"""
Make a small reproduction sample.

python data_generation/make_reprod_sample.py --deposit /path/to/extracted/deposit --out reprod_sample
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

# ecm_representations is the only cache that decorates the sample ID.
_DECORATED = re.compile(r"^\d{4}_(.+)_rarefied$")

# Caches to use and the ground-truth CSVs
CACHE_DIRS = [
    "ecm_representations", "ecm_tessera_p7", "spectral_p3",
    "tessera_2022_p3", "tessera_2023_p3", "tessera_2024_p3",
    "soil_features_cache", "climate_features_cache",
]
GT_CSVS = {"Europe": "ECM_richness_europe.csv", "Asia": "ECM_richness_Asia.csv"}
GT_COLS = ["sample_id", "latitude", "longitude", "continent", "rarefied"]

def strip_id(stem: str) -> str:
    m = _DECORATED.match(stem)
    return m.group(1) if m else stem

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deposit", required=True, type=Path,
                   help="Directory holding the extracted full caches and the two CSVs.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    dep, out = a.deposit, a.out
    frames = []
    for cont, name in GT_CSVS.items():
        f = dep / name
        if not f.is_file():
            sys.exit(f"missing ground truth: {f}")
        frames.append(pd.read_csv(f)[GT_COLS])
    gt = pd.concat(frames, ignore_index=True)
    gt["sample_id"] = gt.sample_id.astype(str)
    gt = gt.dropna(subset=GT_COLS).drop_duplicates("sample_id")

    main_dir = dep / "ecm_representations"
    covered = {strip_id(f.stem) for f in main_dir.glob("*.npy")}
    core = gt[gt.sample_id.isin(covered)]
    print(f"ground truth {len(gt)}, covered by the embedding cache {len(core)}")

    sample = (pd.concat([g.sample(frac=a.frac, random_state=a.seed)
                         for _, g in core.groupby("continent")])
              .sort_values("sample_id").reset_index(drop=True))
    keep = set(sample.sample_id)
    print(f"sampled {len(sample)} ({100 * len(sample) / len(core):.2f}%)  "
          f"{sample.continent.value_counts().to_dict()}")

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": a.seed, "frac": a.frac, "n_sampled": len(sample),
                "n_source": len(core), "by_continent": sample.continent.value_counts().to_dict(),
                "caches": {}}

    for name in CACHE_DIRS:
        src = dep / name
        if not src.is_dir():
            print(f"  {name:24s} not in deposit, skipped")
            continue
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in src.glob("*.npy"):
            if strip_id(f.stem) in keep:
                shutil.copy2(f, dst / f.name)
                n += 1
        manifest["caches"][name] = n
        print(f"  {name:24s} {n:5d} of {len(list(src.glob('*.npy'))):6d}")

    wc = pd.read_csv(dep / "worldcover_features.csv")
    wc = wc[wc.sample_id.astype(str).isin(keep)]
    wc.to_csv(out / "worldcover_features.csv", index=False)
    manifest["caches"]["worldcover_features.csv"] = len(wc)

    for cont, name in GT_CSVS.items():
        sub = sample[sample.continent == cont]
        sub.to_csv(out / name, index=False)
        manifest["caches"][name] = len(sub)

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nwrote {out}  ({total / 1e6:.1f} MB uncompressed)")

if __name__ == "__main__":
    main()
