#!/usr/bin/env python
"""
Prediction intervals via quantile regression.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
import results_io  # noqa: E402
from spun_train_patch import (  # noqa: E402
    CombinedPatchClimateEvaluator, lgbm_fit, reduce_satellite_block,
)

def load_biodiversity(csv_paths):
    frames = []
    for p in csv_paths:
        d = pd.read_csv(p)
        need = {"sample_id", "latitude", "longitude", "rarefied"}
        if not need.issubset(d.columns):
            raise ValueError(f"{p} missing {need - set(d.columns)}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["latitude", "longitude", "rarefied", "sample_id"])
    df["sample_id"] = df["sample_id"].astype(str)
    return df.drop_duplicates(subset=["sample_id"])
def pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))

def fit_quantile(Xtr, ytr, Xva, yva, Xte, q, seed):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=seed,
                          n_estimators=1000, learning_rate=0.05, n_jobs=-1, verbose=-1)
    lgbm_fit(m, Xtr, ytr, eval_X=Xva, eval_y=yva, stopping_rounds=15)
    return m.predict(Xte)

def split_70_10_20(n, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    tv, te = train_test_split(idx, test_size=0.2, random_state=seed)
    tr, va = train_test_split(tv, test_size=0.125, random_state=seed)
    return tr, va, te

def run(args):
    out = Path(args.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    df = load_biodiversity(args.biodiversity_csvs)

    ev = CombinedPatchClimateEvaluator(
        climate_data_path=args.climate_data_dir, use_climate_cache=True,
        climate_features_cache_dir=args.climate_cache_dir,
        soil_features_cache_dir=args.soil_cache_dir,
        geotessera_year=args.year, geotessera_patch_size=args.patch_size,
        satellite_source=args.satellite_source)
    X, y, locations, skipped, patch_dims, fnames, dr_config = ev.prepare_dataset(
        df, representations_dir_path=args.representations_dir,
        worldcover_features_path=args.worldcover_path,
        use_satellite=args.use_satellite, use_climate=args.use_climate,
        use_soil=args.use_soil, use_worldcover=args.use_worldcover,
        satellite_dim_reduction=args.dim_reduction,
        dim_reduction_components=args.dim_reduction_components, random_state=42)
    if X is None:
        raise SystemExit("prepare_dataset returned no data")
    n = len(y)
    lo_q, hi_q = args.lower_quantile, args.upper_quantile
    nominal = hi_q - lo_q
    print(f"X={X.shape}  nominal central interval = {nominal:.0%} "
          f"(q{lo_q:g} to q{hi_q:g})")

    rows, all_lo, all_md, all_hi, all_y, all_idx = [], [], [], [], [], []
    for s in range(1, args.n_seeds + 1):
        tr, va, te = split_70_10_20(n, s)
        Xtr, Xva, Xte = reduce_satellite_block(dr_config, X[tr], X[va], X[te])
        lo = fit_quantile(Xtr, y[tr], Xva, y[va], Xte, lo_q, s)
        md = fit_quantile(Xtr, y[tr], Xva, y[va], Xte, 0.5, s)
        hi = fit_quantile(Xtr, y[tr], Xva, y[va], Xte, hi_q, s)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)   # guard quantile crossing
        inside = (y[te] >= lo) & (y[te] <= hi)
        width = hi - lo
        err = np.abs(y[te] - md)
        rows.append({
            "coverage": float(inside.mean()),
            "mean_width": float(width.mean()), "median_width": float(np.median(width)),
            "pinball_lo": pinball(y[te], lo, lo_q),
            "pinball_md": pinball(y[te], md, 0.5),
            "pinball_hi": pinball(y[te], hi, hi_q),
            "width_error_corr": float(np.corrcoef(width, err)[0, 1]),
        })
        all_lo.append(lo); all_md.append(md); all_hi.append(hi)
        all_y.append(y[te]); all_idx.append(te)

    agg = {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg.update({f"{k}_sd": float(np.std([r[k] for r in rows])) for k in rows[0]})

    LO, MD, HI = np.concatenate(all_lo), np.concatenate(all_md), np.concatenate(all_hi)
    YT, IDX = np.concatenate(all_y), np.concatenate(all_idx)
    inside = (YT >= LO) & (YT <= HI)
    width = HI - LO

    # Stratify by predicted richness
    strata = []
    edges = np.quantile(MD, [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        m = (MD >= edges[i]) & (MD <= edges[i + 1] if i == 3 else MD < edges[i + 1])
        if m.sum() < 10:
            continue
        strata.append({
            "predicted_richness_range": [float(edges[i]), float(edges[i + 1])],
            "n": int(m.sum()), "coverage": float(inside[m].mean()),
            "mean_width": float(width[m].mean()),
            "mean_abs_error": float(np.mean(np.abs(YT[m] - MD[m]))),
        })

    results = {
        "metadata": {
            "n_samples": int(n), "n_features": int(X.shape[1]),
            "quantiles": [lo_q, 0.5, hi_q], "nominal_coverage": float(nominal),
            "n_seeds": args.n_seeds, "model": "lightgbm objective=quantile (pinball)",
            "protocol": "70/10/20 with early stopping, as in the paper",
            "feature_sets": "+".join([nme for nme, on in
                                      [("sat", args.use_satellite), ("clim", args.use_climate),
                                       ("soil", args.use_soil), ("wc", args.use_worldcover)] if on]),
        },
        "overall": agg,
        "by_predicted_richness": strata,
        "summary": {
            "nominal_coverage": float(nominal),
            "empirical_coverage": agg["coverage_mean"],
            "calibration_gap": float(agg["coverage_mean"] - nominal),
        },
    }
    print(f"\ncoverage {agg['coverage_mean']:.3f} (nominal {nominal:.2f})   "
          f"mean width {agg['mean_width_mean']:.1f} species   "
          f"width-error r {agg['width_error_corr_mean']:+.3f}")
    print(f"{'predicted richness':28s} {'n':>6s} {'coverage':>9s} {'width':>8s} {'MAE':>8s}")
    for s_ in strata:
        rng = f"{s_['predicted_richness_range'][0]:.0f}-{s_['predicted_richness_range'][1]:.0f}"
        print(f"  {rng:26s} {s_['n']:6d} {s_['coverage']:9.3f} "
              f"{s_['mean_width']:8.1f} {s_['mean_abs_error']:8.1f}")

    coords = np.array([[l["latitude"], l["longitude"]] for l in locations], dtype=float)
    np.savez_compressed(out / "raw" / "prediction_intervals.npz",
                        y_true=YT, lower=LO, median=MD, upper=HI,
                        sample_index=IDX, latitude=coords[IDX, 0], longitude=coords[IDX, 1])
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    results_io.dump(out, "uncertainty", stats=results)
    print(f"\nwrote {out}/results.json")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representations-dir", default=paths.REPRESENTATIONS_DIR)
    p.add_argument("--biodiversity-csvs", nargs="+", default=paths.ECM_CSVS)
    p.add_argument("--out-dir", default="results/prediction_intervals")
    p.add_argument("--lower-quantile", type=float, default=0.05)
    p.add_argument("--upper-quantile", type=float, default=0.95)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--dim-reduction", default="umap", choices=["none", "pca", "umap"])
    p.add_argument("--dim-reduction-components", type=int, default=256)
    p.add_argument("--use-satellite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-climate", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-soil", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-worldcover", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--climate-data-dir", default=None)
    p.add_argument("--climate-cache-dir", default=None)
    p.add_argument("--soil-cache-dir", default=None)
    p.add_argument("--worldcover-path", default=None)
    p.add_argument("--satellite-source", default="geotessera",
                   choices=["geotessera", "precomputed"])
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--year", type=int, default=2024)
    run(p.parse_args())


if __name__ == "__main__":
    main()
