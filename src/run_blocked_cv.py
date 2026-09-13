#!/usr/bin/env python
"""
Spatially explicit cross-validation robustness check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import results_io  # noqa: E402
from spatial_cv_strategies import (  # noqa: E402
    distance_group_kfold, nn_distance_to_train, random_kfold, spatial_block_kfold_m,
)
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

def fit_predict(Xtr, ytr, Xte, seed, model_name):
    from sklearn.model_selection import train_test_split
    if model_name == "lightgbm":
        import lightgbm as lgb
        tr, va = train_test_split(np.arange(len(Xtr)), test_size=0.125, random_state=seed)
        m = lgb.LGBMRegressor(random_state=seed, n_estimators=1000, learning_rate=0.05,
                              n_jobs=-1, verbose=-1)
        lgbm_fit(m, Xtr[tr], ytr[tr], eval_X=Xtr[va], eval_y=ytr[va],
                 stopping_rounds=15)
    else:
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        m.fit(Xtr, ytr)
    return m.predict(Xte)

def score(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import spearmanr
    sp = spearmanr(y, p).statistic
    return {"r2": float(r2_score(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "spearman": float(0.0 if np.isnan(sp) else sp)}

def run(args):
    out = Path(args.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    df = load_biodiversity(args.biodiversity_csvs)
    print(f"biodiversity records: {len(df)}")

    ev = CombinedPatchClimateEvaluator(
        climate_data_path=args.climate_data_dir, use_climate_cache=True,
        climate_features_cache_dir=args.climate_cache_dir,
        soil_features_cache_dir=args.soil_cache_dir,
        geotessera_year=args.year, geotessera_patch_size=args.patch_size,
        satellite_source=args.satellite_source,
    )
    t0 = time.time()
    X, y, locations, skipped, patch_dims, fnames, dr_config = ev.prepare_dataset(
        df, representations_dir_path=args.representations_dir,
        worldcover_features_path=args.worldcover_path,
        use_satellite=args.use_satellite, use_climate=args.use_climate,
        use_soil=args.use_soil, use_worldcover=args.use_worldcover,
        satellite_dim_reduction=args.dim_reduction,
        dim_reduction_components=args.dim_reduction_components,
        random_state=42,
    )
    if X is None:
        raise SystemExit("prepare_dataset returned no data")
    coords = np.array([[l["latitude"], l["longitude"]] for l in locations], dtype=float)
    print(f"  X={X.shape}  patch_dims={patch_dims}  ({time.time()-t0:.0f}s)")
    uniq = len(np.unique(np.round(coords, 6), axis=0))
    print(f"  {len(coords)} samples at {uniq} distinct locations "
          f"({(1-uniq/len(coords))*100:.1f}% share a location)")

    feature_sets = "+".join([n for n, on in [("sat", args.use_satellite),
                                             ("clim", args.use_climate),
                                             ("soil", args.use_soil),
                                             ("wc", args.use_worldcover)] if on])

    schemes = {"random": lambda s: random_kfold(coords, args.n_splits, s),
               f"group_{args.link_m:g}m":
                   lambda s: distance_group_kfold(coords, args.link_m, args.n_splits, s)}
    for b in args.block_sizes:
        schemes[f"block_{b:g}m"] = (
            lambda s, b=b: spatial_block_kfold_m(coords, b, args.n_splits, s))

    results = {"metadata": {
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_distinct_locations": int(uniq), "feature_sets": feature_sets,
        "patch_dims": list(patch_dims) if patch_dims else None,
        "dim_reduction": f"{args.dim_reduction}->{args.dim_reduction_components}",
        "model": args.model, "n_splits": args.n_splits, "seeds": list(range(1, args.n_seeds + 1)),
        "block_sizes_m": list(args.block_sizes), "link_m": args.link_m,
    }, "schemes": {}}

    print(f"\nfeature set: {feature_sets}   model: {args.model}")
    print(f"{'scheme':14s} {'R2 (per-fold)':>20s} {'R2 pooled':>10s} {'RMSE':>8s} "
          f"{'MAE':>8s} {'nnDist_m':>9s} {'vs random':>10s}")
    print("-" * 86)
    base_r2 = None
    for name, fn in schemes.items():
        per_fold, pooled_t, pooled_p, dists = [], [], [], []
        for s in range(1, args.n_seeds + 1):
            for tr, te in fn(s):
                if len(tr) < 50 or len(te) < 10:
                    continue
                Xtr, Xte = reduce_satellite_block(dr_config, X[tr], X[te])
                p = fit_predict(Xtr, y[tr], Xte, s, args.model)
                per_fold.append(score(y[te], p))
                pooled_t.append(y[te]); pooled_p.append(p)
                d = nn_distance_to_train(coords, te, tr)
                if len(d):
                    dists.append(float(np.median(d)) * 1000.0)
        if not per_fold:
            continue
        pt, pp = np.concatenate(pooled_t), np.concatenate(pooled_p)
        agg = {f"{k}_mean": float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
        agg.update({f"{k}_sd": float(np.std([f[k] for f in per_fold])) for k in per_fold[0]})
        agg.update(score(pt, pp))
        agg = {**agg, "pooled_r2": agg.pop("r2"), "pooled_rmse": agg.pop("rmse"),
               "pooled_mae": agg.pop("mae"), "pooled_spearman": agg.pop("spearman"),
               "n_folds": len(per_fold),
               "median_nn_dist_m": float(np.median(dists)) if dists else float("nan")}
        results["schemes"][name] = agg
        if base_r2 is None:
            base_r2 = agg["r2_mean"]
        np.savez_compressed(out / "raw" / f"{feature_sets}_{name}.npz",
                            y_true=pt, y_pred=pp)
        print(f"{name:14s} {agg['r2_mean']:+.4f} +/- {agg['r2_sd']:.4f}    "
              f"{agg['pooled_r2']:+10.4f} {agg['rmse_mean']:8.2f} {agg['mae_mean']:8.2f} "
              f"{agg['median_nn_dist_m']:9.1f} {agg['r2_mean']-base_r2:+10.4f}")

    if results["schemes"]:
        rs = results["schemes"]
        worst = min(v["r2_mean"] for v in rs.values())
        results["summary"] = {
            "random_r2": rs.get("random", {}).get("r2_mean"),
            "worst_spatial_r2": worst,
            "max_drop_vs_random": float(rs.get("random", {}).get("r2_mean", np.nan) - worst),
            "verdict": ("performance is retained under spatially explicit CV"
                        if worst > 0.5 * rs.get("random", {}).get("r2_mean", 1)
                        else "performance drops substantially under spatially explicit CV"),
        }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    pd.DataFrame(results["schemes"]).T.to_csv(out / "schemes.csv")
    results_io.dump(out, f"blocked_cv_{feature_sets}", stats=results)
    print(f"\nwrote {out}/results.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representations-dir", required=True)
    p.add_argument("--biodiversity-csvs", nargs="+", required=True)
    p.add_argument("--out-dir", default="results/spatial_cv")
    p.add_argument("--block-sizes", type=float, nargs="+", default=[25.0, 50.0])
    p.add_argument("--link-m", type=float, default=25.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--model", default="lightgbm", choices=["lightgbm", "rf"])
    p.add_argument("--satellite-source", default="geotessera",
                   choices=["geotessera", "precomputed"],
                   help="'precomputed' loads {sample_id}.npy straight from "
                        "--representations-dir with no fetching and no channel-count "
                        "assumption; use it for the spectral-index baseline.")
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--year", type=int, default=2024)
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
    run(p.parse_args())


if __name__ == "__main__":
    main()
