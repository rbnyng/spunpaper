#!/usr/bin/env python
"""
Embedding-year ablation
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
from spun_train_patch import lgbm_fit  # noqa: E402

TESSERA_DIM = 128

def load_year(cache_dir: Path, sample_ids, patch: int):
    want = (patch, patch, TESSERA_DIM)
    X, ok = [], []
    for sid in sample_ids:
        f = cache_dir / f"{sid}.npy"
        good = False
        if f.exists():
            a = np.load(f)
            if tuple(a.shape) == want and np.isfinite(a).all():
                X.append(a.reshape(-1)); good = True
        ok.append(good)
    return np.array(ok), np.asarray(X, dtype=np.float32)

def reduce_fit(Xtr, n_components, seed=42, method="umap"):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    if method == "pca":
        from sklearn.decomposition import PCA
        rd = PCA(n_components=n_components, random_state=seed)
    else:
        import umap
        rd = umap.UMAP(n_components=n_components, random_state=seed,
                       n_neighbors=15, min_dist=0.1, metric="euclidean")
    Z = rd.fit_transform(sc.transform(Xtr))
    return sc, rd, np.asarray(Z, dtype=np.float32)

def split_70_10_20(n, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    tv, te = train_test_split(idx, test_size=0.2, random_state=seed)
    tr, va = train_test_split(tv, test_size=0.125, random_state=seed)
    return tr, va, te

def fit_lgbm(Xtr, ytr, Xva, yva, seed):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(random_state=seed, n_estimators=1000, learning_rate=0.05,
                          n_jobs=-1, verbose=-1)
    lgbm_fit(m, Xtr, ytr, eval_X=Xva, eval_y=yva, stopping_rounds=15)
    return m

def metrics(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import spearmanr
    sp = spearmanr(y, p).statistic
    return {"r2": float(r2_score(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "spearman": float(0.0 if np.isnan(sp) else sp)}

def run(args):
    out = Path(args.out_dir); (out / "raw").mkdir(parents=True, exist_ok=True)
    years = args.years
    df = pd.read_csv(args.samples_csv)
    df["sample_id"] = df["sample_id"].astype(str)

    caches = {y: Path(args.cache_template.format(year=y)) for y in years}
    present, raw = {}, {}
    for y in years:
        m, X = load_year(caches[y], df.sample_id.tolist(), args.patch_size)
        present[y] = m
        raw[y] = X
        print(f"  {y}: {m.sum()} / {len(df)} samples present")

    keep = np.logical_and.reduce([present[y] for y in years])
    print(f"samples present in ALL years: {keep.sum()}")
    if keep.sum() < 200:
        raise SystemExit("too few samples common to all years")

    # Re-index each year's matrix onto the common sample set.
    Xy = {}
    for y in years:
        pos = np.cumsum(present[y]) - 1
        Xy[y] = raw[y][pos[keep]]
    sub = df[keep].reset_index(drop=True)
    yv = sub.rarefied.to_numpy(float)
    coords = sub[["latitude", "longitude"]].to_numpy(float)
    n = len(sub)

    results = {"metadata": {
        "n_samples": int(n), "years": years, "patch_size": args.patch_size,
        "n_features_raw": int(Xy[years[0]].shape[1]),
        "dim_reduction": f"{args.dim_reduction}->{args.n_components}", "n_seeds": args.n_seeds,
    }, "embedding_drift": {}, "within_year": {}, "cross_year": {}}

    print("\n=== how much do the raw embeddings move between years? ===")
    base = years[-1]
    for y in years:
        if y == base:
            continue
        a, b = Xy[y], Xy[base]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        rel = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-9)
        results["embedding_drift"][f"{y}_vs_{base}"] = {
            "cosine_mean": float(cos.mean()), "cosine_p05": float(np.percentile(cos, 5)),
            "relative_l2_mean": float(rel.mean()),
        }
        print(f"  {y} vs {base}: cosine {cos.mean():.4f} (p05 {np.percentile(cos,5):.4f}), "
              f"relative L2 {rel.mean():.4f}")

    print("\n=== 1. within-year skill (fit and test inside each year) ===")
    print(f"{'year':6s} {'R2':>18s} {'RMSE':>8s} {'MAE':>8s} {'Spearman':>9s}")
    Z_cache = {}
    for y in years:
        t0 = time.time()
        _, _, Z = reduce_fit(Xy[y], args.n_components, method=args.dim_reduction)
        Z = np.hstack([Z, coords]).astype(np.float32)
        Z_cache[y] = Z
        per = []
        for s in range(1, args.n_seeds + 1):
            tr, va, te = split_70_10_20(n, s)
            m = fit_lgbm(Z[tr], yv[tr], Z[va], yv[va], s)
            per.append(metrics(yv[te], m.predict(Z[te])))
        agg = {f"{k}_mean": float(np.mean([p[k] for p in per])) for k in per[0]}
        agg.update({f"{k}_sd": float(np.std([p[k] for p in per])) for k in per[0]})
        agg["n_seeds"] = len(per); agg["fit_seconds"] = time.time() - t0
        results["within_year"][str(y)] = agg
        print(f"{y:6d} {agg['r2_mean']:+.4f} +/- {agg['r2_sd']:.4f} {agg['rmse_mean']:8.2f} "
              f"{agg['mae_mean']:8.2f} {agg['spearman_mean']:9.4f}")

    print(f"\n=== 2. cross-year transfer (fit on {base}, test same samples in each year) ===")
    sc, um, Ztr_base = reduce_fit(Xy[base], args.n_components, method=args.dim_reduction)
    Zb = np.hstack([Ztr_base, coords]).astype(np.float32)
    Zt = {base: Zb}
    for y in years:
        if y == base:
            continue
        Zt[y] = np.hstack([np.asarray(um.transform(sc.transform(Xy[y])), dtype=np.float32),
                           coords]).astype(np.float32)

    per_year_pred = {y: [] for y in years}
    per_year_true = []
    rows = {y: [] for y in years}
    for s in range(1, args.n_seeds + 1):
        tr, va, te = split_70_10_20(n, s)
        m = fit_lgbm(Zb[tr], yv[tr], Zb[va], yv[va], s)
        per_year_true.append(yv[te])
        for y in years:
            p = m.predict(Zt[y][te])
            per_year_pred[y].append(p)
            rows[y].append(metrics(yv[te], p))
    print(f"{'test year':10s} {'R2':>18s} {'RMSE':>8s} {'MAE':>8s} | "
          f"{'corr with ' + str(base) + ' preds':>22s}")
    for y in years:
        agg = {f"{k}_mean": float(np.mean([r[k] for r in rows[y]])) for k in rows[y][0]}
        agg.update({f"{k}_sd": float(np.std([r[k] for r in rows[y]])) for k in rows[y][0]})
        pc = [float(np.corrcoef(per_year_pred[y][i], per_year_pred[base][i])[0, 1])
              for i in range(len(rows[y]))]
        md = [float(np.mean(np.abs(per_year_pred[y][i] - per_year_pred[base][i])))
              for i in range(len(rows[y]))]
        agg["pred_corr_with_base_mean"] = float(np.mean(pc))
        agg["pred_mean_abs_diff_vs_base"] = float(np.mean(md))
        agg["delta_r2_vs_base"] = agg["r2_mean"] - float(np.mean([r["r2"] for r in rows[base]]))
        results["cross_year"][str(y)] = agg
        print(f"{y:<10d} {agg['r2_mean']:+.4f} +/- {agg['r2_sd']:.4f} {agg['rmse_mean']:8.2f} "
              f"{agg['mae_mean']:8.2f} | {agg['pred_corr_with_base_mean']:22.4f}")
    for y in years:
        if y != base:
            a = results["cross_year"][str(y)]
            print(f"  {y}: dR2 vs {base} = {a['delta_r2_vs_base']:+.4f}, "
                  f"mean |pred diff| = {a['pred_mean_abs_diff_vs_base']:.2f} species")

    np.savez_compressed(out / "raw" / "cross_year_predictions.npz",
                        y_true=np.concatenate(per_year_true),
                        **{f"pred_{y}": np.concatenate(per_year_pred[y]) for y in years})
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    results_io.dump(out, "year_ablation", stats=results,
                    tables={"within_year": pd.DataFrame(results["within_year"]).T,
                            "cross_year": pd.DataFrame(results["cross_year"]).T})
    print(f"\nwrote {out}/results.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-csv", required=True)
    p.add_argument("--cache-template", default="caches/tessera_{year}_p3",
                   help="Pattern for each year's patch cache, e.g. 'caches/tessera_{year}_p3'.")
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024],
                   help="Last year listed is the reference used for cross-year transfer.")
    p.add_argument("--out-dir", default="year_ablation_results")
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--n-components", type=int, default=256)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--dim-reduction", default="umap", choices=["umap", "pca"],
                   help="umap matches the paper; pca is the control for the "
                        "cross-year test (exact linear transform).")
    run(p.parse_args())


if __name__ == "__main__":
    main()
