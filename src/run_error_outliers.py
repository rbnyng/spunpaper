#!/usr/bin/env python
"""
embedding_knn_distance   mean distance to the k nearest training samples in the representation space -- novelty / extrapolation
embedding_mahalanobis    whitened distance to the training centroid
nn_distance_to_train_km  geographic distance to the nearest training sample
n_within_{1,10,50}km     local sampling density
predicted_richness       the model's own output

runs on embeddings alone (pass --no-use-climate --no-use-soil).
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
from spatial_cv_strategies import (  # noqa: E402
    RECOMMENDED_LINK_M, distance_group_kfold, random_kfold,
)
from spun_train_patch import (  # noqa: E402
    CombinedPatchClimateEvaluator, lgbm_fit, reduce_satellite_block,
)

SATELLITE_PREFIXES = ("patch_", "pca_", "umap_")
EARTH_R_KM = 6371.0088

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

def fit_predict(Xtr, ytr, Xte, model, seed):
    if model == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        return m.fit(Xtr, ytr).predict(Xte)
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.125, random_state=seed)
    m = lgb.LGBMRegressor(random_state=seed, n_estimators=2000, learning_rate=0.05,
                          n_jobs=-1, verbose=-1)
    lgbm_fit(m, Xtr[itr], ytr[itr], eval_X=Xtr[iva], eval_y=ytr[iva],
             stopping_rounds=25)
    return m.predict(Xte)

def knn_novelty(E, train_idx, test_idx, k=10):
    from sklearn.neighbors import NearestNeighbors
    k = int(min(k, len(train_idx)))
    nn = NearestNeighbors(n_neighbors=k).fit(E[train_idx])
    d, _ = nn.kneighbors(E[test_idx])
    return d.mean(axis=1)

def whitened_distance(E, train_idx, test_idx, n_comp=32):
    """
    Distance to the training centroid after whitening as Mahalanobis proxy.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(E[train_idx])
    k = int(min(n_comp, E.shape[1], len(train_idx) - 1))
    if k < 2:
        return np.zeros(len(test_idx))
    pca = PCA(n_components=k, whiten=True, random_state=0).fit(sc.transform(E[train_idx]))
    return np.linalg.norm(pca.transform(sc.transform(E[test_idx])), axis=1)

def neighbour_counts(coords, radii_km):
    """Number of other samples within each radius, via a haversine ball tree."""
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(coords), metric="haversine")
    out = {}
    for r in radii_km:
        c = tree.query_radius(np.radians(coords), r=r / EARTH_R_KM, count_only=True)
        out[f"n_within_{r:g}km"] = c.astype(float) - 1.0        # exclude self
    return out

def cohens_d_boot(v, mask, n_boot=1000, seed=0, alpha=0.05):
    """
    Cohen's d (outliers - rest) with a percentile bootstrap CI.
    """
    a, b = v[mask], v[~mask]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), float("nan")
    def d_of(x, y):
        na, nb = x.shape[-1], y.shape[-1]
        sp = np.sqrt(((na - 1) * x.var(axis=-1, ddof=1) +
                      (nb - 1) * y.var(axis=-1, ddof=1)) / (na + nb - 2))
        sp = np.where(sp == 0, np.nan, sp)
        return (x.mean(axis=-1) - y.mean(axis=-1)) / sp
    point = float(d_of(a[None, :], b[None, :])[0])
    if not np.isfinite(point):
        # Zero pooled SD (e.g. an indicator that is constant within both groups).
        # d is undefined; the Mann-Whitney p and the rank correlation still apply.
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    step = 200
    for s in range(0, n_boot, step):
        m = min(step, n_boot - s)
        ia = rng.integers(0, len(a), (m, len(a)))
        ib = rng.integers(0, len(b), (m, len(b)))
        boots[s:s + m] = d_of(a[ia], b[ib])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)

def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    pp = p[ok]
    order = np.argsort(pp)
    n = len(pp)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        prev = min(prev, pp[order[rank]] * n / (rank + 1))
        adj[order[rank]] = prev
    q[ok] = np.minimum(adj, 1.0)
    return q

def capture_curve(score, abs_err, fractions=(0.05, 0.10, 0.20)):
    ok = np.isfinite(score) & np.isfinite(abs_err)
    s, e = score[ok], abs_err[ok]
    total = e.sum()
    order = np.argsort(-s)
    out = []
    for f in fractions:
        m = max(1, int(round(f * len(s))))
        share = float(e[order[:m]].sum() / total) if total > 0 else float("nan")
        out.append({"top_fraction": float(f), "error_share": share,
                    "lift": float(share / f) if f > 0 else float("nan")})
    return out

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
    print("preparing features ...")
    X, y, locations, skipped, patch_dims, fnames, dr_config = ev.prepare_dataset(
        df, representations_dir_path=args.representations_dir,
        worldcover_features_path=args.worldcover_path,
        use_satellite=True, use_climate=args.use_climate, use_soil=args.use_soil,
        use_worldcover=args.use_worldcover,
        satellite_dim_reduction=args.dim_reduction,
        dim_reduction_components=args.dim_reduction_components, random_state=42)
    if X is None:
        raise SystemExit("prepare_dataset returned no data")

    names = np.array([str(n) for n in fnames])
    coord_mask = np.isin(names, ["latitude", "longitude"])
    sat_mask = np.array([n.startswith(SATELLITE_PREFIXES) for n in names])
    env_mask = ~(coord_mask | sat_mask)
    E = X[:, sat_mask]
    coords = np.array([[l["latitude"], l["longitude"]] for l in locations], dtype=float)
    sample_ids = np.array([str(l["sample_id"]) for l in locations])
    n = len(y)
    if args.model_feature_set == "satellite":
        model_mask = sat_mask | coord_mask
    else:
        model_mask = np.ones(len(names), dtype=bool)
    Xm = X[:, model_mask]
    print(f"  n={n}, model features {Xm.shape[1]} ({args.model_feature_set}), "
          f"embedding dims {E.shape[1]}, environmental descriptors {int(env_mask.sum())}")

    # --- out-of-sample predictions + label-free diagnostics, per fold ---------
    P = np.zeros((args.n_seeds, n)); P[:] = np.nan
    KNN = np.full((args.n_seeds, n), np.nan)
    MAH = np.full((args.n_seeds, n), np.nan)
    NND = np.full((args.n_seeds, n), np.nan)
    from spatial_cv_strategies import nn_distance_to_train
    for si in range(args.n_seeds):
        seed = 42 + si
        folds = (random_kfold(coords, args.n_splits, seed) if args.cv == "random"
                 else distance_group_kfold(coords, RECOMMENDED_LINK_M, args.n_splits, seed))
        print(f"  seed {seed}: {len(folds)} {args.cv} folds", flush=True)
        for tr, te in folds:
            Xtr, Xte = reduce_satellite_block(dr_config, Xm[tr], Xm[te])
            P[si, te] = fit_predict(Xtr, y[tr], Xte, args.model, seed)
            KNN[si, te] = knn_novelty(E, tr, te, args.knn)
            MAH[si, te] = whitened_distance(E, tr, te)
            NND[si, te] = nn_distance_to_train(coords, te, tr)

    from sklearn.metrics import r2_score
    pred = np.nanmean(P, axis=0)
    ok = np.isfinite(pred)
    resid = y - pred
    abs_err = np.abs(resid)
    pct_err = np.where(y > 0, 100.0 * abs_err / np.maximum(y, 1e-9), np.nan)
    pooled_r2 = float(r2_score(y[ok], pred[ok]))
    print(f"  pooled out-of-sample R2 = {pooled_r2:+.3f}  MAE = {np.nanmean(abs_err):.2f} species")

    diag = {
        "embedding_knn_distance": np.nanmean(KNN, axis=0),
        "embedding_mahalanobis": np.nanmean(MAH, axis=0),
        "nn_distance_to_train_km": np.nanmean(NND, axis=0),
        "predicted_richness": pred,
    }
    diag.update(neighbour_counts(coords, args.density_radii_km))
    label_free = list(diag.keys())
    diag["observed_richness"] = y.astype(float)

    twin_share = float(np.mean(diag["nn_distance_to_train_km"] < 0.01))
    print(f"  test samples with a training sample within 10 m: {twin_share:.1%}"
          + ("  <- random folds; use --cv grouped for the applicability-domain reading"
             if args.cv == "random" and twin_share > 0.2 else ""))

    thr = float(np.nanquantile(abs_err[ok], args.error_quantile))
    outlier = ok & (abs_err >= thr)
    print(f"  outliers: |error| >= {thr:.2f} species "
          f"(top {(1-args.error_quantile):.0%}) -> {int(outlier.sum())} samples")

    # --- group contrasts ------------------------------------------------------
    from scipy.stats import mannwhitneyu, spearmanr
    rows = []
    candidates = [(k, v, k in label_free) for k, v in diag.items()]
    env_names = names[env_mask]
    for i, en in enumerate(env_names):
        candidates.append((str(en), X[:, env_mask][:, i].astype(float), False))
    for name, v, free in candidates:
        v = np.asarray(v, dtype=float)
        m = ok & np.isfinite(v)
        if m.sum() < 50 or np.nanstd(v[m]) == 0:
            continue
        d, lo, hi = cohens_d_boot(v[m], outlier[m], n_boot=args.n_boot, seed=abs(hash(name)) % 10**6)
        try:
            p = float(mannwhitneyu(v[m & outlier], v[m & ~outlier],
                                   alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        rho = float(spearmanr(v[m], abs_err[m]).statistic)
        rows.append({
            "variable": name, "label_free": bool(free),
            "mean_outliers": float(np.mean(v[m & outlier])),
            "mean_rest": float(np.mean(v[m & ~outlier])),
            "cohens_d": d, "ci_low": lo, "ci_high": hi,
            "p_value": p, "spearman_with_abs_error": rho,
        })
    if not rows:
        raise SystemExit("no usable diagnostics or descriptors")
    q = bh_fdr([r["p_value"] for r in rows])
    for r, qq in zip(rows, q):
        r["q_value_bh"] = float(qq)
    rows.sort(key=lambda r: -abs(r["cohens_d"]) if np.isfinite(r["cohens_d"]) else 0.0)

    print(f"\n{'variable':32s} {'free':>5s} {'d':>7s} {'95% CI':>18s} {'rho|err|':>9s}")
    print("-" * 76)
    for r in rows[:15]:
        print(f"{r['variable'][:32]:32s} {'yes' if r['label_free'] else '  -':>5s} "
              f"{r['cohens_d']:+7.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
              f"{r['spearman_with_abs_error']:+9.3f}")
    if len(rows) > 15:
        print(f"... {len(rows)-15} more in the CSV")

    capture = {k: capture_curve(np.asarray(diag[k], float), abs_err) for k in label_free}
    print(f"\nerror captured by flagging the top 10% of samples "
          f"(lift 1.0 = no better than random):")
    for k in label_free:
        c10 = [c for c in capture[k] if abs(c["top_fraction"] - 0.10) < 1e-9][0]
        print(f"  {k:28s} {c10['error_share']:6.1%}   lift {c10['lift']:.2f}")

    dec = pd.qcut(pd.Series(y[ok]), 10, labels=False, duplicates="drop")
    rtm = []
    for dv in sorted(pd.unique(dec.dropna())):
        m = np.zeros(n, dtype=bool); m[np.flatnonzero(ok)[(dec == dv).values]] = True
        rtm.append({"observed_richness_decile": int(dv) + 1, "n": int(m.sum()),
                    "mean_observed": float(np.mean(y[m])),
                    "outlier_rate": float(outlier[m].mean()),
                    "mean_signed_residual": float(np.mean(resid[m]))})
    shrink_rho = float(spearmanr(y[ok], resid[ok]).statistic)
    print(f"\nregression-to-the-mean control: spearman(observed, signed residual) "
          f"= {shrink_rho:+.3f}")
    print(f"  outlier rate by observed-richness decile: "
          + " ".join(f"{r['outlier_rate']:.2f}" for r in rtm))

    biomes = []
    if args.biomes:
        try:
            from spun_train_patch import BiomeFilter
            bf = BiomeFilter(cache_dir=Path(args.ecoregions_cache) if args.ecoregions_cache else None)
            bdf = bf.assign_biomes(pd.DataFrame({"sample_id": sample_ids,
                                                 "latitude": coords[:, 0],
                                                 "longitude": coords[:, 1]}))
            bdf = bdf.drop_duplicates(subset=["sample_id"]).set_index("sample_id")
            bname = bdf.reindex(sample_ids)["BIOME_NAME"].fillna("unassigned").values
            base = outlier[ok].mean()
            for b in pd.unique(bname):
                m = ok & (bname == b)
                if m.sum() < args.min_biome_n:
                    continue
                biomes.append({"biome": str(b), "n": int(m.sum()),
                               "mean_abs_error": float(np.mean(abs_err[m])),
                               "outlier_rate": float(outlier[m].mean()),
                               "enrichment": float(outlier[m].mean() / base) if base > 0 else float("nan")})
            biomes.sort(key=lambda r: -r["enrichment"])
            print(f"\n{'biome':44s} {'n':>6s} {'MAE':>7s} {'outlier%':>9s} {'enrich':>7s}")
            for r in biomes[:10]:
                print(f"{r['biome'][:44]:44s} {r['n']:6d} {r['mean_abs_error']:7.1f} "
                      f"{r['outlier_rate']:8.1%} {r['enrichment']:7.2f}")
        except Exception as e:                                   # noqa: BLE001
            print(f"\n(biome breakdown skipped: {type(e).__name__}: {e})")

    results = {
        "metadata": {
            "n_samples": int(n), "n_model_features": int(Xm.shape[1]),
            "model_feature_set": args.model_feature_set, "model": args.model,
            "dim_reduction": f"{args.dim_reduction}->{args.dim_reduction_components}",
            "cv": args.cv, "n_splits": args.n_splits, "n_seeds": args.n_seeds,
            "error_quantile": args.error_quantile,
            "prediction": "mean over seeds of the out-of-sample prediction",
            "share_of_test_samples_with_training_twin_within_10m": twin_share,
        },
        "model_performance": {
            "pooled_r2": pooled_r2, "mae": float(np.nanmean(abs_err)),
            "rmse": float(np.sqrt(np.nanmean(resid ** 2))),
            "median_abs_error": float(np.nanmedian(abs_err)),
            "median_pct_error": float(np.nanmedian(pct_err)),
        },
        "outlier_definition": {
            "abs_error_threshold": thr, "n_outliers": int(outlier.sum()),
            "quantile": args.error_quantile,
            "share_of_total_abs_error": float(abs_err[outlier].sum() / abs_err[ok].sum()),
        },
        "group_contrasts": rows,
        "capture_curves": capture,
        "regression_to_the_mean": {
            "spearman_observed_vs_signed_residual": shrink_rho,
            "by_observed_richness_decile": rtm,
        },
        "by_biome": biomes,
        "summary": {
            "best_label_free_diagnostic": max(
                label_free,
                key=lambda k: [c for c in capture[k] if abs(c["top_fraction"] - 0.10) < 1e-9][0]["lift"]),
        },
    }
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "error_outlier_contrasts.csv", index=False)
    pd.DataFrame([{"diagnostic": k, **c} for k, cs in capture.items() for c in cs]).to_csv(
        out / "capture_curves.csv", index=False)
    np.savez_compressed(
        out / "raw" / "per_sample_errors.npz",
        sample_id=sample_ids, latitude=coords[:, 0], longitude=coords[:, 1],
        y_true=y, y_pred=pred, residual=resid, abs_error=abs_err,
        outlier=outlier, **{k: np.asarray(v, dtype=float) for k, v in diag.items()})
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    results_io.dump(out, "error_outliers", stats=results,
                    tables={"group_contrasts": tab})
    print(f"\nbest label-free diagnostic: {results['summary']['best_label_free_diagnostic']}")
    print(f"wrote {out}/results.json")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representations-dir", default=paths.REPRESENTATIONS_DIR)
    p.add_argument("--biodiversity-csvs", nargs="+", default=paths.ECM_CSVS)
    p.add_argument("--out-dir", default="results/error_outliers")
    p.add_argument("--cv", default="random", choices=["random", "grouped"],
                   help="random matches the main body; grouped is the robustness variant.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--model", default="lightgbm", choices=["lightgbm", "rf", "ridge"])
    p.add_argument("--model-feature-set", default="satellite", choices=["satellite", "all"],
                   help="'satellite' = embeddings + lat/lon, the paper's satellite row.")
    p.add_argument("--error-quantile", type=float, default=0.90,
                   help="samples at or above this |error| quantile are the outliers.")
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--density-radii-km", type=float, nargs="+", default=[1.0, 10.0, 50.0])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--biomes", action=argparse.BooleanOptionalAction, default=True,
                   help="RESOLVE ecoregion breakdown; needs geopandas and a download.")
    p.add_argument("--ecoregions-cache", default=paths.ECOREGIONS_CACHE)
    p.add_argument("--min-biome-n", type=int, default=50)
    p.add_argument("--dim-reduction", default="umap", choices=["none", "pca", "umap"])
    p.add_argument("--dim-reduction-components", type=int, default=256)
    p.add_argument("--use-climate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-soil", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-worldcover", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--climate-data-dir", default=paths.WORLDCLIM_DIR)
    p.add_argument("--climate-cache-dir", default=paths.CLIMATE_CACHE_DIR)
    p.add_argument("--soil-cache-dir", default=paths.SOIL_CACHE_DIR)
    p.add_argument("--worldcover-path", default=paths.WORLDCOVER_CSV)
    p.add_argument("--satellite-source", default="geotessera",
                   choices=["geotessera", "precomputed"])
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--year", type=int, default=2024)
    run(p.parse_args())


if __name__ == "__main__":
    main()
