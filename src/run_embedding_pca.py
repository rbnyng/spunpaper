#!/usr/bin/env python
"""
Embedding PCA structure and environment correlations (SI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
import results_io
from spun_train_patch import CombinedPatchClimateEvaluator


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


def bootstrap_ci(x, y, kind="spearman", n_boot=2000, seed=0, alpha=0.05,
                 chunk=256):
    """
    Percentile bootstrap CI for a correlation coefficient.

    The point estimate and p-value are the exact statistic. The interval is a
    percentile bootstrap computed in closed form rather than by calling
    spearmanr once per resample: for Spearman the two vectors are ranked once
    up front and Pearson is then evaluated on the ranks, which is the standard
    treatment and turns 10x114x2000 ranking passes over ~12,000 points into a
    handful of vectorized reductions. The only approximation is that ties are
    broken on the original sample rather than re-broken within each resample,
    which is immaterial for continuous covariates.
    """
    from scipy.stats import pearsonr, rankdata, spearmanr
    fn = spearmanr if kind == "spearman" else pearsonr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    point = fn(x, y)
    stat = float(point.statistic if hasattr(point, "statistic") else point[0])
    pval = float(point.pvalue if hasattr(point, "pvalue") else point[1])
    if not np.isfinite(stat):
        return stat, pval, float("nan"), float("nan")

    a, b = (rankdata(x), rankdata(y)) if kind == "spearman" else (x, y)
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        xa, yb = a[idx], b[idx]
        xa -= xa.mean(axis=1, keepdims=True)
        yb -= yb.mean(axis=1, keepdims=True)
        num = (xa * yb).sum(axis=1)
        den = np.sqrt((xa ** 2).sum(axis=1) * (yb ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            boots[done:done + m] = np.where(den > 0, num / den, np.nan)
        done += m
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return stat, pval, float(lo), float(hi)


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_biodiversity(args.biodiversity_csvs)

    want_env = any([args.use_climate, args.use_soil, args.use_worldcover])
    ev = CombinedPatchClimateEvaluator(
        climate_data_path=args.climate_data_dir, use_climate_cache=True,
        climate_features_cache_dir=args.climate_cache_dir,
        soil_features_cache_dir=args.soil_cache_dir,
        geotessera_year=args.year, geotessera_patch_size=args.patch_size,
        satellite_source=args.satellite_source)
    X, y, locations, skipped, patch_dims, fnames, _dr = ev.prepare_dataset(
        df, representations_dir_path=args.representations_dir,
        worldcover_features_path=args.worldcover_path,
        use_satellite=True, use_climate=args.use_climate, use_soil=args.use_soil,
        use_worldcover=args.use_worldcover,
        satellite_dim_reduction="none", random_state=42)
    if X is None:
        raise SystemExit("prepare_dataset returned no data")

    names = np.array(fnames)
    sat_mask = np.char.startswith(names.astype(str), "patch_")
    coord_mask = np.isin(names, ["latitude", "longitude"])
    env_mask = ~(sat_mask | coord_mask)
    X_sat = X[:, sat_mask]
    print(f"satellite block {X_sat.shape}, environmental variables {int(env_mask.sum())}")

    # --- part 1: variance explained (needs only the embeddings) ---
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X_sat)
    k = int(min(args.n_components, Z.shape[1], Z.shape[0] - 1))
    pca = PCA(n_components=k, random_state=42).fit(Z)
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    comps = pca.transform(Z)

    def n_for(th):
        idx = np.argmax(cum >= th)
        return int(idx + 1) if cum[-1] >= th else None

    print(f"\nvariance explained by the first {min(10, k)} components:")
    for i in range(min(10, k)):
        print(f"  PC{i+1:<3d} {ratio[i]*100:6.2f}%   cumulative {cum[i]*100:6.2f}%")
    thresholds = {f"n_components_for_{int(t*100)}pct": n_for(t) for t in (0.5, 0.8, 0.9, 0.95)}
    print("  " + ", ".join(f"{kk.replace('n_components_for_','')}: {vv}"
                           for kk, vv in thresholds.items()))

    results = {
        "metadata": {
            "n_samples": int(X_sat.shape[0]), "n_raw_features": int(X_sat.shape[1]),
            "n_components": k, "patch_dims": list(patch_dims) if patch_dims else None,
        },
        "variance_explained": {
            "per_component": [float(v) for v in ratio],
            "cumulative": [float(v) for v in cum],
            **thresholds,
        },
        "correlations": [],
    }

    if want_env and env_mask.sum() > 0:
        env_names = names[env_mask]
        Xenv = X[:, env_mask]
        rows = []
        n_pc = min(args.n_report_components, k)
        print(f"\ncorrelating PC1..PC{n_pc} with {len(env_names)} environmental variables "
              f"({args.n_boot} bootstrap resamples each) ...")
        for pi in range(n_pc):
            for ei, en in enumerate(env_names):
                v = Xenv[:, ei]
                ok = np.isfinite(v) & np.isfinite(comps[:, pi])
                if ok.sum() < 50 or np.nanstd(v[ok]) == 0:
                    continue
                r, p, lo, hi = bootstrap_ci(comps[ok, pi], v[ok], "spearman",
                                            n_boot=args.n_boot, seed=pi * 1000 + ei)
                rows.append({"component": f"PC{pi+1}", "variable": str(en),
                             "spearman_r": r, "ci_low": lo, "ci_high": hi, "p_value": p,
                             "abs_r": abs(r)})
        rows.sort(key=lambda d: -d["abs_r"])
        results["correlations"] = rows
        if rows:
            print(f"\nstrongest 10 component-environment correlations:")
            print(f"{'component':10s} {'variable':28s} {'rho':>7s} {'95% CI':>18s}")
            for d in rows[:10]:
                print(f"{d['component']:10s} {d['variable'][:28]:28s} {d['spearman_r']:+7.3f} "
                      f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
            pd.DataFrame(rows).to_csv(out / "component_environment_correlations.csv", index=False)
    else:
        print("\n(no environmental covariates supplied - variance-explained section only)")

    pd.DataFrame({"component": [f"PC{i+1}" for i in range(k)],
                  "explained_variance_ratio": ratio,
                  "cumulative": cum}).to_csv(out / "variance_explained.csv", index=False)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    results_io.dump(out, "embedding_structure", stats=results,
                    arrays={"components": comps[:, :min(k, 20)].astype(np.float32)})
    print(f"\nwrote {out}/results.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representations-dir", default=paths.REPRESENTATIONS_DIR)
    p.add_argument("--biodiversity-csvs", nargs="+", default=paths.ECM_CSVS)
    p.add_argument("--out-dir", default="results/pca_structure")
    p.add_argument("--n-components", type=int, default=50,
                   help="Components to fit for the variance-explained curve.")
    p.add_argument("--n-report-components", type=int, default=10,
                   help="Components to correlate against environmental variables.")
    p.add_argument("--n-boot", type=int, default=2000)
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
