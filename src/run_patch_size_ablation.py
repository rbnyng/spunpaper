#!/usr/bin/env python
"""
Patch-size ablation for embeddings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spatial_cv_strategies import (  # noqa: E402
    RECOMMENDED_LINK_M, distance_group_kfold, evaluate_folds,
)

TESSERA_DIM = 128

def centre_crop(patches: np.ndarray, native: int, target: int) -> np.ndarray:
    if target > native:
        raise ValueError(f"cannot crop {native}x{native} up to {target}x{target}")
    off = (native - target) // 2
    return patches[:, off:off + target, off:off + target, :].reshape(len(patches), -1)

def load_patches(cache_dir: Path, csv_paths, native: int):
    frames = []
    for p in csv_paths:
        d = pd.read_csv(p)
        need = {"sample_id", "latitude", "longitude", "rarefied"}
        if not need.issubset(d.columns):
            raise ValueError(f"{p} missing {need - set(d.columns)}")
        frames.append(d[list(need)])
    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["sample_id", "latitude", "longitude", "rarefied"])
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.drop_duplicates(subset=["sample_id"])

    arrs, keep = [], []
    want = (native, native, TESSERA_DIM)
    for r in df.itertuples(index=False):
        f = cache_dir / f"{r.sample_id}.npy"
        ok = False
        if f.exists():
            a = np.load(f)
            if tuple(a.shape) == want and np.isfinite(a).all():
                arrs.append(a); ok = True
        keep.append(ok)
    df = df[keep].reset_index(drop=True)
    if not arrs:
        raise SystemExit(
            f"no cached patch in {cache_dir} matched the expected shape {want}. "
            f"Check --native-patch-size (currently {native}); the loader also requires "
            f"{TESSERA_DIM} channels, so this ablation is Tessera-specific.")
    return df, np.asarray(arrs, dtype=np.float32)

def run(args):
    out = Path(args.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    native = args.native_patch_size
    sizes = [s for s in args.sizes if s <= native]

    df, patches = load_patches(Path(args.cache_dir), args.biodiversity_csvs, native)
    y = df.rarefied.to_numpy(float)
    C = df[["latitude", "longitude"]].to_numpy(float)
    print(f"{len(df)} samples with {native}x{native} patches; evaluating sizes {sizes}")

    seeds = list(range(1, args.n_seeds + 1))
    # Identical folds for every patch size -> paired comparison.
    if args.cv == "random":
        from sklearn.model_selection import KFold
        fold_sets = {s: list(KFold(args.n_splits, shuffle=True, random_state=s)
                             .split(np.arange(len(df)))) for s in seeds}
        cv_desc = f"random KFold(n_splits={args.n_splits})"
    else:
        fold_sets = {s: distance_group_kfold(C, RECOMMENDED_LINK_M, args.n_splits, s)
                     for s in seeds}
        cv_desc = f"distance_group_kfold(link_m={RECOMMENDED_LINK_M})"

    results = {
        "metadata": {
            "n_samples": int(len(df)), "native_patch_size": native, "sizes": sizes,
            "n_splits": args.n_splits, "seeds": seeds,
            "n_components": args.n_components,
            "dim_reduction": args.dim_reduction,
            "reducer_fit": args.reducer_fit,
            "cv": cv_desc,
        },
        "per_size": [], "paired_comparisons": [],
    }

    # per-sample absolute error, averaged over seeds, for each size (paired bootstrap)
    per_sample_ae = {}
    for p in sizes:
        X = centre_crop(patches, native, p)
        runs, ae_acc, raw_saved = [], np.zeros((len(df), len(seeds))), False
        for si, s in enumerate(seeds):
            m = evaluate_folds(X, y, C, fold_sets[s], n_components=args.n_components,
                               use_coord_features=args.use_coords,
                               return_predictions=True, seed=s,
                               reducer=args.dim_reduction,
                               reducer_fit=args.reducer_fit)
            if not m.get("n_folds"):
                continue
            pr = m.pop("predictions")
            ae = np.full(len(df), np.nan)
            ae[pr["sample_index"]] = np.abs(pr["y_true"] - pr["y_pred"])
            ae_acc[:, si] = ae
            if not raw_saved:
                np.savez_compressed(out / "raw" / f"patch{p}_seed{s}.npz", **pr,
                                    latitude=C[pr["sample_index"], 0],
                                    longitude=C[pr["sample_index"], 1])
                raw_saved = True
            runs.append(m)
        if not runs:
            continue
        per_sample_ae[p] = np.nanmean(ae_acc, axis=1)
        agg = {k: float(np.mean([r[k] for r in runs]))
               for k in ("pooled_r2", "pooled_rmse", "pooled_mae", "pooled_spearman")}
        agg.update(patch_size=p, n_features=int(X.shape[1]),
                   pooled_r2_sd=float(np.std([r["pooled_r2"] for r in runs])),
                   pooled_spearman_sd=float(np.std([r["pooled_spearman"] for r in runs])),
                   pooled_r2_per_seed=[float(r["pooled_r2"]) for r in runs],
                   n_seeds=len(runs))
        results["per_size"].append(agg)
        print(f"  {p}x{p} ({X.shape[1]:5d} feats)  R2={agg['pooled_r2']:+.4f}"
              f" +/- {agg['pooled_r2_sd']:.4f}   Spearman={agg['pooled_spearman']:+.4f}"
              f"   MAE={agg['pooled_mae']:.2f}")

    rng = np.random.default_rng(0)
    by_size = {r["patch_size"]: r for r in results["per_size"]}
    print("\npaired comparisons (same folds, same samples):")
    for i, a in enumerate(sizes):
        for b in sizes[i + 1:]:
            if a not in by_size or b not in by_size:
                continue
            ra = np.array(by_size[a]["pooled_r2_per_seed"])
            rb = np.array(by_size[b]["pooled_r2_per_seed"])
            n = min(len(ra), len(rb))
            d_r2 = rb[:n] - ra[:n]
            # paired bootstrap over samples on mean absolute error
            ae_a, ae_b = per_sample_ae[a], per_sample_ae[b]
            ok = np.isfinite(ae_a) & np.isfinite(ae_b)
            diff = ae_b[ok] - ae_a[ok]          # negative => bigger patch is better
            boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                             for _ in range(args.n_boot)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            comp = {
                "smaller": a, "larger": b,
                "delta_r2_mean": float(d_r2.mean()), "delta_r2_sd": float(d_r2.std()),
                "delta_mae_mean": float(diff.mean()),
                "delta_mae_ci95": [float(lo), float(hi)],
                "mae_difference_significant": bool(lo > 0 or hi < 0),
            }
            results["paired_comparisons"].append(comp)
            verdict = ("larger better" if hi < 0 else
                       "smaller better" if lo > 0 else "no significant difference")
            print(f"  {a}x{a} -> {b}x{b}:  dR2={d_r2.mean():+.4f}+/-{d_r2.std():.4f}"
                  f"   dMAE={diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   {verdict}")

    if results["per_size"]:
        best = max(results["per_size"], key=lambda r: r["pooled_r2"])
        sig = [c for c in results["paired_comparisons"] if c["mae_difference_significant"]]
        results["summary"] = {
            "best_by_r2": best["patch_size"],
            "best_r2": best["pooled_r2"],
            "n_significant_pairs": len(sig),
            "conclusion": (
                f"{best['patch_size']}x{best['patch_size']} scored highest, but "
                f"{len(sig)} of {len(results['paired_comparisons'])} pairwise MAE "
                "differences are significant at 95%; patch size has little effect "
                "where differences are not significant."
                if len(sig) < len(results["paired_comparisons"])
                else f"{best['patch_size']}x{best['patch_size']} is best and differences "
                     "are significant."),
        }

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    pd.DataFrame(results["per_size"]).to_csv(out / "per_size.csv", index=False)
    if results["paired_comparisons"]:
        pd.DataFrame(results["paired_comparisons"]).to_csv(
            out / "paired_comparisons.csv", index=False)
    np.savez_compressed(out / "raw" / "per_sample_abs_error.npz",
                        **{f"patch{p}": v for p, v in per_sample_ae.items()},
                        latitude=C[:, 0], longitude=C[:, 1], y=y)
    print(f"\nwrote {out}/results.json + per_size.csv + raw/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--native-patch-size", type=int, required=True)
    p.add_argument("--biodiversity-csvs", nargs="+", required=True)
    p.add_argument("--out-dir", default="results/patch_size")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-components", "--pca-components", dest="n_components",
                   type=int, default=256,
                   help="Components kept by the reducer. 256 is the main pipeline's "
                        "setting; smaller values are faster but not comparable to the "
                        "reported table.")
    p.add_argument("--dim-reduction", choices=["pca", "umap"], default="umap",
                   help="umap is the main pipeline's reducer; pca is a faster, "
                        "deterministic alternative for exploratory runs.")
    p.add_argument("--reducer-fit", choices=["in_fold", "global"], default="in_fold",
                   help="'in_fold' refits the reducer inside each training fold, as the "
                        "main pipeline does. 'global' fits once on all rows and is "
                        "provided only for diagnosing the cost of that choice.")
    p.add_argument("--cv", choices=["random", "grouped"], default="random",
                   help="'random' matches the main pipeline's splits; 'grouped' is the "
                        "spatially blocked robustness variant.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--use-coords", action="store_true",
                   help="Append lat/lon as predictors (off by default so the "
                        "comparison isolates the embedding patch).")
    a = p.parse_args()
    a.n_components = a.n_components or None
    run(a)


if __name__ == "__main__":
    main()
