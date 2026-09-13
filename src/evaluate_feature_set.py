#!/usr/bin/env python
"""
Score a satellite feature set.

Takes any directory of per-sample ``(P, P, C)`` patches and reports what a model trained on it achieves, so that two feature sets can be compared with everything else held fixed. The embeddings and the spectral-index baseline are both evaluated this way.
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
from spun_train_patch import CombinedPatchClimateEvaluator  # noqa: E402

PAPER_TARGETS = {
    "rf":       {"r2": 0.512, "r2_sd": 0.017, "rmse": 48.16, "mae": 29.20},
    "lightgbm": {"r2": 0.535, "r2_sd": 0.018, "rmse": 47.01, "mae": 29.07},
    "xgboost":  {"r2": 0.529, "r2_sd": 0.017, "rmse": 47.30, "mae": 29.33},
}

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

def summarise(runs):
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return {"n_runs": 0, "errors": [r.get("error") for r in runs][:3]}
    st = pd.DataFrame([r["test_stats"] for r in ok])
    out = {f"{c}_mean": float(st[c].mean()) for c in st.columns}
    out.update({f"{c}_sd": float(st[c].std()) for c in st.columns})
    out["n_runs"] = len(ok)
    return out

def run(args):
    out = Path(args.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    df = load_biodiversity(args.biodiversity_csvs)
    print(f"biodiversity records: {len(df)}")

    evaluator = CombinedPatchClimateEvaluator(
        climate_data_path=None, use_climate_cache=False,
        climate_features_cache_dir=None, soil_features_cache_dir=None,
        geotessera_year=args.year, geotessera_patch_size=args.patch_size,
        satellite_source=args.satellite_source,
    )

    print(f"\npreparing dataset (patch {args.patch_size}x{args.patch_size}, "
          f"{args.dim_reduction} -> {args.dim_reduction_components}) ...")
    t0 = time.time()
    X, y, locations, skipped, patch_dims, fnames, dr_config = evaluator.prepare_dataset(
        df, representations_dir_path=args.representations_dir,
        use_satellite=True, use_climate=False, use_soil=False, use_worldcover=False,
        satellite_dim_reduction=args.dim_reduction,
        dim_reduction_components=args.dim_reduction_components,
        random_state=42,
    )
    prep_s = time.time() - t0
    if X is None:
        raise SystemExit("prepare_dataset returned no data")
    print(f"  X={X.shape}  y={y.shape}  patch_dims={patch_dims}  "
          f"skipped={len(skipped)}  ({prep_s:.0f}s)")

    assert fnames[-2:] == ["latitude", "longitude"], fnames[-2:]

    results = {
        "metadata": {
            "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
            "patch_dims": list(patch_dims) if patch_dims else None,
            "dim_reduction": args.dim_reduction,
            "dim_reduction_components": args.dim_reduction_components,
            "year": args.year, "n_runs": args.n_runs,
            "prepare_seconds": prep_s,
            "n_input_records": int(len(df)),
            "n_skipped": len(skipped),
            "protocol": "70/10/20 random splits with early stopping, as in the paper",
            "paper_targets_table1_satellite_only": PAPER_TARGETS,
        },
        "per_model": {},
    }

    print("\n=== satellite-only ===")
    print(f"{'model':10s} {'R2':>18s} {'RMSE':>8s} {'MAE':>8s} | {'paper R2':>10s} {'delta':>8s}")
    for model in args.models:
        runs = [evaluator.train_and_evaluate(X, y, list(locations), fnames,
                                             random_seed=s, model_name=model,
                                             dr_config=dr_config)
                for s in range(1, args.n_runs + 1)]
        s = summarise(runs)
        results["per_model"][model] = s
        if not s.get("n_runs"):
            print(f"{model:10s} FAILED: {s.get('errors')}")
            continue
        tgt = PAPER_TARGETS.get(model, {})
        d = s["r2_mean"] - tgt.get("r2", float("nan"))
        print(f"{model:10s} {s['r2_mean']:+.4f} +/- {s['r2_sd']:.4f} "
              f"{s['rmse_mean']:8.2f} {s['mae_mean']:8.2f} | "
              f"{tgt.get('r2', float('nan')):10.3f} {d:+8.4f}")
        np.savez_compressed(out / "raw" / f"runs_{model}.npz",
                            r2=np.array([r["test_stats"]["r2"] for r in runs if "error" not in r]),
                            rmse=np.array([r["test_stats"]["rmse"] for r in runs if "error" not in r]),
                            mae=np.array([r["test_stats"]["mae"] for r in runs if "error" not in r]))

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    results_io.dump(out, "feature_set_eval",
                    stats=results,
                    arrays={"y": y, "X": X},
                    tables={"locations": pd.DataFrame(locations)})
    print(f"\nwrote {out}/results.json")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representations-dir", required=True)
    p.add_argument("--biodiversity-csvs", nargs="+", required=True)
    p.add_argument("--out-dir", default="feature_set_eval_results")
    p.add_argument("--satellite-source", default="geotessera",
                   choices=["geotessera", "precomputed"],
                   help="'precomputed' loads {sample_id}.npy straight from --representations-dir with no fetching.")
    p.add_argument("--patch-size", type=int, default=3,
                   help="3 matches the manuscript's stated 3x3 window.")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--dim-reduction", default="umap", choices=["none", "pca", "umap"],
                   help="umap matches the manuscript.")
    p.add_argument("--dim-reduction-components", type=int, default=256)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--models", nargs="+", default=["rf"],
                   choices=["rf", "lightgbm", "xgboost"])
    run(p.parse_args())


if __name__ == "__main__":
    main()
