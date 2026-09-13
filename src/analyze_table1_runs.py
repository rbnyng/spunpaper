#!/usr/bin/env python
"""Paired analysis of ablation runs.

Every configuration was run over the same 50 seeds, so the comparisons between them are paired. Differencing per seed cancels the shared split variance.

This reads the per-run CSVs of each run directory and reports, for each configuration against the satellite-only model of the same algorithm:

  1 the mean paired difference and its 95% confidence interval
  2 a Wilcoxon signed-rank p-value (paired, distribution-free)

    python src/analyze_table1_runs.py --runs-dir <patch_climate_representation_results>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import results_io

CONFIG = {
    "sat(umap)": "Satellite only",
    "clim": "Climate only",
    "soil": "Soil only",
    "wc": "WorldCover only",
    "sat(umap)_clim_soil_wc": "Sat + Clim + Soil + WC",
}
MODEL = {"rf": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
REFERENCE = "Satellite only"
_DIR = re.compile(r"^(rf|lightgbm|xgboost)_(.+)_\d{8}-\d{6}$")


def block_of(feature: str) -> str:
    f = str(feature)
    if f.startswith(("pca_", "umap_", "patch_")):
        return "Satellite (SSL)"
    if f in ("latitude", "longitude"):
        return "Coordinates"
    if f.startswith(("soil_", "wc_class")) or "wrb" in f:
        return "Soil / land cover"
    return "Climate"

def load_runs(runs_dir: Path, metric="r2"):
    out, importances = {}, {}
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir():
            continue
        m = _DIR.match(d.name)
        if not m or CONFIG.get(m.group(2)) is None:
            continue
        key = (CONFIG[m.group(2)], MODEL[m.group(1)])
        f = d / "per_run_performance_metrics.csv"
        if f.exists():
            df = pd.read_csv(f).sort_values("run_seed")
            out[key] = (df["run_seed"].to_numpy(), df[metric].to_numpy(float))
        fi = d / "feature_importance_full.csv"
        if fi.exists():
            importances[key] = pd.read_csv(fi)
    return out, importances

def paired(a, b, alpha=0.05):
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:                                  # all-zero differences
        p = float("nan")
    return {"n_seeds": int(n), "mean_difference": float(d.mean()),
            "sd_difference": float(d.std(ddof=1)),
            "ci_low": float(d.mean() - t * se), "ci_high": float(d.mean() + t * se),
            "wilcoxon_p": p, "seeds_reference_wins": int((d > 0).sum())}

def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, importances = load_runs(Path(args.runs_dir), args.metric)
    if not runs:
        raise SystemExit(f"no run directories parsed under {args.runs_dir}")
    seeds = {tuple(v[0]) for v in runs.values()}
    if len(seeds) != 1:
        raise SystemExit("configurations do not share one seed vector; they cannot be paired.")
    n_seeds = len(next(iter(seeds)))
    print(f"{len(runs)} configurations, {n_seeds} shared seeds -> paired comparisons\n")

    marginal = {}
    for (cfg, model), (_, v) in sorted(runs.items()):
        marginal.setdefault(cfg, {})[model] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1))}

    models = sorted({m for _, m in runs})
    rows, wins, total = [], 0, 0
    for cfg in sorted({c for c, _ in runs}):
        if cfg == REFERENCE:
            continue
        for model in models:
            if (cfg, model) not in runs or (REFERENCE, model) not in runs:
                continue
            s = paired(runs[(REFERENCE, model)][1], runs[(cfg, model)][1])
            single = "+" not in cfg
            s.update({"baseline": cfg, "model": model, "single_source": single,
                      "direction": "satellite better" if s["mean_difference"] > 0
                                   else "baseline better"})
            rows.append(s)
            if single:
                wins += s["seeds_reference_wins"]
                total += s["n_seeds"]

    print(f"{'satellite vs':24s} {'model':14s} {'mean d':>8s} {'95% CI':>20s} "
          f"{'wins':>8s} {'p':>10s}")
    print("-" * 90)
    for r in sorted(rows, key=lambda r: (not r["single_source"], r["baseline"], r["model"])):
        print(f"{r['baseline']:24s} {r['model']:14s} {r['mean_difference']:+8.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"{r['seeds_reference_wins']:>3d}/{r['n_seeds']:<4d} {r['wilcoxon_p']:10.2e}")
    single = [r for r in rows if r["single_source"]]
    n_sep = sum(1 for r in single if r["ci_low"] > 0)
    print("-" * 90)
    print(f"satellite better in {n_sep}/{len(single)} model x baseline comparisons with the "
          f"95% CI excluding zero")
    print(f"and in {wins}/{total} ({wins/total:.1%}) individual seed-level comparisons")

    blocks = {}
    for (cfg, model), df in importances.items():
        if cfg != "Sat + Clim + Soil + WC" or "importance" not in df.columns:
            continue
        d = df.copy()
        d["block"] = d["feature"].map(block_of)
        g = d.groupby("block")["importance"].agg(["sum", "count"])
        g["share"] = g["sum"] / g["sum"].sum()
        blocks[model] = {b: {"share": float(r["share"]), "n_features": int(r["count"])}
                         for b, r in g.iterrows()}
    if blocks:
        print(f"\nfeature importance share in the all-sources model:")
        names = sorted({b for v in blocks.values() for b in v},
                       key=lambda b: -np.mean([blocks[m].get(b, {}).get("share", 0)
                                               for m in blocks]))
        print(f"{'block':20s} {'n':>5s} " + " ".join(f"{m:>14s}" for m in sorted(blocks)))
        for b in names:
            nfeat = next((blocks[m][b]["n_features"] for m in blocks if b in blocks[m]), 0)
            cells = " ".join(f"{blocks[m].get(b, {}).get('share', float('nan')):13.1%} "
                             for m in sorted(blocks))
            print(f"{b:20s} {nfeat:5d} {cells}")
        print("  (share is per block; split-count importance scales with block width, so "
              "read\n   these against the n column rather than per feature)")

    results = {
        "metadata": {"runs_dir": str(args.runs_dir), "metric": args.metric,
                     "n_seeds": n_seeds, "n_configurations": len(runs),
                     "reference": REFERENCE,
                     "design": "all configurations share one seed vector, so every "
                               "comparison is paired and the split variance cancels."},
        "marginal": marginal,
        "paired_vs_reference": rows,
        "summary": {
            "n_single_source_comparisons": len(single),
            "n_with_ci_excluding_zero": n_sep,
            "seed_level_wins": wins, "seed_level_total": total,
            "seed_level_win_rate": wins / total if total else None,
        },
        "importance_blocks": blocks,
    }
    pd.DataFrame(rows).to_csv(out / "paired_comparisons.csv", index=False)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    results_io.dump(out, "table1_paired", stats=results,
                    tables={"paired_comparisons": pd.DataFrame(rows)})
    print(f"\nwrote {out}/results.json")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True,
                   help="Directory of {model}_{sources}_{timestamp}/ run folders.")
    p.add_argument("--out-dir", default="results/table1_paired")
    p.add_argument("--metric", default="r2", help="Column in per_run_performance_metrics.csv.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
