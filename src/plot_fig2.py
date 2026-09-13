#!/usr/bin/env python
"""
    python src/plot_fig2.py \
        --runs-dir results/table1_runs \
        --importance-dir results/importance_2021 \
        --out manuscript/figures/fig2.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_STYLE = Path.home() / ".claude" / "skills" / "journal-plot-style" / "scripts"
if _STYLE.is_dir():
    sys.path.insert(0, str(_STYLE))
try:
    import journalstyle as js
except ImportError:
    js = None

REFERENCE = "Satellite only"
COMBINED = "Sat + Clim + Soil + WC"
MODELS = ["LightGBM", "XGBoost", "Random Forest"]
SHORT = {COMBINED: "All four\nmixed", "Satellite only": "Satellite\n10 m",
         "Climate only": "Climate\n1 km", "Soil only": "Soil\n250 m",
         "WorldCover only": "Land cover\n10 m"}
VS = {"Climate only": "Climate", "Soil only": "Soil",
      "WorldCover only": "Land cover"}
COLORS = {"LightGBM": "#2a9d8f", "XGBoost": "#e76f51", "Random Forest": "#5b7fbd"}
MARKERS = {"LightGBM": "o", "XGBoost": "s", "Random Forest": "D"}

CONFIG_OF = {"sat(umap)": "Satellite only", "clim": "Climate only", "soil": "Soil only",
             "wc": "WorldCover only", "sat(umap)_clim_soil_wc": COMBINED}
MODEL_OF = {"rf": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
_DIR = re.compile(r"^(rf|lightgbm|xgboost)_(.+)_\d{8}-\d{6}$")


def bunch_category(features, merge_climate=False):
    import pandas as pd
    f = pd.Series([str(x) for x in features])
    c = pd.Series(["Other"] * len(f))
    c[f.str.startswith(("pca_", "umap_"))] = "Satellite"
    c[f.str.startswith("patch_")] = "Satellite (raw)"
    c[f.str.startswith(("soil_", "wrb_"))] = "Soil"
    c[f.str.startswith("bio_")] = "Climate" if merge_climate else "Bioclimatic"
    c[f.isin(["elev", "slope", "aspect"])] = "Topography"
    c[f.str.contains("_annual_")] = "Climate"
    c[f.isin(["latitude", "longitude"])] = "Location"
    c[f.str.startswith("wc_class_")] = "Land cover"
    return c.to_numpy()

def load_runs(runs_dir):
    import pandas as pd
    table, per_seed, imp = {}, {}, {}
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir():
            continue
        m = _DIR.match(d.name)
        if m and CONFIG_OF.get(m.group(2)) is not None:
            cfg, model = CONFIG_OF[m.group(2)], MODEL_OF[m.group(1)]
        elif d.name in MODEL_OF:
            cfg, model = COMBINED, MODEL_OF[d.name]
        else:
            continue
        f = d / "per_run_performance_metrics.csv"
        if f.exists():
            v = pd.read_csv(f).sort_values("run_seed")["r2"].to_numpy(float)
            per_seed[(cfg, model)] = v
            # ddof=0
            table.setdefault(cfg, {})[model] = (float(v.mean()), float(v.std()))
        if cfg != COMBINED:
            continue
        npz = next((p for p in (d / "data" / "feature_importance.npz",
                                d / "feature_importance.npz") if p.exists()), None)
        csv = d / "feature_importance_full.csv"
        names = pd.read_csv(csv)["feature"].to_numpy() if csv.exists() else None
        if npz is not None and names is not None:
            with np.load(npz) as z:
                if "per_run_importance" in z:
                    imp[model] = (names, np.asarray(z["per_run_importance"], float))
                    continue
        if csv.exists():
            df = pd.read_csv(csv)
            imp[model] = (df["feature"].to_numpy(), df["importance"].to_numpy(float)[None, :])
    return table, per_seed, imp


def category_shares(imp, merge_climate=False):
    """{model: {category: (mean_share_pct, sd_pct)}}, plus feature counts."""
    shares, sizes, per_run = {}, {}, False
    for model, (names, runs) in imp.items():
        cats = bunch_category(names, merge_climate=merge_climate)
        tot = runs.sum(axis=1, keepdims=True)
        tot[tot == 0] = np.nan
        frac = 100.0 * runs / tot            # share within each run
        per_run = per_run or runs.shape[0] > 1
        shares[model] = {}
        for c in dict.fromkeys(cats):
            col = frac[:, cats == c].sum(axis=1)
            shares[model][c] = (float(np.nanmean(col)),
                                float(np.nanstd(col, ddof=1)) if len(col) > 1 else 0.0)
            sizes[c] = int((cats == c).sum())
    return shares, sizes, per_run


def build(table, out_png, per_seed=None, imp=None, title=None, merge_climate=False):
    import matplotlib.pyplot as plt
    if js is not None:
        js.use()
    n_panels = 2 if imp else 1
    fig = plt.figure(figsize=(7.4, 4.9 if imp else 2.8), dpi=200)
    gs = fig.add_gridspec(n_panels, 1, height_ratios=[1, 1][:n_panels], hspace=0.60)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0]) if imp else None
    axes = [a for a in (ax_a, ax_b) if a is not None]
    if js is not None:
        for a in axes:
            js.clean(a)
    for a in axes:                       # one type scale across both panels
        a.tick_params(labelsize=7)
        a.yaxis.label.set_size(8)
    off = {m: (i - 1) * 0.20 for i, m in enumerate(MODELS)}

    # --- (a) accuracy by feature set, 95% CI on the mean -------------------
    order = sorted(table, key=lambda k: -max(v[0] for v in table[k].values()))
    xpos = {k: i for i, k in enumerate(order)}
    for m in MODELS:
        xs, ys, es = [], [], []
        for k in order:
            if m not in table[k]:
                continue
            mu, sd = table[k][m]
            v = per_seed[(k, m)]
            mu, sd, n = float(v.mean()), float(v.std()), len(v)
            xs.append(xpos[k] + off[m]); ys.append(mu)
            es.append(1.96 * sd / np.sqrt(n))
        ax_a.errorbar(xs, ys, yerr=es, fmt="none", ecolor=COLORS[m], elinewidth=1.4,
                      capsize=0, alpha=0.85, zorder=2)
        ax_a.scatter(xs, ys, s=30, color=COLORS[m], marker=MARKERS[m],
                     edgecolor="white", linewidth=0.7, zorder=3, label=m)
    ax_a.set_xticks(list(xpos.values()))
    ax_a.set_xticklabels([SHORT.get(k, k) for k in order], fontsize=7)
    ax_a.set_xlim(-0.6, len(order) - 0.4)
    ax_a.set_ylabel("$R^2$   (95% CI)")
    ax_a.set_title("Out-of-sample $R^2$ by feature set and model",
                   loc="left", fontweight="bold", fontsize=8.5)
    if js is not None:
        js.style_legend(ax_a, loc="lower left", fontsize=6.5)
    else:
        ax_a.legend(loc="lower left", frameon=False, fontsize=6.5)

    # --- (b) importance share, all-sources model --------------------------
    if ax_b is not None:
        shares, sizes, per_run = category_shares(imp, merge_climate=merge_climate)
        cats = sorted({c for v in shares.values() for c in v},
                      key=lambda c: -np.mean([shares[m].get(c, (0, 0))[0] for m in shares]))
        xpos_b = {c: i for i, c in enumerate(cats)}
        width = 0.26
        for i, m in enumerate(MODELS):
            if m not in shares:
                continue
            xs = np.array([xpos_b[c] for c in cats], dtype=float) + (i - 1) * width
            ys = [shares[m].get(c, (np.nan, 0))[0] for c in cats]
            es = [shares[m].get(c, (np.nan, 0))[1] for c in cats]
            ax_b.bar(xs, ys, width * 0.92, color=COLORS[m], label=m, zorder=2,
                     yerr=es if per_run else None,
                     error_kw=dict(ecolor="#33393a", elinewidth=1.1, capsize=2.2, zorder=4))
        ax_b.set_xticks(list(xpos_b.values()))
        ax_b.set_xticklabels(cats, fontsize=7, rotation=45, ha="right",
                             rotation_mode="anchor")
        ax_b.set_xlim(-0.6, len(cats) - 0.4)
        ax_b.set_ylabel("share of total importance (%)")
        ax_b.set_ylim(0, None)
        ax_b.set_title("Share of total feature importance by feature group",
                       loc="left", fontweight="bold", fontsize=8.5)
        if not per_run:
            ax_b.text(0.99, 0.95, "run-averaged: per-run importances not in this folder",
                      transform=ax_b.transAxes, fontsize=6, color="#8a938f",
                      ha="right", va="top", style="italic")

    if js is not None:
        js.panel_labels(axes, x=-0.02, y=1.10)
    if title:
        fig.suptitle(title, x=0.005, ha="left", fontweight="bold")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(str(Path(out_png).with_suffix(".pdf")), bbox_inches="tight")
    print(f"wrote {out_png} and {Path(out_png).with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True,
                   help="Directory of per-run ablation folders. Panel (a) is computed "
                        "from the per-seed metrics found there.")
    p.add_argument("--importance-dir", default=None,
                   help="Where panel (b) reads importances from, if not --runs-dir. "
                        "Use this when the all-sources runs that carry per-run "
                        "importances live apart from the full feature-set grid.")
    p.add_argument("--out", default="manuscript/figures/table1.png")
    p.add_argument("--title", default=None)
    p.add_argument("--merge-climate", action="store_true",
                   help="fold bio_* into Climate, matching how Table 1 defines "
                        "the climate feature set; use whenever both panels show")
    a = p.parse_args()
    table, per_seed, imp = load_runs(a.runs_dir)
    if not table:
        raise SystemExit(
            f"no run directories matched under {a.runs_dir}. Expected subdirectories "
            f"named {{model}}_{{config}}_{{timestamp}}, e.g. "
            f"lightgbm_sat(umap)_20250827-201410. Nothing was plotted.")
    if a.importance_dir:
        imp = load_runs(a.importance_dir)[2] or imp
    missing = [(k, m) for k in table for m in MODELS if m not in table[k]]
    if missing:
        print(f"warning: {len(missing)} configuration/model pairs absent from "
              f"{a.runs_dir} and omitted from panel (a)")
    build(table, a.out, per_seed=per_seed, imp=imp or None, title=a.title,
          merge_climate=a.merge_climate)


if __name__ == "__main__":
    main()
