#!/usr/bin/env python
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
from spun_train_patch import CombinedPatchClimateEvaluator, lgbm_fit  # noqa: E402


def load_biodiversity(csv_paths):
    frames = []
    for p in csv_paths:
        d = pd.read_csv(p)
        need = {"sample_id", "latitude", "longitude", "rarefied"}
        if not need.issubset(d.columns):
            raise ValueError(f"{p} missing {need - set(d.columns)}")
        keep = list(need | ({"continent"} & set(d.columns)))
        frames.append(d[keep])
    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["latitude", "longitude", "rarefied", "sample_id"])
    df["sample_id"] = df["sample_id"].astype(str)
    return df.drop_duplicates(subset=["sample_id"])


def fit_transform(Xfit, Xapply_list, method, n_components, seed):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xfit)
    if method == "pca":
        from sklearn.decomposition import PCA
        rd = PCA(n_components=n_components, random_state=seed)
    else:
        import umap
        rd = umap.UMAP(n_components=n_components, random_state=seed,
                       n_neighbors=15, min_dist=0.1, metric="euclidean")
    rd.fit(sc.transform(Xfit))
    return [np.asarray(rd.transform(sc.transform(X)), dtype=np.float32) for X in Xapply_list]


def fit_score(Ztr, ytr, Zte, yte, seed, use_val=True):
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from scipy.stats import spearmanr
    m = lgb.LGBMRegressor(random_state=seed, n_estimators=1000, learning_rate=0.05,
                          n_jobs=-1, verbose=-1)
    if use_val:
        tr, va = train_test_split(np.arange(len(Ztr)), test_size=0.125, random_state=seed)
        lgbm_fit(m, Ztr[tr], ytr[tr], eval_X=Ztr[va], eval_y=ytr[va],
                 stopping_rounds=15)
    else:
        m.fit(Ztr, ytr)
    p = m.predict(Zte)
    sp = spearmanr(yte, p).statistic
    return {"r2": float(r2_score(yte, p)),
            "rmse": float(np.sqrt(mean_squared_error(yte, p))),
            "mae": float(mean_absolute_error(yte, p)),
            "spearman": float(0.0 if np.isnan(sp) else sp)}, p


def run(args):
    out = Path(args.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    df = load_biodiversity(args.biodiversity_csvs)
    if "continent" not in df.columns:
        raise SystemExit("input CSVs need a 'continent' column")

    ev = CombinedPatchClimateEvaluator(
        climate_data_path=args.climate_data_dir or None,
        use_climate_cache=bool(args.climate_cache_dir),
        climate_features_cache_dir=args.climate_cache_dir,
        soil_features_cache_dir=args.soil_cache_dir,
        geotessera_year=args.year, geotessera_patch_size=args.patch_size,
        satellite_source=args.satellite_source)
    X, y, locations, skipped, patch_dims, fnames, _dr = ev.prepare_dataset(
        df, representations_dir_path=args.representations_dir,
        worldcover_features_path=args.worldcover_path,
        use_satellite=args.use_satellite, use_climate=args.use_climate,
        use_soil=args.use_soil, use_worldcover=args.use_worldcover,
        satellite_dim_reduction="none", random_state=42)
    if X is None:
        raise SystemExit("prepare_dataset returned no data")
    SAT_PREFIX = ("patch_", "pca_", "umap_")
    names = np.array([str(n) for n in fnames])
    coord_mask = np.isin(names, ["latitude", "longitude"])
    sat_mask = np.array([n.startswith(SAT_PREFIX) for n in names])
    env_mask = ~(coord_mask | sat_mask)
    X_sat, X_env, X_coord = X[:, sat_mask], X[:, env_mask], X[:, coord_mask]
    reduce_block = args.use_satellite and X_sat.shape[1] > args.n_components
    print(f"blocks: satellite {X_sat.shape[1]}, environmental {X_env.shape[1]}, "
          f"coordinates {X_coord.shape[1]}"
          + ("  (satellite reduced)" if reduce_block else "  (no reduction)"))

    def design(idx_tr, idx_te, seed):
        if reduce_block:
            zt, zs = fit_transform(X_sat[idx_tr], [X_sat[idx_tr], X_sat[idx_te]],
                                   args.dim_reduction, args.n_components, seed)
        else:
            zt, zs = X_sat[idx_tr], X_sat[idx_te]
        tr = np.hstack([zt, X_env[idx_tr], X_coord[idx_tr]]).astype(np.float32)
        te = np.hstack([zs, X_env[idx_te], X_coord[idx_te]]).astype(np.float32)
        return tr, te

    cont_by_id = dict(zip(df.sample_id, df.continent.astype(str).str.lower()))
    cont = np.array([cont_by_id.get(str(l["sample_id"]), "?") for l in locations])
    names = [c for c in ("europe", "asia") if (cont == c).sum() > 100]
    print(f"samples per continent: " +
          ", ".join(f"{c}={int((cont==c).sum())}" for c in names))

    results = {"metadata": {
        "n_samples": int(len(y)), "per_continent": {c: int((cont == c).sum()) for c in names},
        "n_components": args.n_components,
        "dim_reduction": args.dim_reduction if reduce_block else "none",
        "feature_sets": "+".join(k for k, v in (("sat", args.use_satellite),
                                                ("clim", args.use_climate),
                                                ("soil", args.use_soil),
                                                ("wc", args.use_worldcover)) if v) or "coords",
        "seeds": list(range(1, args.n_seeds + 1)),
    }, "configs": {}}

    def record(label, scores_list, preds=None, ytrue=None):
        agg = {f"{k}_mean": float(np.mean([s[k] for s in scores_list])) for k in scores_list[0]}
        agg.update({f"{k}_sd": float(np.std([s[k] for s in scores_list])) for k in scores_list[0]})
        agg["n_runs"] = len(scores_list)
        results["configs"][label] = agg
        if preds is not None:
            np.savez_compressed(out / "raw" / f"{label.replace(' ', '')}.npz",
                                y_true=ytrue, y_pred=preds)
        print(f"  {label:22s} R2={agg['r2_mean']:+.4f} +/- {agg['r2_sd']:.4f}  "
              f"Spearman={agg['spearman_mean']:+.4f}  MAE={agg['mae_mean']:6.2f}")

    print("\n=== within-continent references (random 80/20) ===")
    from sklearn.model_selection import train_test_split
    for c in names:
        idx = np.where(cont == c)[0]
        scores, last_p, last_y = [], None, None
        for s in range(1, args.n_seeds + 1):
            tr, te = train_test_split(idx, test_size=0.2, random_state=s)
            Ztr, Zte = design(tr, te, s)
            sc, p = fit_score(Ztr, y[tr], Zte, y[te], s)
            scores.append(sc); last_p, last_y = p, y[te]
        record(f"{c} -> {c}", scores, last_p, last_y)

    print("\n=== cross-continent transfer ===")
    for src in names:
        for dst in names:
            if src == dst:
                continue
            tr = np.where(cont == src)[0]
            te = np.where(cont == dst)[0]
            scores, last_p, last_y = [], None, None
            for s in range(1, args.n_seeds + 1):
                Ztr, Zte = design(tr, te, s)
                sc, p = fit_score(Ztr, y[tr], Zte, y[te], s)
                scores.append(sc); last_p, last_y = p, y[te]
            record(f"{src} -> {dst}", scores, last_p, last_y)

    cfg = results["configs"]
    drops = {}
    for src in names:
        for dst in names:
            if src != dst and f"{src} -> {dst}" in cfg and f"{dst} -> {dst}" in cfg:
                drops[f"{src}->{dst}"] = float(cfg[f"{src} -> {dst}"]["r2_mean"]
                                               - cfg[f"{dst} -> {dst}"]["r2_mean"])
    results["summary"] = {
        "r2_drop_vs_within_continent": drops,
    }
    with open(out / "results.json", "w") as f_:
        json.dump(results, f_, indent=2, default=float)
    pd.DataFrame(cfg).T.to_csv(out / "continent_transfer.csv")
    results_io.dump(out, "continent_transfer", stats=results)
    print(f"\ndrops vs destination's own baseline: "
          + ", ".join(f"{k} {v:+.4f}" for k, v in drops.items()))
    print(f"wrote {out}/results.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representations-dir", default=paths.REPRESENTATIONS_DIR)
    p.add_argument("--biodiversity-csvs", nargs="+", default=paths.ECM_CSVS)
    p.add_argument("--out-dir", default="results/continent_transfer")
    p.add_argument("--dim-reduction", default="umap", choices=["pca", "umap"])
    p.add_argument("--n-components", type=int, default=256)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--satellite-source", default="geotessera",
                   choices=["geotessera", "precomputed"],
                   help="'precomputed' reads patches straight from the cache dir.")
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--use-satellite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-climate", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-soil", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-worldcover", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--climate-data-dir", default=None)
    p.add_argument("--climate-cache-dir", default=None)
    p.add_argument("--soil-cache-dir", default=None)
    p.add_argument("--worldcover-path", default=None)
    p.add_argument("--year", type=int, default=2024)
    run(p.parse_args())


if __name__ == "__main__":
    main()
