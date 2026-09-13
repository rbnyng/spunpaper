from __future__ import annotations

import logging
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EARTH_RADIUS_KM = 6371.0088

Fold = Tuple[np.ndarray, np.ndarray]

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, dtype=float),
                                              np.asarray(lon1, dtype=float),
                                              np.asarray(lat2, dtype=float),
                                              np.asarray(lon2, dtype=float)))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nn_distance_to_train(coords: np.ndarray, test_idx: np.ndarray,
                         train_idx: np.ndarray) -> np.ndarray:
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.array([])
    te = coords[test_idx]
    tr = coords[train_idx]
    d = haversine_km(te[:, 0][:, None], te[:, 1][:, None], tr[:, 0][None, :], tr[:, 1][None, :])
    return d.min(axis=1)


def _groups_to_folds(groups: np.ndarray, n_splits: int,
                     rng: np.random.Generator) -> List[Fold]:
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    uniq, counts = uniq[order], counts[order]
    big_first = np.argsort(-counts)

    fold_of_group: Dict[int, int] = {}
    load = np.zeros(n_splits, dtype=np.int64)
    for gi in big_first:
        f = int(np.argmin(load))
        fold_of_group[uniq[gi]] = f
        load[f] += counts[gi]

    assign = np.array([fold_of_group[g] for g in groups])
    folds: List[Fold] = []
    for f in range(n_splits):
        test = np.where(assign == f)[0]
        train = np.where(assign != f)[0]
        if len(test) and len(train):
            folds.append((train, test))
    return folds

def random_kfold(coords: np.ndarray, n_splits: int = 5, seed: int = 42) -> List[Fold]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(coords))
    return [(np.setdiff1d(np.arange(len(coords)), part), part)
            for part in np.array_split(idx, n_splits)]


def location_group_kfold(coords: np.ndarray, n_splits: int = 5, seed: int = 42,
                         decimals: int = 6) -> List[Fold]:
    rng = np.random.default_rng(seed)
    key = np.round(coords, decimals)
    _, groups = np.unique(key, axis=0, return_inverse=True)
    return _groups_to_folds(groups, n_splits, rng)


def distance_group_kfold(coords: np.ndarray, link_m: float = 25.0, n_splits: int = 5,
                         seed: int = 42) -> List[Fold]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    rng = np.random.default_rng(seed)
    if link_m <= 0:
        return location_group_kfold(coords, n_splits, seed)

    # dedup first so the linkage matrix stays small.
    uniq, inv = np.unique(np.round(coords, 7), axis=0, return_inverse=True)
    if len(uniq) < 2:
        return location_group_kfold(coords, n_splits, seed)

    d = haversine_km(uniq[:, 0][:, None], uniq[:, 1][:, None],
                     uniq[:, 0][None, :], uniq[:, 1][None, :]) * 1000.0
    np.fill_diagonal(d, 0.0)
    lab_u = fcluster(linkage(squareform(d, checks=False), method="single"),
                     t=link_m, criterion="distance")
    groups = lab_u[inv]
    return _groups_to_folds(groups, n_splits, rng)


def spatial_block_kfold_m(coords: np.ndarray, block_m: float = 50.0, n_splits: int = 5,
                          seed: int = 42) -> List[Fold]:
    lat = coords[:, 0]
    lon = coords[:, 1]
    x_m = np.radians(lon) * EARTH_RADIUS_KM * 1000.0 * np.cos(np.radians(lat))
    y_m = np.radians(lat) * EARTH_RADIUS_KM * 1000.0
    bx = np.floor(x_m / float(block_m)).astype(np.int64)
    by = np.floor(y_m / float(block_m)).astype(np.int64)
    blocks = (bx + 2_000_000_000) * 4_000_000_000 + (by + 2_000_000_000)
    return _groups_to_folds(blocks, n_splits, np.random.default_rng(seed))


def apply_buffer(folds: Sequence[Fold], coords: np.ndarray, buffer_km: float) -> List[Fold]:
    out: List[Fold] = []
    for train, test in folds:
        if buffer_km <= 0:
            out.append((train, test))
            continue
        d = nn_distance_to_train(coords, train, test)  # train -> nearest test
        keep = train[d > buffer_km]
        if len(keep) >= 10 and len(test):
            out.append((keep, test))
    return out

def _default_model(seed: int):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)

def evaluate_folds(
    X_emb: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    folds: Sequence[Fold],
    n_components: Optional[int] = 16,
    use_coord_features: bool = False,
    model_factory: Callable[[int], object] = _default_model,
    seed: int = 42,
    return_predictions: bool = False,
    reducer: str = "pca",
    reducer_fit: str = "in_fold",
) -> Dict[str, float]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy.stats import spearmanr

    def _fit_reduce(Xtr_raw, Xte_raw, n_train):
        sc = StandardScaler().fit(Xtr_raw)
        Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
        if not n_components:
            return Xtr, Xte
        k = int(min(n_components, Xtr.shape[1], max(1, n_train - 1)))
        if reducer == "umap":
            import umap
            k = int(min(k, max(2, n_train - 2)))
            red = umap.UMAP(n_components=k, random_state=seed,
                            n_neighbors=int(min(15, max(2, n_train - 1))))
        else:
            red = PCA(n_components=k, random_state=seed)
        red.fit(Xtr)
        return red.transform(Xtr), red.transform(Xte)

    pre = None
    if reducer_fit == "global" and n_components:
        sc_g = StandardScaler().fit(X_emb)
        Zg = sc_g.transform(X_emb)
        kg = int(min(n_components, Zg.shape[1], max(1, len(Zg) - 2)))
        if reducer == "umap":
            import umap
            red_g = umap.UMAP(n_components=kg, random_state=seed).fit(Zg)
        else:
            red_g = PCA(n_components=kg, random_state=seed).fit(Zg)
        pre = red_g.transform(Zg)

    r2s, rmses, maes, spears, dists, null_r2s, n_trains = [], [], [], [], [], [], []
    pooled_true, pooled_pred = [], []
    pooled_fold, pooled_idx, pooled_nn = [], [], []

    for fold_i, (train, test) in enumerate(folds):
        Xtr_raw, Xte_raw = X_emb[train], X_emb[test]
        ytr, yte = y[train], y[test]
        if len(np.unique(yte)) < 2 or len(train) < 10:
            continue

        if pre is not None:
            Xtr, Xte = pre[train], pre[test]
        else:
            Xtr, Xte = _fit_reduce(Xtr_raw, Xte_raw, len(train))

        if use_coord_features:
            Xtr = np.hstack([Xtr, coords[train]])
            Xte = np.hstack([Xte, coords[test]])

        model = model_factory(seed)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        r2s.append(r2_score(yte, pred))
        rmses.append(float(np.sqrt(mean_squared_error(yte, pred))))
        maes.append(mean_absolute_error(yte, pred))
        sp = spearmanr(yte, pred).statistic
        spears.append(0.0 if np.isnan(sp) else float(sp))
        null_r2s.append(r2_score(yte, np.full_like(yte, ytr.mean(), dtype=float)))
        d = nn_distance_to_train(coords, test, train)
        dists.append(float(np.median(d)) if len(d) else np.nan)
        n_trains.append(len(train))
        pooled_true.append(yte)
        pooled_pred.append(pred)
        pooled_fold.append(np.full(len(test), fold_i, dtype=int))
        pooled_idx.append(np.asarray(test, dtype=int))
        pooled_nn.append(d if len(d) else np.full(len(test), np.nan))

    if not r2s:
        return {"n_folds": 0}

    pt = np.concatenate(pooled_true)
    pp = np.concatenate(pooled_pred)
    pooled_sp = spearmanr(pt, pp).statistic
    out = {
        "n_folds": len(r2s),
        "pooled_r2": float(r2_score(pt, pp)),
        "pooled_rmse": float(np.sqrt(mean_squared_error(pt, pp))),
        "pooled_mae": float(mean_absolute_error(pt, pp)),
        "pooled_spearman": float(0.0 if np.isnan(pooled_sp) else pooled_sp),
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "spearman_mean": float(np.mean(spears)),
        "null_r2_mean": float(np.mean(null_r2s)),
        "median_nn_dist_km": float(np.nanmedian(dists)),
        "mean_train_size": float(np.mean(n_trains)),
        "n_test_total": int(len(pt)),
    }
    if return_predictions:
        out["predictions"] = {
            "y_true": pt,
            "y_pred": pp,
            "fold": np.concatenate(pooled_fold),
            "sample_index": np.concatenate(pooled_idx),
            "nn_dist_km": np.concatenate(pooled_nn),
        }
    return out

RECOMMENDED_LINK_M = 25.0

def recommended_folds(coords: np.ndarray, n_splits: int = 5, seed: int = 42,
                      link_m: float = RECOMMENDED_LINK_M) -> List[Fold]:
    return distance_group_kfold(coords, link_m, n_splits, seed)


def report_cv(X: np.ndarray, y: np.ndarray, coords: np.ndarray, n_splits: int = 5,
              seeds: Sequence[int] = (1, 2, 3, 4, 5), n_components: Optional[int] = 16,
              use_coord_features: bool = False, link_m: float = RECOMMENDED_LINK_M
              ) -> Dict[str, Dict[str, float]]:
    def _avg(fold_fn):
        runs = [evaluate_folds(X, y, coords, fold_fn(s), n_components=n_components,
                               use_coord_features=use_coord_features)
                for s in seeds]
        runs = [r for r in runs if r.get("n_folds")]
        if not runs:
            return {}
        keys = ("pooled_r2", "pooled_rmse", "pooled_mae", "pooled_spearman",
                "median_nn_dist_km")
        out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        out["pooled_r2_sd"] = float(np.std([r["pooled_r2"] for r in runs]))
        return out

    return {
        "spatial": _avg(lambda s: recommended_folds(coords, n_splits, s, link_m)),
        "random_ref": _avg(lambda s: random_kfold(coords, n_splits, s)),
    }
