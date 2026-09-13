"""
End-to-end test on synthetic data.

Run: python3 tests_smoke.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_STUBS = Path(tempfile.mkdtemp(prefix="spun_stubs_"))
try:
    import richdem  # noqa: F401
except ImportError:
    (_STUBS / "richdem.py").write_text("# stub: see tests_smoke.py\n")
    sys.path.insert(0, str(_STUBS))

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

N = 260
PATCH = 3
DIM = 128
NCOMP = 8          # small, so UMAP is quick
RAW_SAT = PATCH * PATCH * DIM   # 1152

root = Path(tempfile.mkdtemp(prefix="spun_smoke_"))
cache = root / "reps"; cache.mkdir()

rng = np.random.default_rng(0)
lat = rng.uniform(45, 60, N)
lon = rng.uniform(5, 25, N)
sample_ids = [f"S{i:05d}" for i in range(N)]

latent = rng.normal(size=(N, 4))
for sid, z in zip(sample_ids, latent):
    base = np.repeat(z, DIM // 4)[None, None, :] * 0.5
    patch = base + rng.normal(scale=0.1, size=(PATCH, PATCH, DIM))
    np.save(cache / f"{sid}.npy", patch.astype(np.float32))

y = 100 + 30 * latent[:, 0] - 20 * latent[:, 1] + rng.normal(scale=5, size=N)
df = pd.DataFrame({"sample_id": sample_ids, "latitude": lat,
                   "longitude": lon, "rarefied": y})
eur = root / "eur.csv"; asia = root / "asia.csv"
df.iloc[:180].to_csv(eur, index=False)
df.iloc[180:].to_csv(asia, index=False)

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)

print("\n[1] prepare_dataset arity + reduction contract")
from spun_train_patch import (  # noqa: E402
    CombinedPatchClimateEvaluator, reduce_satellite_block,
)

ev = CombinedPatchClimateEvaluator(
    climate_data_path=None, use_climate_cache=False,
    climate_features_cache_dir=None, soil_features_cache_dir=None,
    satellite_source="precomputed",
)
out = ev.prepare_dataset(
    df, representations_dir_path=cache,
    use_satellite=True, use_climate=False, use_soil=False, use_worldcover=False,
    satellite_dim_reduction="umap", dim_reduction_components=NCOMP,
    random_state=42,
)
check("returns 7 values", len(out) == 7, f"got {len(out)}")
X, yv, locs, skipped, dims, fnames, dr = out
check("X is unreduced from prepare_dataset",
      X.shape[1] == RAW_SAT + 2, f"X={X.shape} (1152 satellite + 2 coords)")
check("dr_config carries the request",
      dr and dr["method"] == "umap" and dr["n_components"] == NCOMP
      and dr["n_satellite_raw"] == RAW_SAT, str(dr))

print("\n[2] train_and_evaluate applies the reduction in-fold")
res = ev.train_and_evaluate(X, yv, list(locs), fnames, random_seed=1,
                            model_name="lightgbm", dr_config=dr)
check("run succeeded", "error" not in res, res.get("error", ""))
r2 = res["test_stats"]["r2"]
check("R2 is sane on synthetic signal", 0.3 < r2 < 1.0, f"R2={r2:.3f}")
names = res.get("feature_names") or []
n_umap = sum(1 for n in names if str(n).startswith("umap_"))
check("satellite block became umap_* components",
      n_umap == NCOMP, f"{n_umap} umap_* of {len(names)} features")
check("evaluator persisted the fitted transform",
      ev.scaler is not None and ev.dim_reduction_model is not None)

print("\n[3] reduce_satellite_block (the per-fold helper for CV scripts)")
tr, te = np.arange(0, 200), np.arange(200, N)
Xtr, Xte = reduce_satellite_block(dr, X[tr], X[te])
check("train block reduced", Xtr.shape[1] == NCOMP + 2, f"{Xtr.shape}")
check("test block reduced to match", Xte.shape[1] == NCOMP + 2, f"{Xte.shape}")
check("row counts preserved", len(Xtr) == len(tr) and len(Xte) == len(te))
noop = reduce_satellite_block({"method": "none", "n_satellite_raw": RAW_SAT}, X[tr])
check("no-op when method='none'", noop.shape == X[tr].shape)

print("\n[4] fit_satellite_reduction (the inference path for map scripts)")
ev2 = CombinedPatchClimateEvaluator(
    climate_data_path=None, use_climate_cache=False,
    climate_features_cache_dir=None, soil_features_cache_dir=None,
    satellite_source="precomputed")
Xr = ev2.fit_satellite_reduction(X, dr)
check("returns reduced matrix", Xr.shape[1] == NCOMP + 2, f"{Xr.shape}")
check("scaler/model persisted for tile replay",
      ev2.scaler is not None and ev2.dim_reduction_model is not None)

replayed = ev2.dim_reduction_model.transform(ev2.scaler.transform(X[:, :RAW_SAT]))
_maxdiff = float(np.abs(replayed - Xr[:, :NCOMP]).max())
check("replay reproduces the stored transform",
      np.allclose(replayed, Xr[:, :NCOMP], atol=1e-4),
      "max|diff|={:.2e}".format(_maxdiff))
unseen = ev2.dim_reduction_model.transform(
    ev2.scaler.transform(rng.normal(size=(20, RAW_SAT))))
check("transforms unseen pixels (the tile-inference path)",
      unseen.shape == (20, NCOMP), str(unseen.shape))

print("\n[5] CLI entry points still parse (--help)")
for script in ["run_blocked_cv.py", "run_uncertainty.py", "run_error_outliers.py",
               "run_continent_transfer.py", "run_embedding_pca.py",
               "evaluate_feature_set.py", "run_patch_size_ablation.py",
               "run_year_ablation.py"]:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_STUBS), str(SRC), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    p = subprocess.run([sys.executable, str(SRC / script), "--help"],
                       capture_output=True, text=True, timeout=180, env=env)
    check(f"{script} --help", p.returncode == 0,
          (p.stderr.strip().splitlines() or [""])[-1][:90])

print("\n[6] BiomeFilter imports resolve (requests/zipfile/tempfile)")
import spun_train_patch as stp  # noqa: E402
check("requests imported", hasattr(stp, "requests"))
check("zipfile imported", hasattr(stp, "zipfile"))
check("tempfile imported", hasattr(stp, "tempfile"))
bf = stp.BiomeFilter(cache_dir=None)
check("BiomeFilter constructs without NameError", bf.cache_dir is not None)

shutil.rmtree(root, ignore_errors=True)
shutil.rmtree(_STUBS, ignore_errors=True)
print(f"\n{'='*58}")
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")
