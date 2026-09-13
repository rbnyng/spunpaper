"""
Shared, overridable paths for the pipeline.

Precedence highest first:

1. an environment variable of the same name, e.g. ``SPUN_DATA_ROOT=/my/data``
2. the value in ``config.env``
3. the fallback baked in below

Point at a different config file with ``SPUN_CONFIG=/path/to/other.env``. The
same file is sourced directly by the shell drivers in ``data_generation/acquisition``, so the
two languages cannot drift apart.

Values are plain strings (not ``Path``) because that is what the call sites want
-- argparse defaults, ``Path(...)`` constructors and f-strings all accept them.

    from paths import ECM_EUROPE_CSV, REPRESENTATIONS_DIR
    parser.add_argument("--representations_dir", default=REPRESENTATIONS_DIR)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

_FALLBACKS: Dict[str, str] = {
    "SPUN_ROOT": "",           # filled in by load(); see repo_root()
    "SPUN_DATA_ROOT": "/maps-priv/maps/ray25/data",
    # DEPOSIT_DIR belongs with the roots rather than with the leaves: several
    # cache keys expand ${DEPOSIT_DIR}, and this dict is resolved in order, so a
    # deposit-based config only works if it is defined before its dependants.
    "DEPOSIT_DIR": "",
    "SPUN_SCRATCH": "/scratch/ray25",
    "SPUN_RESULTS_ROOT": "/maps-priv/maps/ray25/config_src",
    "SPUN_NATIONALPARKS": "/maps-priv/maps/ray25/nationalparks",
    "ECM_EUROPE_CSV": "${SPUN_DATA_ROOT}/spun_data/ECM_richness_europe.csv",
    "ECM_ASIA_CSV": "${SPUN_DATA_ROOT}/spun_data/ECM_richness_Asia.csv",
    "REPRESENTATIONS_DIR": "${SPUN_DATA_ROOT}/ecm_representations",
    "REPRESENTATIONS_P7_DIR": "${SPUN_DATA_ROOT}/ecm_tessera_p7",
    "SPECTRAL_REPRESENTATIONS_DIR": "${SPUN_DATA_ROOT}/spectral_representations",
    "YEAR_CACHE_TEMPLATE": "${SPUN_DATA_ROOT}/caches/tessera_{year}_p3",
    "WORLDCLIM_DIR": "${SPUN_DATA_ROOT}/worldclim/data",
    "CLIMATE_CACHE_DIR": "${SPUN_SCRATCH}/climate_features_cache",
    "CLIMATE_METADATA_CACHE": "${SPUN_SCRATCH}/climate_metadata_cache_v3.pkl",
    "SOIL_CACHE_DIR": "${SPUN_SCRATCH}/soil_features_cache",
    "WORLDCOVER_CSV": "${SPUN_SCRATCH}/worldcover_features.csv",
    "WORLDCOVER_TILES": "${SPUN_SCRATCH}/worldcover_tiles",
    "ECOREGIONS_CACHE": "${SPUN_SCRATCH}/ecoregions_cache",
    "RESULTS_DIR": "${SPUN_RESULTS_ROOT}/patch_climate_representation_results",
    "PATCH_PROC_TMP": "${SPUN_SCRATCH}/spun_patch_proc/tmp",
    "ACQ_PYTHON_ENV": "/maps/zf281/miniconda3/envs/detectree-env/bin/python",
    "ACQ_DATA_ROOT": "/scratch/zf281/create_d-pixels_biomassters/data",
    "ACQ_LOG_DIR": "${ACQ_DATA_ROOT}/test_agbm_d-pixel/logs_s2",
}

_LINE = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*"?\$\{\1:-(.*?)\}"?\s*$')
_PLAIN = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*"?([^"#]*?)"?\s*$')

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def config_path() -> Path:
    env = os.environ.get("SPUN_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config.env"

def _read_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line) or _PLAIN.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out

def _expand(value: str, resolved: Dict[str, str], depth: int = 0) -> str:
    if depth > 10:
        return value

    def sub(m):
        key = m.group(1)
        if key in resolved:
            return resolved[key]
        return os.environ.get(key, m.group(0))

    new = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", sub, value)
    return new if new == value else _expand(new, resolved, depth + 1)

def load() -> Dict[str, str]:
    from_file = _read_file(config_path())
    resolved: Dict[str, str] = {}
    from_file.setdefault("SPUN_ROOT", str(repo_root()))
    for key, fallback in _FALLBACKS.items():
        raw = os.environ.get(key) or from_file.get(key) or fallback
        resolved[key] = _expand(raw, resolved)
    for key, raw in from_file.items():          # keys unknown to _FALLBACKS
        if key not in resolved:
            resolved[key] = _expand(os.environ.get(key) or raw, resolved)
    return resolved

_CONFIG = load()

def get(key: str, default: str | None = None) -> str:
    return os.environ.get(key) or _CONFIG.get(key) or (default or "")

# Module-level constants, so call sites read as `paths.ECM_EUROPE_CSV`.
SPUN_DATA_ROOT = _CONFIG["SPUN_DATA_ROOT"]
SPUN_SCRATCH = _CONFIG["SPUN_SCRATCH"]
SPUN_RESULTS_ROOT = _CONFIG["SPUN_RESULTS_ROOT"]
SPUN_NATIONALPARKS = _CONFIG["SPUN_NATIONALPARKS"]
ECM_EUROPE_CSV = _CONFIG["ECM_EUROPE_CSV"]
ECM_ASIA_CSV = _CONFIG["ECM_ASIA_CSV"]
ECM_CSVS = [ECM_EUROPE_CSV, ECM_ASIA_CSV]
REPRESENTATIONS_DIR = _CONFIG["REPRESENTATIONS_DIR"]
REPRESENTATIONS_P7_DIR = _CONFIG["REPRESENTATIONS_P7_DIR"]
SPECTRAL_REPRESENTATIONS_DIR = _CONFIG["SPECTRAL_REPRESENTATIONS_DIR"]
YEAR_CACHE_TEMPLATE = _CONFIG["YEAR_CACHE_TEMPLATE"]
WORLDCLIM_DIR = _CONFIG["WORLDCLIM_DIR"]
CLIMATE_CACHE_DIR = _CONFIG["CLIMATE_CACHE_DIR"]
CLIMATE_METADATA_CACHE = _CONFIG["CLIMATE_METADATA_CACHE"]
SOIL_CACHE_DIR = _CONFIG["SOIL_CACHE_DIR"]
WORLDCOVER_CSV = _CONFIG["WORLDCOVER_CSV"]
WORLDCOVER_TILES = _CONFIG["WORLDCOVER_TILES"]
ECOREGIONS_CACHE = _CONFIG["ECOREGIONS_CACHE"]
RESULTS_DIR = _CONFIG["RESULTS_DIR"]
PATCH_PROC_TMP = _CONFIG["PATCH_PROC_TMP"]
ACQ_PYTHON_ENV = _CONFIG["ACQ_PYTHON_ENV"]
ACQ_DATA_ROOT = _CONFIG["ACQ_DATA_ROOT"]
ACQ_LOG_DIR = _CONFIG["ACQ_LOG_DIR"]


if __name__ == "__main__": 
    print(f"config file: {config_path()}"
          f"{'' if config_path().is_file() else '  (not found - using fallbacks)'}\n")
    for k, v in _CONFIG.items():
        src = "env" if os.environ.get(k) else "file/default"
        print(f"  {k:26s} = {v}   [{src}]")
