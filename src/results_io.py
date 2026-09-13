from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

def _jsonable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)

def dump(results_dir: Union[str, Path], name: str, *,
         stats: Optional[Mapping[str, Any]] = None,
         arrays: Optional[Mapping[str, np.ndarray]] = None,
         tables: Optional[Mapping[str, Any]] = None,
         meta: Optional[Mapping[str, Any]] = None) -> Path:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    if stats is not None or meta is not None:
        payload: Dict[str, Any] = dict(stats or {})
        if meta:
            payload["meta"] = dict(meta)
        with open(out / f"{name}.json", "w") as f:
            json.dump(payload, f, indent=2, default=_jsonable)

    if arrays:
        clean = {}
        for k, v in arrays.items():
            try:
                clean[k] = np.asarray(v)
            except Exception:
                clean[k] = np.asarray(v, dtype=object)
        np.savez_compressed(out / f"{name}.npz", **clean)

    if tables:
        for k, t in tables.items():
            try:
                frame = t if (pd is not None and isinstance(t, pd.DataFrame)) \
                    else pd.DataFrame(t)
                frame.to_csv(out / f"{name}__{k}.csv", index=False)
            except Exception as exc:  # pragma: no cover
                logging.warning("results_io: could not write table '%s' of bundle "
                                "'%s': %s", k, name, exc)

    logging.info("results_io: wrote bundle '%s' to %s", name, out)
    return out

def load(results_dir: Union[str, Path], name: str) -> Dict[str, Any]:
    out = Path(results_dir)
    bundle: Dict[str, Any] = {"stats": {}, "arrays": {}, "tables": {}}

    jp = out / f"{name}.json"
    if jp.exists():
        with open(jp) as f:
            bundle["stats"] = json.load(f)

    npz = out / f"{name}.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=True) as z:
            bundle["arrays"] = {k: z[k] for k in z.files}

    if pd is not None:
        for csv in sorted(out.glob(f"{name}__*.csv")):
            key = csv.stem.split("__", 1)[1]
            try:
                bundle["tables"][key] = pd.read_csv(csv)
            except Exception as exc:  # pragma: no cover
                logging.warning("results_io: could not read %s: %s", csv, exc)
    return bundle

def list_bundles(results_dir: Union[str, Path]) -> Iterable[str]:
    out = Path(results_dir)
    if not out.is_dir():
        return []
    names = {p.stem for p in out.glob("*.json")}
    names |= {p.stem for p in out.glob("*.npz")}
    names |= {p.stem.split("__", 1)[0] for p in out.glob("*__*.csv")}
    return sorted(names)
