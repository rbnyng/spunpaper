#!/usr/bin/env python
"""Drive spun_train_patch.py across models x feature sets.

Produces the fifteen run directories behind the main results table: three models
(Random Forest, LightGBM, XGBoost) times five feature sets (each of satellite,
climate, soil and land cover alone, plus all four together). Latitude and
longitude are included in every configuration.

Satellite-bearing configurations reduce the satellite block with UMAP; the others
have no satellite block to reduce and pass --dim_reduction none.

Each run writes into RESULTS_DIR (config.env), which is where summarize_results.py
and the figure scripts look for it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

TRAIN_SCRIPT = Path(__file__).resolve().parent / "spun_train_patch.py"

MODELS = ["rf", "lightgbm", "xgboost"]
DATA_FLAGS = ["--use-satellite", "--use-climate", "--use-soil", "--use-worldcover"]


def build_configs():
    combos = [(f,) for f in DATA_FLAGS] + [tuple(DATA_FLAGS)]
    return [
        {"model": m, "sources": c,
         "dim_reduction": "umap" if "--use-satellite" in c else "none"}
        for m in MODELS for c in combos
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-runs", type=int, default=50,
                   help="Seeds per configuration (the manuscript uses 50).")
    p.add_argument("--results-dir", default=paths.RESULTS_DIR)
    p.add_argument("--log-dir", default="logs/ablations")
    p.add_argument("--dim-reduction-components", type=int, default=256)
    p.add_argument("--satellite-source", default="precomputed",
                   choices=["geotessera", "precomputed"],
                   help="'precomputed' reads the deposited {sample_id}.npy cache; "
                        "'geotessera' fetches embeddings on demand.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands and exit.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going if a configuration fails (default: stop).")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    print(f"{len(configs)} configurations x {args.num_runs} seeds "
          f"-> {args.results_dir}")

    failures = []
    for i, cfg in enumerate(configs, 1):
        cmd = [sys.executable, str(TRAIN_SCRIPT),
               "--num_runs", str(args.num_runs),
               "--model", cfg["model"],
               "--dim_reduction", cfg["dim_reduction"],
               "--dim_reduction_components", str(args.dim_reduction_components),
               "--satellite-source", args.satellite_source,
               "--results_dir", str(args.results_dir)]
        for flag in DATA_FLAGS:
            cmd.append(flag if flag in cfg["sources"]
                       else "--no-" + flag[2:])

        label = f"{cfg['model']}_" + "_".join(
            s.rsplit("-", 1)[-1] for s in cfg["sources"]) + f"_{cfg['dim_reduction']}"
        print(f"\n[{i}/{len(configs)}] {label}")
        print("  $ " + " ".join(cmd))
        if args.dry_run:
            continue

        log_path = log_dir / f"{label}.log"
        with open(log_path, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            print(f"  ok  (log: {log_path})")
        else:
            print(f"  FAILED (exit {rc}); see {log_path}", file=sys.stderr)
            failures.append(label)
            if not args.continue_on_error:
                sys.exit(rc)

    if failures:
        print(f"\n{len(failures)} configuration(s) failed: {failures}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(configs)} configurations completed.")


if __name__ == "__main__":
    main()
