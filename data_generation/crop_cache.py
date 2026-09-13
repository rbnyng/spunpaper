#!/usr/bin/env python
"""
center-crop a patch cache to a smaller patch size, without refetching.

    python data_generation/crop_cache.py --src ecm_tessera_p7 --dst ecm_tessera_p3 --size 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def crop_one(a: np.ndarray, size: int) -> np.ndarray:
    h, w = a.shape[0], a.shape[1]
    if size > h or size > w:
        raise ValueError(f"cannot crop {h}x{w} up to {size}x{size}")
    top, left = (h - size) // 2, (w - size) // 2
    return a[top:top + size, left:left + size, :]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Existing cache of wider patches.")
    p.add_argument("--dst", required=True, help="Directory to write the cropped cache.")
    p.add_argument("--size", type=int, required=True, help="Target patch width, e.g. 3.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-crop files that already exist in --dst.")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")
    if src.resolve() == dst.resolve():
        raise SystemExit("--src and --dst must differ; cropping in place would destroy the "
                         "wider cache the ablation needs.")
    # Learn the source width from one file so a wrong --size fails immediately
    # rather than once per file across a cache of tens of thousands.
    probe = next((e.path for e in os.scandir(src) if e.name.endswith(".npy")), None)
    if probe is None:
        raise SystemExit(f"no .npy files in {src}")
    shape = np.load(probe, mmap_mode="r").shape
    if len(shape) != 3:
        raise SystemExit(f"{probe} has shape {shape}; expected (H, W, C)")
    if args.size > min(shape[0], shape[1]):
        raise SystemExit(f"--size {args.size} exceeds the cached patch width "
                         f"{shape[0]}x{shape[1]} (from {Path(probe).name}); "
                         f"cropping cannot invent pixels.")
    print(f"cropping {shape[0]}x{shape[1]}x{shape[2]} -> "
          f"{args.size}x{args.size}x{shape[2]}")
    dst.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    with os.scandir(src) as it:
        for e in it:
            if not e.name.endswith(".npy"):
                continue
            target = dst / e.name
            if target.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                a = np.load(e.path)
                if a.ndim != 3:
                    failed += 1
                    continue
                if a.shape[0] == args.size and a.shape[1] == args.size:
                    np.save(target, a)          # already the right size; copy through
                else:
                    np.save(target, crop_one(a, args.size))
                done += 1
            except Exception as exc:                              # noqa: BLE001
                print(f"  {e.name}: {type(exc).__name__}: {exc}")
                failed += 1
            if (done + skipped) % 5000 == 0 and done + skipped:
                print(f"  {done + skipped:,} processed ...", flush=True)

    print(f"wrote {done:,} cropped patches to {dst}"
          + (f", skipped {skipped:,} already present" if skipped else "")
          + (f", {failed:,} failed" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
