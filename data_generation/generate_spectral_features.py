#!/usr/bin/env python
"""
Spectral-index baseline features from raw Sentinel stacks.

Output format is identical to the Tessera patch cache
``{sample_id}.npy`` of shape ``(P, P, C)`` per sample so the result drops
straight into the existing evaluation with no other changes:

    python generate_spectral_features.py --out-dir spectral_features
    python run_blocked_cv.py --representations-dir spectral_features \\
        --satellite-source precomputed        # same protocol, different features

Features per pixel

  1 Sentinel-2 per-band annual statistics: mean, median, p10, p90, std over
    cloud-free observations (10 bands x 5 = 50)
  2 Vegetation indices NDVI, EVI, NDWI, NBR: mean, median, p10, p90, std, plus
    amplitude (p90 - p10) (4 x 6 = 24)
  3 Phenology from the NDVI trajectory: day-of-year of maximum, day-of-year of
    minimum, integrated NDVI, and the mean absolute successive difference as a
    roughness/greenness-dynamics term (4)
  4 Sentinel-1 VV and VH, ascending and descending: mean, median, std (4 x 3 = 12)

Expects the per-MGRS-tile stacks produced by `acquisition/` --
``bands.npy`` (T, H, W, 10), ``masks.npy`` (T, H, W), ``doys.npy`` (T,),
``sar_ascending.npy`` / ``sar_descending.npy`` (T, H, W, 2) and their DOY files --
plus a reference GeoTIFF per tile for georeferencing, exactly as the original
pipeline consumed them. Everything starts from what the Planetary Computer
serves, so the baseline can be rebuilt from public data by someone with no access
to this project's scratch space.

Band order is the acquisition pipeline's: B02, B03, B04, B05, B06, B07, B08, B8A,
B11, B12. ``masks.npy`` is 1 where the observation is clear (SCL-derived), the
convention ``ssl_dataset.py`` used; masked observations become NaN rather than
zero, so they drop out of the statistics instead of biasing them downward.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Sentinel-2 band order in bands.npy, as written by the acquisition pipeline.
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = 0, 1, 2, 6, 8, 9


def mgrs_tile(lat: float, lon: float):
    import mgrs
    try:
        return mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)[:5]
    except Exception:
        return None


def _stats(a: np.ndarray, axis=0):
    """mean, median, p10, p90, std along `axis`, NaN-safe."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.stack([np.nanmean(a, axis=axis), np.nanmedian(a, axis=axis),
                         np.nanpercentile(a, 10, axis=axis),
                         np.nanpercentile(a, 90, axis=axis),
                         np.nanstd(a, axis=axis)], axis=-1)


def spectral_features(s2, mask, doys, s1a, s1d, patch):
    """(T,P,P,10), (T,P,P), (T,), (Ta,P,P,2), (Td,P,P,2) -> (P,P,C) float32."""
    names = []
    out = []
    s2 = s2.astype(np.float32).copy()
    s2[mask.astype(bool)[..., None].repeat(s2.shape[-1], -1) == 0] = np.nan

    st = _stats(s2, axis=0)                                    # (P,P,10,5)
    out.append(st.reshape(patch, patch, -1))
    for b in S2_BANDS:
        names += [f"{b}_{s}" for s in ("mean", "median", "p10", "p90", "std")]

    def idx(num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(den) > 1e-6, num / den, np.nan)

    b = {n: s2[..., i] for i, n in enumerate(S2_BANDS)}
    ndvi = idx(b["B8"] - b["B4"], b["B8"] + b["B4"])
    evi = idx(2.5 * (b["B8"] - b["B4"]), b["B8"] + 6.0 * b["B4"] - 7.5 * b["B2"] + 1.0)
    ndwi = idx(b["B3"] - b["B8"], b["B3"] + b["B8"])
    nbr = idx(b["B8"] - b["B12"], b["B8"] + b["B12"])
    for nm, arr in (("NDVI", ndvi), ("EVI", evi), ("NDWI", ndwi), ("NBR", nbr)):
        s = _stats(arr, axis=0)                                # (P,P,5)
        amp = (s[..., 3] - s[..., 2])[..., None]               # p90 - p10
        out.append(np.concatenate([s, amp], axis=-1))
        names += [f"{nm}_{k}" for k in ("mean", "median", "p10", "p90", "std", "amplitude")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid = np.isfinite(ndvi)
        filled = np.where(valid, ndvi, np.nan)
        amax = np.nanargmax(np.where(valid, filled, -np.inf), axis=0)
        amin = np.nanargmin(np.where(valid, filled, np.inf), axis=0)
        doy_max = np.asarray(doys)[amax].astype(np.float32)
        doy_min = np.asarray(doys)[amin].astype(np.float32)
        integ = np.nansum(filled, axis=0).astype(np.float32)
        rough = np.nanmean(np.abs(np.diff(filled, axis=0)), axis=0).astype(np.float32)
    out.append(np.stack([doy_max, doy_min, integ, rough], axis=-1))
    names += ["NDVI_doy_max", "NDVI_doy_min", "NDVI_integral", "NDVI_roughness"]

    for tag, arr in (("VVasc_VHasc", s1a), ("VVdes_VHdes", s1d)):
        pol_names = tag.split("_")
        if arr is None or arr.size == 0:
            out.append(np.full((patch, patch, 6), np.nan, dtype=np.float32))
            names += [f"{p}_{s}" for p in pol_names for s in ("mean", "median", "std")]
            continue
        a = arr.astype(np.float32).copy()
        a[np.all(a == 0, axis=-1)[..., None].repeat(a.shape[-1], -1)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = np.stack([np.nanmean(a, axis=0), np.nanmedian(a, axis=0),
                          np.nanstd(a, axis=0)], axis=-1)       # (P,P,2,3)
        out.append(s.reshape(patch, patch, -1))
        names += [f"{p}_{k}" for p in pol_names for k in ("mean", "median", "std")]

    feats = np.concatenate([o.astype(np.float32) for o in out], axis=-1)
    return feats, names


def load_tile(tile_dir: Path):
    def opt(name, mmap=True):
        f = tile_dir / name
        if not f.exists():
            return None
        return np.load(f, mmap_mode="r" if mmap else None)
    return {"s2": opt("bands.npy"), "mask": opt("masks.npy"),
            "doys": opt("doys.npy", mmap=False),
            "s1a": opt("sar_ascending.npy"), "s1d": opt("sar_descending.npy")}


def pixel_index(lat, lon, ref_tiff):
    import rasterio
    from rasterio.warp import transform_geom
    import rasterio.transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(ref_tiff) as src:
            g = transform_geom("EPSG:4326", src.crs, {"type": "Point", "coordinates": (lon, lat)})
            x, y = g["coordinates"]
            r, c = rasterio.transform.rowcol(src.transform, x, y)
            return int(r), int(c), src.height, src.width


def find_reference(tile: str, roots):
    for root in roots:
        d = Path(root) / tile / "red"
        if d.is_dir():
            t = sorted(list(d.glob("*.tif")) + list(d.glob("*.tiff")))
            if t:
                return t[0]
    return None


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    half = args.patch_size // 2

    frames = [pd.read_csv(p) for p in args.biodiversity_csvs]
    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["latitude", "longitude", "sample_id"])
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.drop_duplicates("sample_id")
    logging.info("%d samples to process", len(df))

    df["tile"] = [mgrs_tile(a, b) for a, b in zip(df.latitude, df.longitude)]
    df = df[df.tile.notna()]

    written, skipped, feat_names = 0, [], None
    for tile, grp in df.groupby("tile"):
        tdir = None
        for root in args.tiles_roots:
            cand = Path(root) / tile
            if (cand / "bands.npy").exists():
                tdir = cand
                break
        if tdir is None:
            skipped += [(s, "tile stacks not found") for s in grp.sample_id]
            continue
        ref = find_reference(tile, args.reference_roots)
        if ref is None:
            skipped += [(s, "reference tiff not found") for s in grp.sample_id]
            continue
        data = load_tile(tdir)
        if data["s2"] is None or data["mask"] is None or data["doys"] is None:
            skipped += [(s, "incomplete S2 stack") for s in grp.sample_id]
            continue
        H, W = data["s2"].shape[1], data["s2"].shape[2]

        for row in grp.itertuples(index=False):
            fout = out / f"{row.sample_id}.npy"
            if fout.exists() and not args.refresh:
                written += 1
                continue
            try:
                r, c, _, _ = pixel_index(row.latitude, row.longitude, ref)
            except Exception as e:
                skipped.append((row.sample_id, f"georef failed: {e}"))
                continue
            r0, r1, c0, c1 = r - half, r + half + 1, c - half, c + half + 1
            if not (r0 >= 0 and c0 >= 0 and r1 <= H and c1 <= W):
                skipped.append((row.sample_id, "patch out of bounds"))
                continue
            s2 = np.asarray(data["s2"][:, r0:r1, c0:c1, :])
            mk = np.asarray(data["mask"][:, r0:r1, c0:c1])
            s1a = np.asarray(data["s1a"][:, r0:r1, c0:c1, :]) if data["s1a"] is not None else None
            s1d = np.asarray(data["s1d"][:, r0:r1, c0:c1, :]) if data["s1d"] is not None else None
            try:
                feats, names = spectral_features(s2, mk, data["doys"], s1a, s1d, args.patch_size)
            except Exception as e:
                skipped.append((row.sample_id, f"feature computation failed: {e}"))
                continue
            # Impute any all-NaN channel (e.g. a pixel with no clear observation)
            # with the patch mean so downstream code sees finite values.
            if not np.isfinite(feats).all():
                col_mean = np.nanmean(feats.reshape(-1, feats.shape[-1]), axis=0)
                col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
                feats = np.where(np.isfinite(feats), feats, col_mean)
            np.save(fout, feats.astype(np.float32))
            feat_names = names
            written += 1
        logging.info("tile %s: %d/%d written", tile, written, len(df))

    meta = {"n_written": written, "n_skipped": len(skipped),
            "patch_size": args.patch_size,
            "n_features_per_pixel": len(feat_names) if feat_names else None,
            "feature_names": feat_names,
            "skipped_examples": skipped[:20]}
    with open(out / "_feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logging.info("wrote %d patches (%d skipped); %s features per pixel",
                 written, len(skipped), meta["n_features_per_pixel"])
    logging.info("evaluate with: run_blocked_cv.py --representations-dir %s "
                 "--satellite-source precomputed", out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--biodiversity-csvs", nargs="+", default=paths.ECM_CSVS)
    p.add_argument("--tiles-roots", nargs="+", required=True,
                   help="Directories holding per-MGRS-tile stacks (bands.npy etc).")
    p.add_argument("--reference-roots", nargs="+", required=True,
                   help="Directories holding <TILE>/red/*.tif reference rasters.")
    p.add_argument("--out-dir", default="spectral_features")
    p.add_argument("--patch-size", type=int, default=3)
    p.add_argument("--refresh", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
