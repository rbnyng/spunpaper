"""
Per-sample Tessera embedding patches
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try: 
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Tessera embeddings are 128-dimensional at 10 m resolution.
TESSERA_DIM = 128
DEFAULT_YEAR = 2024
DEFAULT_PATCH_SIZE = 3
_M_PER_DEG_LAT = 111_320.0
_TESSERA_PIXEL_M = 10.0


def _sanitize_sample_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "_").replace("\\", "_").replace("\x00", "")


class GeoTesseraPatchExtractor:
    """Fetch Tessera embedding patches for point samples, with on-disk caching.

    Parameters
    ----------
    year:
        Tessera embedding year to fetch (Tessera publishes 2017-2025).
    patch_size:
        Spatial extent of the extracted patch in pixels (10 m each). Odd values
        give a well-defined centre pixel. ``1`` yields a single-pixel embedding.
    cache_dir:
        Directory for the per-sample ``{sample_id}.npy`` patch cache. This is
        also the directory that can be passed to the downstream pipeline as
        ``representations_dir``.
    embeddings_dir:
        Directory geotessera uses to persist downloaded tiles. Defaults to
        ``<cache_dir>/tessera_tiles`` so the large tile downloads live beside the
        (small) per-sample cache rather than in the current working directory.
    dataset_version, dataset_variant:
        Passed straight through to :class:`geotessera.GeoTessera`
        (defaults ``"v1"`` / ``"vultr"``).
    fill_value:
        Value used to pad the patch when it extends past available coverage
        (e.g. the very edge of the mosaic). ``np.nan`` by default; the downstream
        pipeline imputes NaNs with column means.
    gt:
        Optional pre-constructed ``GeoTessera`` instance (mainly for testing).
    """

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        patch_size: int = DEFAULT_PATCH_SIZE,
        cache_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
        dataset_version: str = "v1",
        dataset_variant: str = "vultr",
        registry_dir: Optional[Union[str, Path]] = None,
        fill_value: float = np.nan,
        gt: Optional[object] = None,
        tile_mem_cache: int = 1,
        edge_mode: str = "neighbors",
        allow_mixed_cache: bool = False,
    ):
        if not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("patch_size must be a positive integer.")
        if patch_size % 2 == 0:
            logging.warning(
                "patch_size (%d) is even; odd patch sizes give a clear centre pixel.",
                patch_size,
            )

        self.year = int(year)
        self.patch_size = int(patch_size)
        self.half_patch = self.patch_size // 2
        self.dataset_version = dataset_version
        self.dataset_variant = dataset_variant
        self.registry_dir = Path(registry_dir) if registry_dir else None
        self.fill_value = fill_value
        if edge_mode not in ("neighbors", "pad", "mosaic"):
            raise ValueError("edge_mode must be 'neighbors', 'pad' or 'mosaic'")
        self.edge_mode = edge_mode

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if not allow_mixed_cache:
                self._assert_cache_patch_size()

        if embeddings_dir is not None:
            self.embeddings_dir = Path(embeddings_dir)
        elif self.cache_dir is not None:
            self.embeddings_dir = self.cache_dir / "tessera_tiles"
        else:
            self.embeddings_dir = Path("tessera_tiles")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._gt = gt
        self._transformers: Dict[str, object] = {}
        self._rowcol = None

        self.tile_mem_cache = max(int(tile_mem_cache), 0)
        self._tile_mem: "OrderedDict[Tuple[float, float, int], tuple]" = OrderedDict()

    def _assert_cache_patch_size(self, sample: int = 25) -> None:
        found: Dict[Tuple[int, ...], int] = {}
        n = 0
        try:
            with os.scandir(self.cache_dir) as it:
                for e in it:
                    if not e.name.endswith(".npy"):
                        continue
                    try:
                        shp = tuple(np.load(e.path, mmap_mode="r").shape)
                    except Exception:  # noqa: BLE001 - unreadable entries are handled per-sample
                        continue
                    found[shp] = found.get(shp, 0) + 1
                    n += 1
                    if n >= sample:
                        break
        except OSError:
            return
        bad = {s: c for s, c in found.items()
               if len(s) == 3 and s[2] == TESSERA_DIM and s[0] != self.patch_size}
        if bad and not any(s[0] == self.patch_size for s in found):
            shapes = ", ".join(f"{s[0]}x{s[1]}" for s in bad)
            raise ValueError(
                f"{self.cache_dir} already holds {shapes} patches but this run wants "
                f"{self.patch_size}x{self.patch_size}. The cache is keyed by sample_id "
                f"only, so continuing would refetch every sample and overwrite it. "
                f"Use a separate cache directory per patch size (e.g. "
                f"{self.cache_dir.name}_p{self.patch_size}), or pass "
                f"allow_mixed_cache=True / --allow-mixed-cache to override."
            )

    # The geotessera client is created lazily, so importing this module stays
    # cheap and works offline.
    @property
    def gt(self):
        if self._gt is None:
            try:
                from geotessera import GeoTessera
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "The 'geotessera' package is required for Tessera embeddings. "
                    "Install it with `pip install geotessera` (requires Python >= 3.12)."
                ) from exc
            logging.info(
                "Initialising GeoTessera (version=%s, variant=%s, embeddings_dir=%s)",
                self.dataset_version,
                self.dataset_variant,
                self.embeddings_dir,
            )
            kwargs = dict(
                dataset_version=self.dataset_version,
                dataset_variant=self.dataset_variant,
                embeddings_dir=str(self.embeddings_dir),
            )
            if self.registry_dir is not None:
                kwargs["registry_dir"] = str(self.registry_dir)
            self._gt = GeoTessera(**kwargs)
        return self._gt

    def _rowcol_fn(self):
        if self._rowcol is None:
            from rasterio.transform import rowcol

            self._rowcol = rowcol
        return self._rowcol

    def _to_crs_xy(self, crs, lon: float, lat: float):
        key = str(crs)
        transformer = self._transformers.get(key)
        if transformer is None:
            from pyproj import Transformer

            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            self._transformers[key] = transformer
        return transformer.transform(lon, lat)

    def cache_path(self, sample_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{_sanitize_sample_id(sample_id)}.npy"

    def get_patch(
        self,
        lat: float,
        lon: float,
        sample_id: Optional[str] = None,
        refresh: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Return a (patch_size, patch_size, 128) float32 patch, or None.
        """
        if lat is None or lon is None or (isinstance(lat, float) and math.isnan(lat)) \
                or (isinstance(lon, float) and math.isnan(lon)):
            logging.warning("Invalid coordinates for sample %s: (%s, %s)", sample_id, lat, lon)
            return None

        cpath = self.cache_path(sample_id) if sample_id is not None else None
        if cpath is not None and cpath.exists() and not refresh:
            try:
                patch = np.load(cpath)
                if self._valid_shape(patch):
                    return patch.astype(np.float32, copy=False)
                logging.warning(
                    "Cached patch %s has shape %s, expected (%d, %d, %d); refetching.",
                    cpath, patch.shape, self.patch_size, self.patch_size, TESSERA_DIM,
                )
            except Exception as exc:  # corrupt cache file -> refetch
                logging.warning("Failed to read cached patch %s (%s); refetching.", cpath, exc)

        try:
            patch = self._fetch_patch(lat, lon)
        except Exception as exc:  # missing tile / no coverage / network issue
            logging.warning("Tessera fetch failed for sample %s (%s, %s): %s",
                            sample_id, lat, lon, exc)
            return None

        if patch is None:
            return None

        patch = patch.astype(np.float32, copy=False)
        if cpath is not None:
            try:
                np.save(cpath, patch)
            except Exception as exc:  # pragma: no cover - disk issues
                logging.warning("Failed to write patch cache %s: %s", cpath, exc)
        return patch

    def _valid_shape(self, patch: np.ndarray) -> bool:
        return (
            isinstance(patch, np.ndarray)
            and patch.ndim == 3
            and patch.shape[0] == self.patch_size
            and patch.shape[1] == self.patch_size
            and patch.shape[2] == TESSERA_DIM
        )

    def _load_tile(self, lon: float, lat: float):
        """Return (embedding, crs, transform) for the tile holding a point, memoized.
        """
        key = (*self.tile_of(lon, lat), self.year)
        hit = self._tile_mem.get(key)
        if hit is not None:
            self._tile_mem.move_to_end(key)
            return hit

        result = self.gt.fetch_embedding(lon=lon, lat=lat, year=self.year)
        if self.tile_mem_cache > 0:
            self._tile_mem[key] = result
            while len(self._tile_mem) > self.tile_mem_cache:
                self._tile_mem.popitem(last=False)
        return result

    def _fetch_patch(self, lat: float, lon: float) -> Optional[np.ndarray]:
        # Fetch the single 0.1-degree tile that contains the point. geotessera
        # snaps arbitrary coordinates to the enclosing tile centre and returns
        # the dequantized (H, W, 128) array plus its rasterio CRS + transform.
        embedding, crs, transform = self._load_tile(lon, lat)
        H, W = embedding.shape[0], embedding.shape[1]

        x, y = self._to_crs_xy(crs, lon, lat)
        row, col = self._rowcol_fn()(transform, x, y)
        row, col = int(row), int(col)

        r0, r1 = row - self.half_patch, row + self.half_patch + 1
        c0, c1 = col - self.half_patch, col + self.half_patch + 1

        if r0 >= 0 and c0 >= 0 and r1 <= H and c1 <= W:
            # Common case: whole patch lies inside this tile's native grid.
            return np.array(embedding[r0:r1, c0:c1, :], dtype=np.float32)

        logging.debug("Patch for (%.5f, %.5f) crosses a tile edge (mode=%s).",
                      lat, lon, self.edge_mode)
        if self.edge_mode == "pad":
            return self._padded_slice(embedding, row, col)
        if self.edge_mode == "mosaic":
            return self._fetch_patch_via_mosaic(lat, lon, crs)
        return self._fetch_patch_via_neighbors(lat, lon, embedding, crs, transform, row, col)

    def _fetch_patch_via_neighbors(self, lat, lon, embedding, crs, transform, row, col):
        """Fill an edge-crossing patch by splicing in the adjacent tile(s).
        """
        from rasterio.transform import xy as rio_xy

        out = np.full((self.patch_size, self.patch_size, embedding.shape[2]),
                      self.fill_value, dtype=np.float32)

        # World coordinates (in the primary tile's CRS) of every patch pixel.
        rows = np.arange(row - self.half_patch, row + self.half_patch + 1)
        cols = np.arange(col - self.half_patch, col + self.half_patch + 1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        xs, ys = rio_xy(transform, rr.ravel(), cc.ravel())
        xs = np.asarray(xs, dtype=float).reshape(rr.shape)
        ys = np.asarray(ys, dtype=float).reshape(rr.shape)

        filled = np.zeros(rr.shape, dtype=bool)

        def take_from(emb, e_crs, e_transform):
            """Copy whatever this tile can supply into the unfilled patch cells."""
            if str(e_crs) == str(crs):
                px, py = xs, ys
            else:
                # Different UTM zone: go via lon/lat.
                from pyproj import Transformer

                to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lo, la = to_wgs.transform(xs.ravel(), ys.ravel())
                fwd = Transformer.from_crs("EPSG:4326", e_crs, always_xy=True)
                pxr, pyr = fwd.transform(lo, la)
                px = np.asarray(pxr).reshape(xs.shape)
                py = np.asarray(pyr).reshape(ys.shape)
            inv = ~e_transform
            fc, fr = inv * (px, py)
            r_i = np.floor(np.asarray(fr)).astype(int)
            c_i = np.floor(np.asarray(fc)).astype(int)
            ok = ((~filled) & (r_i >= 0) & (r_i < emb.shape[0])
                  & (c_i >= 0) & (c_i < emb.shape[1]))
            if ok.any():
                out[ok] = emb[r_i[ok], c_i[ok], :]
                filled[ok] = True

        take_from(embedding, crs, transform)

        if not filled.all():
            t_lon, t_lat = self.tile_of(lon, lat)
            for dlon in (-0.1, 0.0, 0.1):
                for dlat in (-0.1, 0.0, 0.1):
                    if filled.all():
                        break
                    if dlon == 0.0 and dlat == 0.0:
                        continue
                    try:
                        n_emb, n_crs, n_tr = self._load_tile(round(t_lon + dlon, 2),
                                                             round(t_lat + dlat, 2))
                    except Exception:
                        continue  # neighbour tile absent (coast, no coverage)
                    take_from(n_emb, n_crs, n_tr)

        if not filled[self.half_patch, self.half_patch]:
            return None  # centre pixel itself unavailable
        return out

    def _fetch_patch_via_mosaic(self, lat: float, lon: float, crs) -> Optional[np.ndarray]:
        # Pad the query box by the patch half-width (plus one pixel of margin),
        # expressed in degrees, so the returned mosaic definitely contains the
        # full patch even for points near a tile corner.
        pad_m = (self.half_patch + 1) * _TESSERA_PIXEL_M
        pad_lat = pad_m / _M_PER_DEG_LAT
        cos_lat = max(math.cos(math.radians(lat)), 1e-6)
        pad_lon = pad_m / (_M_PER_DEG_LAT * cos_lat)
        bbox = (lon - pad_lon, lat - pad_lat, lon + pad_lon, lat + pad_lat)

        mosaic, transform, mcrs = self.gt.fetch_mosaic_for_region(
            bbox, year=self.year, target_crs=str(crs)
        )
        x, y = self._to_crs_xy(mcrs, lon, lat)
        row, col = self._rowcol_fn()(transform, x, y)
        return self._padded_slice(mosaic, int(row), int(col))

    def _padded_slice(self, arr: np.ndarray, row: int, col: int) -> Optional[np.ndarray]:
        """Slice a (P, P, C) patch centred on (row, col), padding out-of-bounds."""
        H, W = arr.shape[0], arr.shape[1]
        if not (0 <= row < H and 0 <= col < W):
            # Centre pixel itself is outside coverage -> treat as a miss.
            return None

        out = np.full((self.patch_size, self.patch_size, arr.shape[2]),
                      self.fill_value, dtype=np.float32)
        r0, r1 = row - self.half_patch, row + self.half_patch + 1
        c0, c1 = col - self.half_patch, col + self.half_patch + 1
        sr0, sc0 = max(r0, 0), max(c0, 0)
        sr1, sc1 = min(r1, H), min(c1, W)
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0, :] = arr[sr0:sr1, sc0:sc1, :]
        return out

    @staticmethod
    def tile_of(lon: float, lat: float) -> Tuple[float, float]:
        """Return the (tile_lon, tile_lat) centre of the 0.1-degree tile holding a point."""
        try:
            from geotessera.registry import tile_from_world

            return tile_from_world(lon, lat)
        except Exception:
            # Same rule as geotessera, without requiring the import.
            return (round(math.floor(lon * 10) / 10 + 0.05, 2),
                    round(math.floor(lat * 10) / 10 + 0.05, 2))

    def _tile_files(self, tile_lon: float, tile_lat: float) -> List[Path]:
        """On-disk files geotessera creates for one tile (embedding, scales, landmask)."""
        base = Path(self.embeddings_dir)
        try:
            from geotessera.registry import (
                tile_to_embedding_paths,
                tile_to_landmask_filename,
            )

            emb_rel, scales_rel = tile_to_embedding_paths(tile_lon, tile_lat, self.year)
            landmask = tile_to_landmask_filename(tile_lon, tile_lat)
            return [
                base / "global_0.1_degree_representation" / emb_rel,
                base / "global_0.1_degree_representation" / scales_rel,
                base / "global_0.1_degree_tiff_all" / landmask,
            ]
        except Exception:
            grid = f"grid_{tile_lon:.2f}_{tile_lat:.2f}"
            rep = base / "global_0.1_degree_representation" / str(self.year) / grid
            return [
                rep / f"{grid}.npy",
                rep / f"{grid}_scales.npy",
                base / "global_0.1_degree_tiff_all" / f"{grid}.tiff",
            ]

    def evict_tile(self, tile_lon: float, tile_lat: float) -> int:
        """Delete the cached tile rasters for one tile. Returns bytes freed.
        """
        freed = 0
        for path in self._tile_files(tile_lon, tile_lat):
            try:
                if path.exists():
                    freed += path.stat().st_size
                    path.unlink()
            except Exception as exc:  # pragma: no cover
                logging.debug("Could not evict %s: %s", path, exc)
        try:
            grid_dir = (Path(self.embeddings_dir) / "global_0.1_degree_representation"
                        / str(self.year) / f"grid_{tile_lon:.2f}_{tile_lat:.2f}")
            if grid_dir.is_dir() and not any(grid_dir.iterdir()):
                shutil.rmtree(grid_dir, ignore_errors=True)
        except Exception:  # pragma: no cover
            pass
        return freed

    def extract_for_dataframe(
        self,
        df,
        id_col: str = "sample_id",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        refresh: bool = False,
        show_progress: bool = True,
        prune_tiles: bool = False,
    ) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}
        n_missing = 0
        freed_total = 0

        groups: Dict[Tuple[float, float], List[Tuple[str, float, float]]] = {}
        for sample_id, lat, lon in df[[id_col, lat_col, lon_col]].itertuples(index=False):
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (TypeError, ValueError):
                logging.warning("Invalid coordinates for sample %s; skipping.", sample_id)
                n_missing += 1
                continue
            groups.setdefault(self.tile_of(lon_f, lat_f), []).append(
                (str(sample_id), lat_f, lon_f)
            )

        logging.info(
            "%d samples span %d distinct 0.1-degree tiles (%.1f samples/tile).",
            sum(len(v) for v in groups.values()), len(groups),
            (sum(len(v) for v in groups.values()) / len(groups)) if groups else 0.0,
        )

        tile_iter = sorted(groups.items())
        if show_progress:
            tile_iter = tqdm(tile_iter, total=len(groups), desc="Tessera tiles")

        for (tile_lon, tile_lat), members in tile_iter:
            # If every sample in this tile is already cached, skip it entirely --
            # no download, no tile touched.
            if not refresh and all(
                (p := self.cache_path(sid)) is not None and p.exists() for sid, _, _ in members
            ):
                for sid, _, _ in members:
                    try:
                        patch = np.load(self.cache_path(sid))
                        if self._valid_shape(patch):
                            results[sid] = patch.astype(np.float32, copy=False)
                        else:
                            n_missing += 1
                    except Exception:
                        n_missing += 1
                continue

            downloaded_here = False
            for sid, lat_f, lon_f in members:
                cpath = self.cache_path(sid)
                was_cached = cpath is not None and cpath.exists() and not refresh
                patch = self.get_patch(lat=lat_f, lon=lon_f, sample_id=sid, refresh=refresh)
                if not was_cached:
                    downloaded_here = True
                if patch is None:
                    n_missing += 1
                    continue
                results[sid] = patch

            self._tile_mem.pop((tile_lon, tile_lat, self.year), None)
            if prune_tiles and downloaded_here:
                freed_total += self.evict_tile(tile_lon, tile_lat)

        msg = ("Fetched %d/%d Tessera patches across %d tiles (%d without coverage)."
               % (len(results), len(df), len(groups), n_missing))
        if prune_tiles:
            msg += f" Evicted {freed_total/1e9:.2f} GB of intermediate tile data."
        logging.info(msg)
        return results


def build_filtered_registry(
    points,
    output_dir: Union[str, Path],
    year: int = DEFAULT_YEAR,
    dataset_version: str = "v1",
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if cache_dir is None:
        env = os.environ.get("XDG_CACHE_HOME")
        cache_dir = Path(env) / "geotessera" if env else Path.home() / ".cache" / "geotessera"
    src = Path(cache_dir) / dataset_version
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = set()
    for lon, lat in points:
        t_lon, t_lat = GeoTesseraPatchExtractor.tile_of(float(lon), float(lat))
        keys.add(int(round(t_lon * 100)) * 100_000 + int(round(t_lat * 100)))
    wanted = np.fromiter(keys, dtype=np.int64)
    logging.info("Filtering registry to %d tiles for year %d", len(keys), year)

    def _key(frame):
        return (frame.lon_i.astype(np.int64) * 100_000 + frame.lat_i.astype(np.int64)).to_numpy()

    man = pq.read_table(src / "manifest.parquet").to_pandas()
    keep = (man.year == year).to_numpy() & np.isin(_key(man), wanted)
    pq.write_table(pa.Table.from_pandas(man[keep], preserve_index=False),
                   out / "manifest.parquet")
    logging.info("  manifest: %d of %d rows", int(keep.sum()), len(man))
    del man

    lm_path = src / "landmasks.parquet"
    if lm_path.exists():
        lm = pq.read_table(lm_path).to_pandas()
        if {"lon_i", "lat_i"}.issubset(lm.columns):
            lm = lm[np.isin(_key(lm), wanted)]
        pq.write_table(pa.Table.from_pandas(lm, preserve_index=False),
                       out / "landmasks.parquet")
        logging.info("  landmasks: %d rows", len(lm))
    return out


def generate_representations(
    csv_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    year: int = DEFAULT_YEAR,
    patch_size: int = DEFAULT_PATCH_SIZE,
    embeddings_dir: Optional[Union[str, Path]] = None,
    dataset_version: str = "v1",
    dataset_variant: str = "vultr",
    registry_dir: Optional[Union[str, Path]] = None,
    id_col: str = "sample_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    refresh: bool = False,
    prune_tiles: bool = False,
    allow_mixed_cache: bool = False,
) -> int:
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = {id_col, lat_col, lon_col} - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        frames.append(frame[[id_col, lat_col, lon_col]])
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[id_col, lat_col, lon_col])
    df[id_col] = df[id_col].astype(str)
    df = df.drop_duplicates(subset=[id_col], keep="first")
    logging.info("Loaded %d unique sample locations from %d CSV(s).", len(df), len(csv_paths))

    extractor = GeoTesseraPatchExtractor(
        year=year,
        patch_size=patch_size,
        cache_dir=output_dir,
        embeddings_dir=embeddings_dir,
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        registry_dir=registry_dir,
        allow_mixed_cache=allow_mixed_cache,
    )
    results = extractor.extract_for_dataframe(
        df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, refresh=refresh,
        prune_tiles=prune_tiles,
    )
    logging.info("Representations available for %d samples in %s", len(results), output_dir)
    return len(results)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-sample Tessera embedding patches via geotessera."
    )
    parser.add_argument("--coords-csv", nargs="+", required=True,
                        help="One or more CSVs with sample_id, latitude, longitude columns.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write {sample_id}.npy patches (also the cache).")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR,
                        help=f"Tessera embedding year (default {DEFAULT_YEAR}).")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                        help=f"Patch size in 10 m pixels (default {DEFAULT_PATCH_SIZE}).")
    parser.add_argument("--embeddings-dir", default=None,
                        help="Where geotessera persists downloaded tiles "
                             "(default <output-dir>/tessera_tiles).")
    parser.add_argument("--dataset-version", default="v1", help="geotessera dataset version.")
    parser.add_argument("--dataset-variant", default="vultr", help="geotessera dataset variant.")
    parser.add_argument("--registry-dir", default=None,
                        help="Directory holding a pre-filtered manifest.parquet/landmasks.parquet "
                             "(see build_filtered_registry). geotessera otherwise loads the full "
                             "4.7M-row global manifest (~3.7 GB) into every process; filtering it "
                             "to the tiles a job needs drops that to ~0.15 GB and is what makes "
                             "parallel workers feasible on a memory-limited machine.")
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--lat-col", default="latitude")
    parser.add_argument("--lon-col", default="longitude")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if a cached patch already exists.")
    parser.add_argument("--prune-tiles", action="store_true",
                        help="Delete each ~110 MB Tessera tile once its samples have been "
                             "extracted. Keeps peak disk at about one tile instead of the "
                             "full dataset footprint (hundreds of GB at continent scale). "
                             "The per-sample patch cache is kept either way.")
    parser.add_argument("--allow-mixed-cache", action="store_true",
                        help="Proceed even if the cache directory already holds patches of a "
                             "different size. Off by default because the cache is keyed by "
                             "sample_id alone, so a size mismatch silently refetches and "
                             "overwrites everything.")
    return parser.parse_args()


def main():
    args = _parse_args()
    generate_representations(
        csv_paths=args.coords_csv,
        output_dir=args.output_dir,
        year=args.year,
        patch_size=args.patch_size,
        embeddings_dir=args.embeddings_dir,
        dataset_version=args.dataset_version,
        dataset_variant=args.dataset_variant,
        registry_dir=args.registry_dir,
        id_col=args.id_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        refresh=args.refresh,
        prune_tiles=args.prune_tiles,
        allow_mixed_cache=args.allow_mixed_cache,
    )


if __name__ == "__main__":
    main()
