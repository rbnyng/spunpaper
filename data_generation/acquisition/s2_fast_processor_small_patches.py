#!/usr/bin/env python3
"""
s2_fast_processor_small_patches.py — Sentinel-2 L2A processor optimised for small patches
Updated: 2025-05-26 (fixes the TIFF block-size issue)
Tuned for 256x256 patches; works around the TIFF write block-size limitation
"""

from __future__ import annotations
import os, sys, argparse, logging, datetime, time, warnings, signal
from pathlib import Path
import multiprocessing
from contextlib import contextmanager
import concurrent.futures
import uuid
import tempfile
import shutil
import gc
import random

import numpy as np
import psutil, rasterio, xarray as xr, rioxarray
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
import pystac_client, planetary_computer, stackstac

import dask
from dask.distributed import Client, LocalCluster, performance_report, wait

# ▶ distributed version compatibility
try:
    from distributed.comm.core import CommClosedError
except ImportError:
    from distributed import CommClosedError

warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask.core")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

BAND_MAPPING = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22",
    "SCL": "scl",
}
S2_BANDS        = list(BAND_MAPPING.keys())
BASELINE_CUTOFF = datetime.datetime(2022, 1, 25)
BASELINE_OFFSET = 1000

# SCL values treated as invalid (cloud-free / shadow-free / non-water pixels are valid)
SCL_INVALID = {0, 1, 2, 3, 8, 9, np.nan}

# SCL value descriptions, used in logs
SCL_DESCRIPTIONS = {
    0: "no data", 1: "saturated or defective", 2: "dark area", 3: "unclassified", 4: "vegetation",
    5: "bare soil", 6: "water", 7: "unused", 8: "cloud", 9: "thin cloud", 10: "snow", 11: "cloud shadow"
}

# Valid-coverage threshold (skip processing below this value)
MIN_VALID_COVERAGE = 5.0  # percent

# Temp file directory (defaults to the system temp dir)
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# Timeouts tuned for small patches (seconds) - much shorter than the defaults
PROCESS_TIMEOUT = 60 * 60   # overall: 60 min
DAY_TIMEOUT = 20 * 60       # per day: 20 min  
ITEM_TIMEOUT = 10 * 60      # per item: 10 min
BAND_TIMEOUT = 5 * 60       # per band: 5 min
SCL_BAND_TIMEOUT = 5 * 60   # SCL band: 5 min

# Network retry settings
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 5

# Small-patch concurrency
DEFAULT_MAX_WORKERS = 2

def ensure_numpy_array(data, name="data"):
    """Return data as a numpy array, handling xarray DataArray inputs"""
    if hasattr(data, 'values'):
        if hasattr(data.values, 'compute'):
            return data.values.compute()
        else:
            return data.values
    elif hasattr(data, 'compute'):
        return data.compute()
    else:
        return np.asarray(data)

def calculate_optimal_blocksize(width, height):
    """
    Pick a TIFF block size that is a multiple of 16 and suits small patches
    """
    if width <= 512 and height <= 512:
        return None, None  # untiled
    
    def round_to_multiple_of_16(value, max_val):
        """Round up to a multiple of 16, capped at max_val"""
        if value <= 16:
            return 16
        rounded = ((value + 15) // 16) * 16
        return min(rounded, max_val)
    
    target_block_size = 256
    
    blockx = round_to_multiple_of_16(min(target_block_size, width), width)
    blocky = round_to_multiple_of_16(min(target_block_size, height), height)
    
    return blockx, blocky

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    """Timeout context manager"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out ({seconds}s)")
    
    # Unix signals can only be handled in the main thread
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def get_args():
    P = argparse.ArgumentParser("Small Patch Optimized Sentinel-2 L2A Processor")
    P.add_argument("--input_tiff",   required=True, help="ROI mask or template raster")
    P.add_argument("--start_date",   required=True, help="Start date (YYYY-MM-DD[THH:MM:SS]), inclusive")
    P.add_argument("--end_date",     required=True, help="End date (YYYY-MM-DD[THH:MM:SS]), inclusive")
    P.add_argument("--output",       default="sentinel2_output", help="Output directory")
    P.add_argument("--max_cloud",    type=float, default=90, help="Maximum cloud cover (percent)")
    P.add_argument("--dask_workers", type=int,   default=1, help="Number of Dask workers for this partition")
    P.add_argument("--worker_memory",type=int,   default=4, help="Memory per worker (GB)")
    P.add_argument("--chunksize",    type=int,   default=256, help="stackstac x/y chunk size")
    P.add_argument("--resolution",   type=int,   default=10, help="Output resolution (metres)")
    P.add_argument("--overwrite",    action="store_true", help="Overwrite existing files")
    P.add_argument("--debug",        action="store_true", help="Enable debug logging")
    P.add_argument("--min_coverage", type=float, default=MIN_VALID_COVERAGE,
                   help="Minimum valid-pixel coverage (percent)")
    P.add_argument("--partition_id", default="unknown",
                   help="Partition ID (used to tag log output)")
    P.add_argument("--temp_dir",     default=TEMP_DIR,
                   help="Directory for temp files; defaults to the system temp dir")
    return P.parse_args()

def setup_logging(debug: bool, out_dir: Path, partition_id: str):
    """Set up logging, tagging records with the partition ID"""
    fmt = f"%(asctime)s [{partition_id}] [%(levelname)s] %(message)s"
    lvl = logging.DEBUG if debug else logging.INFO
    
    logger = logging.getLogger()
    logger.setLevel(lvl)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(fmt)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(
        out_dir / f"s2_{partition_id}_detail.log", 
        "a", 
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_sys(partition_id: str):
    m = psutil.virtual_memory()
    logging.info(f"[{partition_id}] System - CPU {os.cpu_count()} | "
                 f"RAM {m.total/1e9:.1f} GB (free {m.available/1e9:.1f} GB)")

def fmt_bbox(b):
    return f"{b[0]:.5f},{b[1]:.5f} ⇢ {b[2]:.5f},{b[3]:.5f}"

def make_simple_client(req_workers:int, req_mem:int, partition_id: str):
    """Create a lightweight Dask client tuned for small patches"""
    workers = min(req_workers, 2)
    memory_gb = min(req_mem, 8)
    
    logging.info(f"[{partition_id}] Creating lightweight Dask client: {workers} workers, {memory_gb}GB memory")
    
    # Per-partition dashboard port, to avoid collisions
    port_base = 8900
    port_range = 200
    dashboard_port = port_base + (abs(hash(partition_id)) % port_range)
    
    dask_config = {
        "distributed.worker.memory.target": 0.70,
        "distributed.worker.memory.spill": 0.80,   
        "distributed.worker.memory.pause": 0.90,
        "array.slicing.split_large_chunks": False,  # small arrays need no splitting
        "optimization.fuse.active": False,          # keep the graph simple
        "distributed.worker.daemon": False,
        "distributed.scheduler.work-stealing": False,  # reduce complexity
    }
    
    dask.config.set(dask_config)
    
    try:
        cluster = LocalCluster(
            n_workers         = workers,
            threads_per_worker= 2,
            processes         = True,
            memory_limit      = f"{memory_gb}GB",
            dashboard_address = f":{dashboard_port}",
            silence_logs      = "ERROR",
        )
        
        cli = Client(cluster, asynchronous=False, timeout=30)
        logging.info(f"[{partition_id}] Dask client created, dashboard: {cli.dashboard_link}")
        return cli
    except Exception as e:
        logging.error(f"[{partition_id}] Failed to create the Dask client: {e}")
        # Return None; processing then falls back to synchronous mode
        return None

def load_roi(tiff: Path, partition_id: str):
    """Load the ROI, optimised for small patches"""
    with rasterio.open(tiff) as src:
        tpl = dict(crs=src.crs,
                   transform=src.transform,
                   width=src.width,
                   height=src.height)
        bbox_proj = src.bounds
        bbox_ll   = transform_bounds(src.crs, "EPSG:4326", *bbox_proj,
                                     densify_pts=21)
        
        # Read the mask and binarise it to save memory
        mask_np = (src.read(1) > 0).astype(np.uint8)
        
    roi_size_kb = (mask_np.size * mask_np.itemsize) / 1024
    is_small_patch = tpl['width'] <= 512 and tpl['height'] <= 512
    
    logging.info(f"[{partition_id}] ROI (CRS={tpl['crs']}): {tpl['width']}×{tpl['height']} ({roi_size_kb:.1f} KB)")
    logging.info(f"[{partition_id}] Small-patch mode: {'yes' if is_small_patch else 'no'}")
    logging.info(f"[{partition_id}] ROI bbox proj: {fmt_bbox(bbox_proj)}")
    logging.info(f"[{partition_id}] ROI bbox lon/lat: {fmt_bbox(bbox_ll)}")
    
    return tpl, bbox_proj, bbox_ll, mask_np, is_small_patch

def search_items(bbox_ll, date_range:str, max_cloud, partition_id: str):
    """Search STAC items, with retry logic tuned for small patches"""
    start_date, end_date = date_range.split("/")
    
    # Add one second to the end time so the end point is included
    try:
        end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00').replace(' ', 'T'))
        end_dt_plus = end_dt + datetime.timedelta(seconds=1)
        search_date_range = f"{start_date}/{end_dt_plus.isoformat()}"
    except ValueError:
        logging.warning(f"[{partition_id}] Could not parse the end date, using the original range: {date_range}")
        search_date_range = date_range
    
    logging.info(f"[{partition_id}] STAC search date range: {search_date_range}")
    
    retries = 0
    max_retries = MAX_RETRIES
    retry_delay = 1
    
    while retries <= max_retries:
        try:
            cat = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace)
            q = cat.search(collections=["sentinel-2-l2a"],
                       bbox=bbox_ll, datetime=search_date_range,
                       query={"eo:cloud_cover": {"lt": max_cloud}})
            items = list(q.get_items())
            logging.info(f"[{partition_id}] STAC returned {len(items)} items (cloud < {max_cloud}%)")
            if items:
                b = np.array([it.bbox for it in items])
                union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
                logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
            return items
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logging.error(f"[{partition_id}] STAC search failed (attempt {retries}/{max_retries+1}): {e}")
                raise
            
            retry_delay = min(30, retry_delay * RETRY_BACKOFF_FACTOR)
            jitter = random.uniform(0.8, 1.2)
            actual_delay = retry_delay * jitter
            
            logging.warning(f"[{partition_id}] STAC search failed (attempt {retries}/{max_retries+1}): {e}, retrying in {actual_delay:.1f}s...")
            time.sleep(actual_delay)

def group_by_date(items, partition_id: str):
    """Group items by date"""
    g = {}
    for it in items:
        d = it.properties["datetime"][:10]
        g.setdefault(d, []).append(it)
    logging.info(f"[{partition_id}] ⇒ {len(g)} observation days")
    return dict(sorted(g.items()))

def harmonize_arr(arr: np.ndarray, date_key:str):
    """Apply the processing-baseline offset correction"""
    if datetime.datetime.strptime(date_key, "%Y-%m-%d") > BASELINE_CUTOFF:
        # Skip NaN to avoid warnings
        valid_mask = ~np.isnan(arr) & (arr >= BASELINE_OFFSET)
        np.subtract(arr, BASELINE_OFFSET, out=arr, where=valid_mask)
    return arr

def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    """Write a GeoTIFF, tuned for small patches, with the block-size issue fixed"""
    if np.isnan(np_arr).any():
        np_arr = np.nan_to_num(np_arr, nan=0)
    
    blockx, blocky = calculate_optimal_blocksize(tpl["width"], tpl["height"])
    
    profile = dict(
        driver="GTiff", 
        dtype=dtype, 
        count=1,
        width=tpl["width"], 
        height=tpl["height"],
        crs=tpl["crs"], 
        transform=tpl["transform"],
        compress="lzw",
        nodata=0
    )
    
    if blockx is not None and blocky is not None:
        profile.update({
            "tiled": True,
            "blockxsize": blockx,
            "blockysize": blocky
        })
        logging.debug(f"Tiled mode: {blockx}x{blocky}")
    else:
        profile["tiled"] = False
        logging.debug(f"Untiled mode (small patch {tpl['width']}x{tpl['height']})")
    
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np_arr.astype(dtype, copy=False), 1)
        
        if metadata:
            dst.update_tags(**metadata)

def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    try:
        with rasterio.open(file_path) as src:
            if src.shape != expected_shape:
                logging.warning(f"Validation failed: {file_path} shape mismatch. Expected {expected_shape}, got {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logging.warning(f"Validation failed: {file_path} CRS mismatch. Expected {expected_crs}, got {src.crs}")
                return False
            
            # Check for data via the band statistics, avoiding a full array read
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"Validation failed: {file_path} band is all zeros")
                return False
            
            logging.debug(f"TIFF validation passed: {file_path}, shape={src.shape}")
            return True
            
    except Exception as e:
        logging.error(f"Error validating TIFF {file_path}: {e}")
        return False

def is_valid_scl(scl_arr):
    """True where the SCL value marks a valid observation (not cloud/shadow/water)"""
    return ~np.isin(np.nan_to_num(scl_arr, nan=0), list(SCL_INVALID - {np.nan}))

def process_scl_simple(scl_arr, roi_mask, partition_id="unknown"):
    """SCL handling tuned for small patches."""
    scl_arr = ensure_numpy_array(scl_arr, "scl_arr")
    
    # Handle both single-tile and multi-tile arrays
    if len(scl_arr.shape) == 3:
        n_tiles, scl_height, scl_width = scl_arr.shape
    else:
        scl_height, scl_width = scl_arr.shape
        n_tiles = 1
        scl_arr = scl_arr.reshape(1, scl_height, scl_width)
    
    roi_mask = roi_mask.astype(bool)
    roi_height, roi_width = roi_mask.shape
    
    if scl_height != roi_height or scl_width != roi_width:
        logging.debug(f"[{partition_id}] SCL shape adjusted: {(scl_height, scl_width)} -> {(roi_height, roi_width)}")
        use_height = min(scl_height, roi_height)
        use_width = min(scl_width, roi_width)
        scl_arr = scl_arr[:, :use_height, :use_width]
        roi_mask = roi_mask[:use_height, :use_width]
    
    valid_mask = is_valid_scl(scl_arr)
    
    roi_pixel_count = np.sum(roi_mask)
    
    # Simplified tile selection: keep the first valid tile
    tile_selection = np.full(roi_mask.shape, -1, dtype=np.int8)
    
    for tile_idx in range(n_tiles):
        current_valid = valid_mask[tile_idx] & roi_mask & (tile_selection < 0)
        tile_selection[current_valid] = tile_idx
    
    valid_pixel_count = np.sum(tile_selection >= 0)
    valid_pct = 100.0 * valid_pixel_count / roi_pixel_count if roi_pixel_count > 0 else 0.0
    
    logging.info(f"[{partition_id}] SCL result: valid pixels {valid_pixel_count}/{roi_pixel_count}, coverage {valid_pct:.2f}%")
    
    return valid_mask, tile_selection, valid_pct

def create_scl_mosaic_simple(scl_arr, tile_selection, roi_mask, target_shape, date_key=None, partition_id="unknown"):
    """
    Build the SCL mosaic (simplified), with the DataArray issue fixed
    """
    try:
        scl_arr = ensure_numpy_array(scl_arr, "scl_arr")
        
        if len(scl_arr.shape) == 3:
            n_tiles, arr_height, arr_width = scl_arr.shape
        else:
            arr_height, arr_width = scl_arr.shape
            n_tiles = 1
            scl_arr = scl_arr.reshape(1, arr_height, arr_width)
        
        target_height, target_width = target_shape
        result = np.zeros(target_shape, dtype=np.uint8)
        
        common_height = min(arr_height, roi_mask.shape[0], target_height, tile_selection.shape[0])
        common_width = min(arr_width, roi_mask.shape[1], target_width, tile_selection.shape[1])
        
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            y_coords, x_coords = np.where(valid_roi)
            tile_indices = tile_sel_crop[y_coords, x_coords]
            result[y_coords, x_coords] = scl_arr[tile_indices, y_coords, x_coords]
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] SCL mosaic failed: {e}")
        raise

def smart_mosaic_simple(data_arr, tile_selection, roi_mask, partition_id="unknown"):
    """Mosaic the selected tiles, tuned for small patches."""
    try:
        data_arr = ensure_numpy_array(data_arr, "data_arr")
        
        # Single tile
        if len(data_arr.shape) < 3 or data_arr.shape[0] == 1:
            result = data_arr[0] if len(data_arr.shape) == 3 else data_arr
            
            if result.shape != roi_mask.shape:
                common_height = min(result.shape[0], roi_mask.shape[0])
                common_width = min(result.shape[1], roi_mask.shape[1])
                
                final_result = np.zeros(roi_mask.shape, dtype=result.dtype)
                result_cropped = result[:common_height, :common_width] 
                roi_mask_cropped = roi_mask[:common_height, :common_width]
                final_result[:common_height, :common_width] = result_cropped * roi_mask_cropped
                return final_result
            
            return result * roi_mask
        
        # Multiple tiles
        n_tiles, data_height, data_width = data_arr.shape
        roi_mask = roi_mask.astype(bool)
        result = np.zeros(roi_mask.shape, dtype=data_arr.dtype)
        
        common_height = min(data_height, roi_mask.shape[0], tile_selection.shape[0])
        common_width = min(data_width, roi_mask.shape[1], tile_selection.shape[1])
        
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            y_coords, x_coords = np.where(valid_roi)
            tile_indices = tile_sel_crop[y_coords, x_coords]
            result[y_coords, x_coords] = data_arr[tile_indices, y_coords, x_coords]
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] Mosaicking failed: {e}")
        raise

def process_band_simple(items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                res, chunksize, out_path, is_small_patch, dask_client, partition_id="unknown", retries=2):
    """Simplified band processing, tuned for small patches, with the DataArray issue fixed"""
    t0 = time.time()
    
    if band_name == "SCL":
        logging.info(f"[{partition_id}]     band {band_name} already handled during quality assessment, skipping")
        return True
    
    logging.info(f"[{partition_id}]     Processing band {band_name}")
    
    if out_path.exists():
        if validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]     {band_name} valid file already exists, skipping")
            return True
        else:
            logging.warning(f"[{partition_id}]     {band_name} file exists but is invalid, reprocessing")
            out_path.unlink()
    
    for attempt in range(retries + 1):
        try:
            with timeout_handler(BAND_TIMEOUT):
                small_chunksize = min(chunksize, 256) if is_small_patch else chunksize
                
                da = stackstac.stack(
                    items=items,
                    assets=[band_name],
                    resolution=res,
                    epsg=tpl["crs"].to_epsg(),
                    bounds=bbox_proj,
                    chunksize=small_chunksize,
                    rescale=False,
                    resampling=Resampling.nearest
                )
                
                item_dim = None
                for dim in da.dims:
                    if dim not in ('band', 'x', 'y'):
                        if da.sizes[dim] > 1:
                            item_dim = dim
                        elif da.sizes[dim] == 1:
                            da = da.squeeze(dim, drop=True)
                
                band_da = da.sel(band=band_name)
                
                # Small patches compute synchronously; larger ones go through dask
                if is_small_patch or dask_client is None:
                    try:
                        band_arr = band_da.compute()
                    except Exception as e:
                        logging.warning(f"[{partition_id}]     synchronous compute failed: {e}, falling back to dask")
                        if dask_client:
                            band_arr = ensure_numpy_array(band_da, "band_da")
                        else:
                            raise
                else:
                    band_arr = ensure_numpy_array(band_da, "band_da")
                
                band_arr = ensure_numpy_array(band_arr, "band_arr")
                
                logging.debug(f"[{partition_id}]     {band_name} array shape: {band_arr.shape}, ROI shape: {mask_np.shape}")
                
                # Multi-tile case
                if item_dim:
                    if tile_selection is not None:
                        arr = smart_mosaic_simple(band_arr, tile_selection, mask_np, partition_id)
                    else:
                        # Fall back to the first tile
                        if len(band_arr.shape) == 3:
                            arr = band_arr[0]
                        else:
                            arr = band_arr
                        
                        if arr.shape != mask_np.shape:
                            common_height = min(arr.shape[0], mask_np.shape[0])
                            common_width = min(arr.shape[1], mask_np.shape[1])
                            
                            final_arr = np.zeros((tpl["height"], tpl["width"]), dtype=arr.dtype)
                            arr_crop = arr[:common_height, :common_width]
                            mask_crop = mask_np[:common_height, :common_width]
                            final_arr[:common_height, :common_width] = arr_crop * mask_crop
                            arr = final_arr
                        else:
                            arr = arr * mask_np
                else:
                    # Single tile
                    arr = band_arr
                    if arr.shape != mask_np.shape:
                        final_arr = np.zeros((tpl["height"], tpl["width"]), dtype=arr.dtype)
                        common_height = min(arr.shape[0], tpl["height"])
                        common_width = min(arr.shape[1], tpl["width"])
                        
                        arr_crop = arr[:common_height, :common_width]
                        mask_crop = mask_np[:common_height, :common_width]
                        final_arr[:common_height, :common_width] = arr_crop * mask_crop
                        arr = final_arr
                    else:
                        arr = arr * mask_np
                
                harmonize_arr(arr, date_key)
                
                metadata = {
                    "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    "DATE_ACQUIRED": date_key,
                    "BAND_NAME": band_name,
                    "ITEMS_COUNT": len(items)
                }
                
                dtype = "uint16"
                write_tiff(arr, out_path, tpl, dtype, metadata)
                
                if not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                    logging.error(f"[{partition_id}]     ✗ band {band_name} failed validation")
                    if out_path.exists():
                        out_path.unlink()
                    continue
                
                logging.info(f"[{partition_id}]     ✓ {band_name:9s}  "
                            f"{os.path.getsize(out_path)/1e6:.2f} MB, in {time.time()-t0:.1f}s")
                
                return True
                
        except TimeoutException as e:
            if attempt < retries:
                retry_delay = min(15, (attempt + 1) * 2)
                logging.warning(f"[{partition_id}]     band {band_name} timed out, retrying in {retry_delay}s ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ band {band_name} timed out: {e}")
                return False
                
        except Exception as e:
            if attempt < retries:
                retry_delay = min(15, (attempt + 1) * 2)
                logging.warning(f"[{partition_id}]     band {band_name} error: {e}, retrying in {retry_delay}s ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ band {band_name} failed: {e}")
                return False
    
    return False

def process_scl_assessment_simple(items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                                       min_coverage, out_root, overwrite, is_small_patch, dask_client, partition_id="unknown"):
    """
    Simplified SCL processing, tuned for small patches, with the DataArray issue fixed
    """
    t0 = time.time()
    logging.info(f"[{partition_id}]   Processing the SCL band for quality assessment and writing the SCL output")
    
    scl_out_name = BAND_MAPPING["SCL"]
    scl_dir = out_root / scl_out_name
    scl_dir.mkdir(parents=True, exist_ok=True)
    scl_out_path = scl_dir / f"{date_key}_mosaic.tiff"
    
    if not overwrite and scl_out_path.exists():
        if validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]   SCL file already exists and is valid, skipping")
            # Return a trivial tile_selection
            tile_selection = np.zeros(mask_np.shape, dtype=np.int8)
            return True, 100.0, tile_selection
    
    if not all('SCL' in item.assets for item in items):
        scl_items = [item for item in items if 'SCL' in item.assets]
        if not scl_items:
            logging.warning(f"[{partition_id}]   no item has an SCL asset!")
            return False, 0.0, None
        items = scl_items
    
    try:
        with timeout_handler(SCL_BAND_TIMEOUT):
            small_chunksize = min(chunksize, 256) if is_small_patch else chunksize
            
            da = stackstac.stack(
                items=items,
                assets=['SCL'],
                resolution=res,
                epsg=tpl["crs"].to_epsg(),
                bounds=bbox_proj,
                chunksize=small_chunksize,
                rescale=False,
                resampling=Resampling.nearest
            )
            
            item_dim = None
            for dim in da.dims:
                if dim not in ('band', 'x', 'y'):
                    if da.sizes[dim] > 1:
                        item_dim = dim
                    elif da.sizes[dim] == 1:
                        da = da.squeeze(dim, drop=True)
            
            scl_da = da.sel(band='SCL')
            
            if is_small_patch or dask_client is None:
                try:
                    scl_arr = scl_da.compute()
                except Exception as e:
                    logging.warning(f"[{partition_id}]   SCL synchronous compute failed: {e}, falling back to dask")
                    scl_arr = ensure_numpy_array(scl_da, "scl_da")
            else:
                scl_arr = ensure_numpy_array(scl_da, "scl_da")
            
            scl_arr = ensure_numpy_array(scl_arr, "scl_arr")
            
            valid_mask, tile_selection, valid_pct = process_scl_simple(scl_arr, mask_np, partition_id)
            
            if valid_pct < min_coverage:
                logging.warning(f"[{partition_id}]   ⚠️ {date_key} valid coverage {valid_pct:.2f}% < {min_coverage}%, skipping SCL output")
                return False, valid_pct, None
            
            scl_output = create_scl_mosaic_simple(scl_arr, tile_selection, mask_np, (tpl["height"], tpl["width"]), date_key, partition_id)
            
            metadata = {
                "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                "DATE_ACQUIRED": date_key,
                "BAND_NAME": "SCL",
                "ITEMS_COUNT": len(items),
                "VALID_COVERAGE_PCT": f"{valid_pct:.2f}"
            }
            
            write_tiff(scl_output, scl_out_path, tpl, "uint8", metadata)
            
            if not validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                logging.error(f"[{partition_id}]   ✗ SCL file failed validation")
                if scl_out_path.exists():
                    scl_out_path.unlink()
                return False, 0.0, None
            
            logging.info(f"[{partition_id}]   ✓ SCL done, valid {valid_pct:.2f}%, "
                        f"size: {os.path.getsize(scl_out_path)/1e6:.2f} MB, in {time.time()-t0:.1f}s")
            
            return True, valid_pct, tile_selection
            
    except TimeoutException as e:
        logging.error(f"[{partition_id}]   ✗ SCL processing timed out: {e}")
        return False, 0.0, None
            
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ SCL processing failed: {e}")
        return False, 0.0, None

def process_day_simple(date_key:str, items, tpl, bbox_proj, mask_np,
                out_root:Path, res:int, chunksize:int,
                overwrite:bool, min_coverage:float, is_small_patch:bool, dask_client,
                partition_id:str="unknown") -> bool:
    """Simplified single-day processing, tuned for small patches"""
    logging.info(f"[{partition_id}] → {date_key} (item={len(items)})")
    t0 = time.time()
    
    try:
        with timeout_handler(DAY_TIMEOUT):
            for outname in BAND_MAPPING.values():
                band_dir = out_root / outname
                band_dir.mkdir(parents=True, exist_ok=True)
            
            # Skip the day when every band is already present and valid
            if not overwrite:
                all_exist = True
                for band_name in S2_BANDS:
                    out_name = BAND_MAPPING[band_name]
                    out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                    if not out_path.exists() or not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        all_exist = False
                        break
                
                if all_exist:
                    logging.info(f"[{partition_id}]   all bands already exist and are valid, skipping")
                    return True
            
            scl_success, valid_pct, tile_selection = process_scl_assessment_simple(
                items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                min_coverage, out_root, overwrite, is_small_patch, dask_client, partition_id
            )
            
            if not scl_success:
                if valid_pct < min_coverage:
                    logging.warning(f"[{partition_id}]   {date_key} valid coverage {valid_pct:.2f}% < {min_coverage}%, skipping the remaining bands")
                    return True
                else:
                    logging.error(f"[{partition_id}]   {date_key} SCL processing failed, skipping this date")
                    return False
            
            day_temp_dir = tempfile.mkdtemp(prefix=f"s2_{date_key}_", dir=TEMP_DIR)
            
            try:
                other_bands = [band for band in S2_BANDS if band != "SCL"]
                
                max_workers = min(DEFAULT_MAX_WORKERS, len(other_bands)) if is_small_patch else min(4, len(other_bands))
                logging.info(f"[{partition_id}]   using {max_workers} threads for {len(other_bands)} bands")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    
                    for band_name in other_bands:
                        out_name = BAND_MAPPING[band_name]
                        out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                        
                        if not overwrite and out_path.exists() and validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                            logging.info(f"[{partition_id}]     band {band_name} valid file already exists, skipping")
                            continue
                        
                        temp_path = Path(day_temp_dir) / f"{band_name}_{date_key}.tiff"
                        
                        future = executor.submit(
                            process_band_simple, 
                            items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                            res, chunksize, temp_path, is_small_patch, dask_client, partition_id
                        )
                        futures[future] = (band_name, out_path, temp_path)
                    
                    success_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        band_name, out_path, temp_path = futures[future]
                        try:
                            success = future.result()
                            if success:
                                if temp_path.exists() and validate_tiff(temp_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                                    shutil.copy2(temp_path, out_path)
                                    success_count += 1
                                    logging.info(f"[{partition_id}]     ✓ {band_name} done")
                                else:
                                    logging.error(f"[{partition_id}]     ✗ {band_name} temp file is invalid")
                            else:
                                logging.warning(f"[{partition_id}]     ✗ {band_name} failed")
                        except Exception as e:
                            logging.error(f"[{partition_id}]     ✗ {band_name} raised an exception: {e}")
            finally:
                try:
                    shutil.rmtree(day_temp_dir)
                except:
                    pass
            
            total_other_bands = len(other_bands)
            total_bands = len(S2_BANDS)
            proc_time = time.time() - t0
            
            total_success = (1 if scl_success else 0) + success_count
            
            if total_success == total_bands:
                logging.info(f"[{partition_id}] ← {date_key} all bands succeeded ({total_success}/{total_bands}) in {proc_time:.1f}s")
                return True
            elif total_success > 0:
                logging.warning(f"[{partition_id}] ← {date_key} some bands succeeded ({total_success}/{total_bands}) in {proc_time:.1f}s")
                return True
            else:
                logging.error(f"[{partition_id}] ← {date_key} all bands failed after {proc_time:.1f}s")
                return False
                
    except TimeoutException as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} timed out ({proc_time:.1f}s): {e}")
        return False
    except Exception as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} failed: {type(e).__name__} - {e}")
        return False

def main():
    a = get_args()
    out_dir = Path(a.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    global TEMP_DIR
    TEMP_DIR = a.temp_dir
    
    setup_logging(a.debug, out_dir, a.partition_id)
    logging.info(f"[{a.partition_id}] ⚡ S2 Small Patch Processor started (TIFF block-size fix)")
    log_sys(a.partition_id)
    logging.info(f"[{a.partition_id}] Timeouts: overall {PROCESS_TIMEOUT//60} min, per day {DAY_TIMEOUT//60} min")
    logging.info(f"[{a.partition_id}] Temp dir: {TEMP_DIR}")
    logging.info(f"[{a.partition_id}] Date range: {a.start_date} → {a.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np, is_small_patch = load_roi(Path(a.input_tiff), a.partition_id)
    
    search_date_range = f"{a.start_date}/{a.end_date}"
    
    items = search_items(bbox_ll, search_date_range, a.max_cloud, a.partition_id)
    if not items:
        logging.warning(f"[{a.partition_id}] No matching scenes, exiting")
        return

    groups = group_by_date(items, a.partition_id)

    base_temp_dir = tempfile.mkdtemp(prefix=f"s2_proc_{a.partition_id}_", dir=TEMP_DIR)
    
    try:
        # Dask client; on failure we fall back to synchronous processing
        dask_client = make_simple_client(a.dask_workers, a.worker_memory, a.partition_id)
        
        results = []
        for i, (d, its) in enumerate(groups.items()):
            # Small patches do not need the heavier GC/restart logic
            if i > 0 and not is_small_patch:
                gc.collect()
            
            try:
                success = process_day_simple(
                    d, its, tpl, bbox_proj, mask_np,
                    out_dir, a.resolution, a.chunksize,
                    a.overwrite, a.min_coverage, is_small_patch, dask_client, a.partition_id
                )
                results.append(success)
            except Exception as day_error:
                logging.error(f"[{a.partition_id}] exception while processing date {d}: {day_error}")
                results.append(False)
        
        if dask_client:
            try:
                dask_client.close(timeout=30)
            except:
                pass
        
        success_count = sum(results)
        total_count = len(results)
        
        logging.info(f"[{a.partition_id}] ✅ Partition complete: succeeded {success_count}/{total_count} days")
        
        if success_count == 0 and total_count > 0:
            sys.exit(1)
        elif success_count < total_count:
            logging.warning(f"[{a.partition_id}] ⚠️  some dates failed ({total_count - success_count}/{total_count})")
            sys.exit(2)
        else:
            sys.exit(0)
    
    finally:
        try:
            shutil.rmtree(base_temp_dir)
        except:
            pass

if __name__ == "__main__":
    main()