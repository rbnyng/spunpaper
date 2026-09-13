#!/usr/bin/env python3
"""
s1_fast_processor.py — Sentinel-1 RTC fast download & ROI mosaicking (flexible parallel edition)
Updated: 2025-05-20
Supports flexible parallel partition processing, with robust error handling and timeout control
"""

from __future__ import annotations
import os, sys, argparse, logging, datetime, time, warnings, signal
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import psutil, rasterio, xarray as xr, rioxarray
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
from rasterio.merge import merge
import pystac_client, planetary_computer, stackstac
import shapely.geometry
import concurrent.futures
import uuid
import tempfile
import shutil

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

# Sentinel-1 resolution (metres)
SAR_RESOLUTION = 10.0

# Valid-coverage threshold (skip processing below this value)
MIN_VALID_COVERAGE = 10.0  # percent

# Timeouts (seconds)
PROCESS_TIMEOUT = 120 * 60  # overall
DAY_TIMEOUT = 40 * 60      # per day
ITEM_TIMEOUT = 20 * 60      # per item

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
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
    P = argparse.ArgumentParser("Fast Sentinel-1 RTC Processor (Flexible Parallel Edition)")
    P.add_argument("--input_tiff",   required=True)
    P.add_argument("--start_date",   required=True)
    P.add_argument("--end_date",     required=True)
    P.add_argument("--output",       default="sentinel1_output")
    P.add_argument("--orbit_state",  default="both", choices=["ascending", "descending", "both"])
    P.add_argument("--dask_workers", type=int,   default=8)
    P.add_argument("--worker_memory",type=int,   default=16)
    P.add_argument("--chunksize",    type=int,   default=1024)
    P.add_argument("--workers",      type=int,   default=8)
    P.add_argument("--overwrite",    action="store_true")
    P.add_argument("--debug",        action="store_true")
    P.add_argument("--min_coverage", type=float, default=MIN_VALID_COVERAGE,
                   help="Minimum valid-pixel coverage (percent)")
    P.add_argument("--partition_id", default="unknown",
                   help="Partition ID (used to tag log output)")
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
        out_dir / f"s1_{partition_id}_detail.log", 
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

def make_client(req_workers:int, req_mem:int, partition_id: str):
    """Create the Dask client, using a per-partition dashboard port"""
    total_mem = psutil.virtual_memory().total / 1e9
    workers = min(req_workers, os.cpu_count(),
                  max(1, int(total_mem // (req_mem*1.2))))
    if workers < req_workers:
        logging.warning(f"⚠️  workers {req_workers}→{workers} (resource limit)")
    
    # Derive the dashboard port from the partition ID hash, so the same partition
    # always lands on the same port in 8700-8779.
    port_base = 8700
    port_range = 80
    dashboard_port = port_base + (hash(partition_id) % port_range)
    
    cluster = LocalCluster(
        n_workers         = workers,
        threads_per_worker= 4,
        processes         = True,
        memory_limit      = f"{req_mem}GB",
        dashboard_address = f":{dashboard_port}",
        silence_logs      = "ERROR",
    )
    dask.config.set({
        "distributed.worker.memory.target": 0.80,
        "distributed.worker.memory.spill":  0.90,
        "distributed.worker.memory.pause":  0.95,
    })
    cli = Client(cluster, asynchronous=False)
    logging.info(f"[{partition_id}] Dask dashboard → {cli.dashboard_link}")
    return cli

def load_roi(tiff: Path, partition_id: str):
    with rasterio.open(tiff) as src:
        tpl = dict(crs=src.crs,
                   transform=src.transform,
                   width=src.width,
                   height=src.height)
        bbox_proj = src.bounds
        bbox_ll   = transform_bounds(src.crs, "EPSG:4326", *bbox_proj,
                                     densify_pts=21)
        mask_np   = (src.read(1) > 0).astype(np.uint8)
    logging.info(f"[{partition_id}] ROI (CRS={tpl['crs']}): {tpl['width']}×{tpl['height']}")
    logging.info(f"[{partition_id}] ROI bbox proj: {fmt_bbox(bbox_proj)}")
    logging.info(f"[{partition_id}] ROI bbox lon/lat: {fmt_bbox(bbox_ll)}")
    return tpl, bbox_proj, bbox_ll, mask_np

def mask_to_xr(mask_np, tpl):
    da = xr.DataArray(mask_np, dims=("y", "x"))
    return da.rio.write_crs(tpl["crs"]).rio.write_transform(tpl["transform"])

def search_items(bbox_ll, date_range:str, orbit_state="both", partition_id="unknown"):
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)
    
    query = {"collections": ["sentinel-1-rtc"], "bbox": bbox_ll, "datetime": date_range}
    
    if orbit_state != "both":
        query["query"] = {"sat:orbit_state": {"eq": orbit_state}}
    
    q = cat.search(**query)
    items = list(q.get_items())
    logging.info(f"[{partition_id}] STAC returned {len(items)} items")
    if items:
        b = np.array([it.bbox for it in items])
        union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
        logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
    return items

def group_by_date_orbit(items, partition_id: str):
    g = defaultdict(list)
    for it in items:
        d = it.properties["datetime"][:10]
        orbit = it.properties.get("sat:orbit_state", "unknown")
        key = f"{d}_{orbit}"
        g[key].append(it)
    logging.info(f"[{partition_id}] ⇒ {len(g)} date-orbit combinations")
    return dict(sorted(g.items()))

def amplitude_to_db(amp, mask=None):
    """
    Convert amplitude to dB, stored as int16: 20*log10(amp), offset by +50 to
    keep it positive, then scaled by 200 to preserve precision.
    """
    if hasattr(amp, 'values'):
        amp_array = amp.values
    elif hasattr(amp, 'compute'):
        amp_array = amp.compute()
    else:
        amp_array = np.asarray(amp)
    
    output = np.zeros_like(amp_array, dtype=np.int16)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        amp_finite = np.isfinite(amp_array)
        valid_mask = amp_finite & (amp_array > 0)
    
    if np.any(valid_mask):
        with np.errstate(invalid='ignore', divide='ignore'):
            # Compute directly on the valid positions to avoid boolean-indexing issues
            valid_indices = np.where(valid_mask)
            valid_amp = amp_array[valid_indices]
            
            db = 20.0 * np.log10(valid_amp)
            db_shift = db + 50.0
            scaled = db_shift * 200.0
            clipped = np.clip(scaled, 0, 32767)  # clip into the int16 range
        
        output[valid_indices] = clipped.astype(np.int16)
    
    if mask is not None:
        if hasattr(mask, 'values'):
            mask_array = mask.values
        else:
            mask_array = np.asarray(mask)
        
        if output.shape != mask_array.shape:
            common_shape = tuple(min(output.shape[i], mask_array.shape[i]) for i in range(len(output.shape)))
            
            if len(common_shape) == 2:
                output_cropped = output[:common_shape[0], :common_shape[1]]
                mask_cropped = mask_array[:common_shape[0], :common_shape[1]]
                output[:common_shape[0], :common_shape[1]] = np.where(mask_cropped > 0, output_cropped, 0)
                # Zero out anything beyond the mask extent
                if output.shape[0] > common_shape[0]:
                    output[common_shape[0]:, :] = 0
                if output.shape[1] > common_shape[1]:
                    output[:, common_shape[1]:] = 0
            else:
                output = np.where(mask_array > 0, output, 0)
        else:
            output = np.where(mask_array > 0, output, 0)
        
    return output

def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    if np.isnan(np_arr).any():
        np_arr = np.nan_to_num(np_arr, nan=0)
        
    profile = dict(driver="GTiff", dtype=dtype, count=1,
                   width=tpl["width"], height=tpl["height"],
                   crs=tpl["crs"], transform=tpl["transform"],
                   compress="lzw", tiled=True,
                   blockxsize=256, blockysize=256,
                   nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np_arr.astype(dtype, copy=False), 1)
        
        if metadata:
            dst.set_band_description(1, metadata.get("band_desc", ""))
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
            
            if not np.allclose(np.array(src.transform)[:6], np.array(expected_transform)[:6], rtol=1e-05, atol=1e-08):
                logging.warning(f"Validation failed: {file_path} transform mismatch.")
                return False
            
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"Validation failed: {file_path} band is all zeros")
                return False
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            expected_size_mb = (src.width * src.height * src.count * 2) / (1024 * 1024)  # int16 = 2 bytes
            if file_size_mb < expected_size_mb * 0.05:  # allow for compression, but not this small
                logging.warning(f"Validation failed: {file_path} file too small. Expected ~{expected_size_mb:.2f}MB, got {file_size_mb:.2f}MB")
                return False
            
            logging.debug(f"TIFF validation passed: {file_path}, shape={src.shape}, size={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logging.error(f"Error validating TIFF {file_path}: {e}")
        return False

def analyze_coverage(data_arr, roi_mask, partition_id: str):
    """Analyse how well the data covers the ROI, handling shape mismatches"""
    if hasattr(data_arr, 'values'):
        data_values = data_arr.values
    elif hasattr(data_arr, 'compute'):
        data_values = data_arr.compute()
    else:
        data_values = np.asarray(data_arr)
    
    # Valid pixels are non-zero and finite; suppress invalid-value warnings
    with np.errstate(invalid='ignore'):
        valid_mask = (data_values > 0) & np.isfinite(data_values)
    
    if len(valid_mask.shape) == 2 and valid_mask.shape != roi_mask.shape:
        logging.info(f"[{partition_id}]     Data shape {valid_mask.shape} does not match ROI shape {roi_mask.shape}, cropping to the common region")
        common_height = min(valid_mask.shape[0], roi_mask.shape[0])
        common_width = min(valid_mask.shape[1], roi_mask.shape[1])
        
        valid_mask_cropped = valid_mask[:common_height, :common_width]
        roi_mask_cropped = roi_mask[:common_height, :common_width]
        
        valid_count = int(np.sum(valid_mask_cropped & roi_mask_cropped))
        roi_count = int(np.sum(roi_mask_cropped))
        valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
        logging.info(f"[{partition_id}]     Single tile: valid pixels in ROI {valid_count}/{roi_count} ({valid_pct:.2f}%)")
        
        return valid_mask, valid_pct
    
    elif len(valid_mask.shape) == 3:
        # Multiple tiles
        n_tiles = valid_mask.shape[0]
        tile_stats = []
        
        for i in range(n_tiles):
            if valid_mask[i].shape != roi_mask.shape:
                logging.info(f"[{partition_id}]     Tile {i} shape {valid_mask[i].shape} does not match ROI shape {roi_mask.shape}, cropping to the common region")
                common_height = min(valid_mask[i].shape[0], roi_mask.shape[0])
                common_width = min(valid_mask[i].shape[1], roi_mask.shape[1])
                
                tile_valid = valid_mask[i][:common_height, :common_width]
                roi_cropped = roi_mask[:common_height, :common_width]
            else:
                tile_valid = valid_mask[i]
                roi_cropped = roi_mask
            
            valid_count = int(np.sum(tile_valid & roi_cropped))
            roi_count = int(np.sum(roi_cropped))
            valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
            logging.info(f"[{partition_id}]     Tile {i}: valid pixels in ROI {valid_count}/{roi_count} ({valid_pct:.2f}%)")
            tile_stats.append(valid_pct)
        
        # Overall coverage is the maximum over tiles (simplification)
        if tile_stats:
            total_valid_pct = max(tile_stats)
            logging.info(f"[{partition_id}]     Merged: valid-pixel coverage in ROI {total_valid_pct:.2f}%")
        else:
            total_valid_pct = 0
            logging.info(f"[{partition_id}]     No valid coverage")
        
        return valid_mask, total_valid_pct
    
    else:
        # Single tile, shapes already match
        tile_valid = valid_mask & roi_mask
        valid_count = int(np.sum(tile_valid))
        roi_count = int(np.sum(roi_mask))
        valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
        logging.info(f"[{partition_id}]     Single tile: valid pixels in ROI {valid_count}/{roi_count} ({valid_pct:.2f}%)")
        
        return valid_mask, valid_pct

def process_item(item, tpl, bbox_proj, mask_np, resolution, chunksize, temp_dir, min_coverage, partition_id, retries=2):
    """Process one Sentinel-1 item and write VV/VH TIFFs, with retries
    
    Returns:
        tuple: (vv_path, vh_path, status) 
        status: "success", "skipped", "failed"
    """
    orbit_state = item.properties.get("sat:orbit_state", "unknown")
    date_str = item.properties.get("datetime").split("T")[0]
    item_id = item.id
    
    temp_dir = Path(temp_dir)
    
    uid = uuid.uuid4().hex[:8]
    vv_temp = temp_dir / f"{date_str}_vv_{orbit_state}_{uid}.tiff"
    vh_temp = temp_dir / f"{date_str}_vh_{orbit_state}_{uid}.tiff"
    
    logging.info(f"[{partition_id}]   Processing item {item_id} ({date_str}_{orbit_state})")
    
    for attempt in range(retries + 1):
        try:
            with timeout_handler(ITEM_TIMEOUT):
                ds = stackstac.stack(
                    [item], 
                    bounds=bbox_proj,
                    epsg=tpl["crs"].to_epsg(),
                    resolution=resolution,
                    chunksize=chunksize
                )
                
                if 'vv' not in ds.band.values or 'vh' not in ds.band.values:
                    logging.warning(f"[{partition_id}]   {date_str}_{orbit_state} missing required bands, skipping")
                    return None, None, "skipped"
                    
                vv_data = ds.sel(band="vv").squeeze()
                vh_data = ds.sel(band="vh").squeeze()
                
                # Force the actual data load
                try:
                    vv_values = vv_data.compute()
                    vh_values = vh_data.compute()
                except Exception as compute_error:
                    if attempt < retries:
                        logging.warning(f"[{partition_id}]   attempt {attempt+1}/{retries+1} failed to compute data: {compute_error}, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        raise
                
                vv_shape = vv_values.shape
                logging.debug(f"[{partition_id}]   VV data shape: {vv_shape}, ROI shape: {mask_np.shape}")
                
                logging.info(f"[{partition_id}]   Analysing VV coverage")
                vv_valid_mask, vv_valid_pct = analyze_coverage(vv_values, mask_np, partition_id)
                
                logging.info(f"[{partition_id}]   Analysing VH coverage")
                vh_valid_mask, vh_valid_pct = analyze_coverage(vh_values, mask_np, partition_id)
                
                if vv_valid_pct < min_coverage and vh_valid_pct < min_coverage:
                    logging.warning(f"[{partition_id}]   ⚠️ {date_str}_{orbit_state} valid coverage VV={vv_valid_pct:.2f}%, VH={vh_valid_pct:.2f}% both below {min_coverage}%, skipping")
                    return None, None, "skipped"
                
                if vv_valid_pct >= min_coverage:
                    logging.info(f"[{partition_id}]   Processing VV band")
                    
                    common_height = min(vv_values.shape[0], mask_np.shape[0])
                    common_width = min(vv_values.shape[1], mask_np.shape[1])
                    
                    vv_cropped = vv_values[:common_height, :common_width]
                    mask_cropped = mask_np[:common_height, :common_width]
                    
                    vv_db = amplitude_to_db(vv_cropped, mask=mask_cropped)
                    
                    vv_final = np.zeros((tpl["height"], tpl["width"]), dtype=np.int16)
                    vv_final[:common_height, :common_width] = vv_db
                    
                    vv_metadata = {
                        "band_desc": "VV polarization, amplitude to dB, +50 offset, scale=200",
                        "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                        "ORBIT_STATE": orbit_state,
                        "DATE_ACQUIRED": date_str,
                        "POLARIZATION": "VV",
                        "DESCRIPTION": "Sentinel-1 SAR data (VV). Values are amplitude converted to dB, shifted by +50, scaled by 200."
                    }
                    write_tiff(vv_final, vv_temp, tpl, "int16", vv_metadata)
                    
                    if not validate_tiff(vv_temp, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        logging.error(f"[{partition_id}]   ✗ VV output failed validation")
                        if vv_temp.exists():
                            vv_temp.unlink()
                        vv_temp = None
                    else:
                        logging.info(f"[{partition_id}]   ✓ VV: {os.path.getsize(vv_temp)/1e6:.2f} MB")
                else:
                    logging.warning(f"[{partition_id}]   ⚠️ insufficient VV coverage, skipping")
                    vv_temp = None
                
                if vh_valid_pct >= min_coverage:
                    logging.info(f"[{partition_id}]   Processing VH band")
                    
                    common_height = min(vh_values.shape[0], mask_np.shape[0])
                    common_width = min(vh_values.shape[1], mask_np.shape[1])
                    
                    vh_cropped = vh_values[:common_height, :common_width]
                    mask_cropped = mask_np[:common_height, :common_width]
                    
                    vh_db = amplitude_to_db(vh_cropped, mask=mask_cropped)
                    
                    vh_final = np.zeros((tpl["height"], tpl["width"]), dtype=np.int16)
                    vh_final[:common_height, :common_width] = vh_db
                    
                    vh_metadata = {
                        "band_desc": "VH polarization, amplitude to dB, +50 offset, scale=200",
                        "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                        "ORBIT_STATE": orbit_state,
                        "DATE_ACQUIRED": date_str,
                        "POLARIZATION": "VH",
                        "DESCRIPTION": "Sentinel-1 SAR data (VH). Values are amplitude converted to dB, shifted by +50, scaled by 200."
                    }
                    write_tiff(vh_final, vh_temp, tpl, "int16", vh_metadata)
                    
                    if not validate_tiff(vh_temp, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        logging.error(f"[{partition_id}]   ✗ VH output failed validation")
                        if vh_temp.exists():
                            vh_temp.unlink()
                        vh_temp = None
                    else:
                        logging.info(f"[{partition_id}]   ✓ VH: {os.path.getsize(vh_temp)/1e6:.2f} MB")
                else:
                    logging.warning(f"[{partition_id}]   ⚠️ insufficient VH coverage, skipping")
                    vh_temp = None
                
                if vv_temp or vh_temp:
                    return vv_temp, vh_temp, "success"
                else:
                    # Nothing written, but only because coverage was too low: count as skipped
                    return None, None, "skipped"
                
        except TimeoutException as e:
            if attempt < retries:
                logging.warning(f"[{partition_id}]   attempt {attempt+1}/{retries+1} timed out on {date_str}_{orbit_state}, retrying...")
                time.sleep(5)
                continue
            else:
                logging.error(f"[{partition_id}]   ⚠️ timed out processing {date_str}_{orbit_state}: {e}")
                return None, None, "failed"
        except (RuntimeError, Exception) as e:
            error_msg = str(e).lower()
            is_retriable_error = any(keyword in error_msg for keyword in [
                'rasterio', 'read', 'tiff', 'network', 'timeout', 'connection', 'io'
            ])
            
            if is_retriable_error and attempt < retries:
                logging.warning(f"[{partition_id}]   attempt {attempt+1}/{retries+1} failed on {date_str}_{orbit_state}: {type(e).__name__} - {e}, retrying...")
                time.sleep(3)
                continue
            else:
                logging.error(f"[{partition_id}]   ✗ error processing {date_str}_{orbit_state}: {type(e).__name__} - {e}")
                return None, None, "failed"
    
    logging.error(f"[{partition_id}]   ✗ all retries failed for {date_str}_{orbit_state}")
    return None, None, "failed"

def mosaic_tiffs(tiff_paths, output_path, tpl, date_str, orbit_state, polarization, partition_id):
    try:
        output_path = Path(output_path)
        
        src_files = []
        for path in tiff_paths:
            if path and os.path.exists(path):
                try:
                    src = rasterio.open(path)
                    src_files.append(src)
                except Exception as e:
                    logging.warning(f"[{partition_id}]   failed to open {path} for mosaicking: {e}")
        
        if not src_files:
            logging.warning(f"[{partition_id}]   no valid files to mosaic for {date_str}_{polarization}_{orbit_state}")
            return None
        
        logging.info(f"[{partition_id}]   Mosaicking {len(src_files)} {polarization} files ({date_str}_{orbit_state})")
        mosaic_data, out_transform = merge(src_files, nodata=0)
        
        for src in src_files:
            src.close()
        
        if mosaic_data.shape[0] < 1:
            logging.error(f"[{partition_id}]   unexpected mosaic data structure ({date_str}_{polarization}_{orbit_state})")
            return None
        
        metadata = {
            "band_desc": f"{polarization} polarization, amplitude to dB, +50 offset, scale=200",
            "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "ORBIT_STATE": orbit_state,
            "DATE_ACQUIRED": date_str,
            "POLARIZATION": polarization,
            "MOSAIC_SOURCE_COUNT": len(src_files),
            "DESCRIPTION": f"Mosaicked Sentinel-1 SAR data ({polarization}). Values are amplitude converted to dB, shifted by +50, scaled by 200."
        }
        
        write_tiff(mosaic_data[0], output_path, tpl, "int16", metadata)
        
        if not validate_tiff(output_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.error(f"[{partition_id}]   ✗ mosaic TIFF failed validation ({date_str}_{polarization}_{orbit_state})")
            if Path(output_path).exists():
                Path(output_path).unlink()
            return None
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"[{partition_id}]   ✓ created mosaic {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
    
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ error creating mosaic {date_str}_{polarization}_{orbit_state}: {e}")
        return None

def process_day_orbit(key, items, tpl, bbox_proj, mask_np, out_dir, resolution, chunksize, min_coverage, partition_id, overwrite=False):
    """Process every item for one date and orbit state."""
    date_str, orbit_state = key.split("_")
    logging.info(f"[{partition_id}] → {key} (item={len(items)})")
    t0 = time.time()
    
    try:
        with timeout_handler(DAY_TIMEOUT):
            out_dir = Path(out_dir)
            
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Output goes straight into the target directory, no sub-folders
            vv_out = out_dir / f"{date_str}_vv_{orbit_state}.tiff"
            vh_out = out_dir / f"{date_str}_vh_{orbit_state}.tiff"
            
            if not overwrite and vv_out.exists() and vh_out.exists():
                vv_valid = validate_tiff(vv_out, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"])
                vh_valid = validate_tiff(vh_out, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"])
                
                if vv_valid and vh_valid:
                    logging.info(f"[{partition_id}]   valid files already exist, skipping")
                    return True
                else:
                    logging.warning(f"[{partition_id}]   files exist but failed validation, reprocessing")
                    if not vv_valid and vv_out.exists():
                        vv_out.unlink()
                    if not vh_valid and vh_out.exists():
                        vh_out.unlink()
            
            temp_dir = tempfile.mkdtemp(prefix=f"s1_{date_str}_{orbit_state}_")
            logging.debug(f"[{partition_id}]   temp dir: {temp_dir}")
            
            try:
                # Process each item; partial failure is tolerated
                vv_temp_files = []
                vh_temp_files = []
                processed_count = 0
                failed_count = 0
                skipped_count = 0
                
                for i, item in enumerate(items):
                    item_start_time = time.time()
                    logging.info(f"[{partition_id}]   processing item {i+1}/{len(items)}")
                    
                    vv_path, vh_path, status = process_item(item, tpl, bbox_proj, mask_np, resolution, chunksize, temp_dir, min_coverage, partition_id)
                    
                    if status == "success":
                        processed_count += 1
                        if vv_path:
                            vv_temp_files.append(str(vv_path))
                        if vh_path:
                            vh_temp_files.append(str(vh_path))
                        
                        item_duration = time.time() - item_start_time
                        logging.info(f"[{partition_id}]   item {i+1} succeeded in {item_duration:.1f}s")
                    elif status == "skipped":
                        skipped_count += 1
                        item_duration = time.time() - item_start_time
                        logging.info(f"[{partition_id}]   item {i+1} skipped (insufficient coverage or missing bands) in {item_duration:.1f}s")
                    else:  # status == "failed"
                        failed_count += 1
                        item_duration = time.time() - item_start_time
                        logging.warning(f"[{partition_id}]   item {i+1} failed in {item_duration:.1f}s")
                
                logging.info(f"[{partition_id}]   item stats: succeeded {processed_count}, skipped {skipped_count}, failed {failed_count} (of {len(items)})")
                
                if not vv_temp_files and not vh_temp_files:
                    if processed_count == 0 and skipped_count > 0:
                        logging.info(f"[{partition_id}]   {key} all items skipped for insufficient coverage or missing bands")
                        return True  # skipping is not a failure
                    else:
                        logging.warning(f"[{partition_id}]   no valid files produced for {key}")
                        return False
                
                vv_success = False
                if vv_temp_files:
                    if len(vv_temp_files) == 1:
                        logging.info(f"[{partition_id}]   only one valid VV file, using it directly")
                        shutil.copy2(vv_temp_files[0], vv_out)
                        vv_success = True
                    else:
                        vv_mosaic = mosaic_tiffs(vv_temp_files, vv_out, tpl, date_str, orbit_state, "VV", partition_id)
                        vv_success = vv_mosaic is not None
                
                vh_success = False
                if vh_temp_files:
                    if len(vh_temp_files) == 1:
                        logging.info(f"[{partition_id}]   only one valid VH file, using it directly")
                        shutil.copy2(vh_temp_files[0], vh_out)
                        vh_success = True
                    else:
                        vh_mosaic = mosaic_tiffs(vh_temp_files, vh_out, tpl, date_str, orbit_state, "VH", partition_id)
                        vh_success = vh_mosaic is not None
                
                total_duration = time.time() - t0
                if vv_success and vh_success:
                    logging.info(f"[{partition_id}] ← {key} VV and VH processed in {total_duration:.1f}s")
                    return True
                elif vv_success:
                    logging.info(f"[{partition_id}] ← {key} only VV processed, {total_duration:.1f}s")
                    return True
                elif vh_success:
                    logging.info(f"[{partition_id}] ← {key} only VH processed, {total_duration:.1f}s")
                    return True
                else:
                    logging.error(f"[{partition_id}] ← {key} failed after {total_duration:.1f}s")
                    return False
                    
            finally:
                try:
                    shutil.rmtree(temp_dir)
                    logging.debug(f"[{partition_id}]   cleaned temp dir: {temp_dir}")
                except Exception as e:
                    logging.warning(f"[{partition_id}]   failed to clean temp dir: {e}")
                    
    except TimeoutException as e:
        total_duration = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {key} timed out ({total_duration:.1f}s > {DAY_TIMEOUT}s): {e}")
        return False
    except Exception as e:
        total_duration = time.time() - t0
        logging.error(f"[{partition_id}] ✗ error processing {key} ({total_duration:.1f}s): {type(e).__name__} - {e}")
        return False

def main():
    args = get_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.debug, out_dir, args.partition_id)
    logging.info(f"[{args.partition_id}] ⚡ S1 Fast Processor started (flexible parallel edition)"); 
    log_sys(args.partition_id)
    logging.info(f"[{args.partition_id}] Timeouts: overall {PROCESS_TIMEOUT//60} min, per day {DAY_TIMEOUT//60} min, per item {ITEM_TIMEOUT//60} min")
    logging.info(f"[{args.partition_id}] Date range: {args.start_date} → {args.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np = load_roi(Path(args.input_tiff), args.partition_id)
    
    if args.orbit_state == "both":
        logging.info(f"[{args.partition_id}] Searching ascending and descending orbits")
        items = search_items(bbox_ll, f"{args.start_date}/{args.end_date}", partition_id=args.partition_id)
    else:
        logging.info(f"[{args.partition_id}] Searching {args.orbit_state} orbit")
        items = search_items(bbox_ll, f"{args.start_date}/{args.end_date}", args.orbit_state, args.partition_id)
    
    if not items:
        logging.warning(f"[{args.partition_id}] No matching scenes, exiting")
        return

    groups = group_by_date_orbit(items, args.partition_id)

    with make_client(args.dask_workers, args.worker_memory, args.partition_id):
        report_path = out_dir / f"dask-report-{args.partition_id}.html"
        with performance_report(filename=report_path):
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_group = {}
                for key, group_items in groups.items():
                    future = executor.submit(
                        process_day_orbit, key, group_items, tpl, bbox_proj, mask_np,
                        str(out_dir), SAR_RESOLUTION, args.chunksize, args.min_coverage,
                        args.partition_id, args.overwrite
                    )
                    future_to_group[future] = key
                
                results = []
                for future in concurrent.futures.as_completed(future_to_group):
                    key = future_to_group[future]
                    try:
                        success = future.result()
                        results.append(success)
                    except Exception as e:
                        logging.error(f"[{args.partition_id}] exception while processing {key}: {e}")
                        results.append(False)
    
    success_count = sum(results)
    total_count = len(results)
    
    logging.info(f"[{args.partition_id}] ✅ Partition complete: succeeded {success_count}/{total_count} days")
    logging.info(f"[{args.partition_id}] 📊 Dask performance report saved: {report_path}")
    
    if success_count == 0:
        sys.exit(1)  # all failed
    elif success_count < total_count:
        logging.warning(f"[{args.partition_id}] ⚠️  some dates failed ({total_count - success_count}/{total_count})")
        sys.exit(2)  # partial failure
    else:
        sys.exit(0)  # all succeeded

if __name__ == "__main__":
    main()