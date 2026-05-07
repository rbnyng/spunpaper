#!/usr/bin/env python3
"""
s2_fast_processor_small_patches.py — 专为小patch优化的Sentinel-2 L2A处理器
更新：2025-05-26 (修复TIFF块大小问题)
特别针对256x256小patch优化，修复了TIFF写入块大小限制问题
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

# ▶ distributed 版本兼容
try:
    from distributed.comm.core import CommClosedError
except ImportError:
    from distributed import CommClosedError

# 抑制不必要的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask.core")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

# ─── 常量 ──────────────────────────────────────────────────────────────────────
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

# SCL 无效值列表 (无云/无阴影/非水体等为有效)
SCL_INVALID = {0, 1, 2, 3, 8, 9, np.nan}

# SCL 值描述，用于日志
SCL_DESCRIPTIONS = {
    0: "无数据", 1: "饱和或缺陷", 2: "暗影区", 3: "未分类", 4: "植被",
    5: "裸露土壤", 6: "水体", 7: "未使用", 8: "云", 9: "薄云", 10: "雪", 11: "云阴影"
}

# 有效覆盖率阈值 (低于此值跳过处理)
MIN_VALID_COVERAGE = 5.0  # 百分比

# 临时文件目录设置（默认使用系统临时目录）
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# 小patch优化的超时设置（秒）- 大幅减少
PROCESS_TIMEOUT = 60 * 60   # 整体处理超时：60分钟
DAY_TIMEOUT = 20 * 60       # 单日处理超时：20分钟  
ITEM_TIMEOUT = 10 * 60      # 单个item处理超时：10分钟
BAND_TIMEOUT = 5 * 60       # 单个波段处理超时：5分钟
SCL_BAND_TIMEOUT = 5 * 60   # SCL波段处理超时：5分钟

# 网络请求重试配置
MAX_RETRIES = 3  # 减少重试次数
RETRY_BACKOFF_FACTOR = 5

# 小patch并行处理配置
DEFAULT_MAX_WORKERS = 2     # 大幅减少并发数

# ─── 辅助函数：确保数据是numpy数组 ──────────────────────────────────────────────
def ensure_numpy_array(data, name="data"):
    """确保数据是numpy数组，处理xarray DataArray的情况"""
    if hasattr(data, 'values'):
        # xarray DataArray 或 dask array
        if hasattr(data.values, 'compute'):
            # dask array
            return data.values.compute()
        else:
            # numpy array 在 DataArray 中
            return data.values
    elif hasattr(data, 'compute'):
        # 直接的 dask array
        return data.compute()
    else:
        # 已经是 numpy array
        return np.asarray(data)

# ─── TIFF 块大小优化函数 ──────────────────────────────────────────────────────────
def calculate_optimal_blocksize(width, height):
    """
    计算最优的TIFF块大小，确保是16的倍数且适合小patch
    """
    # 对于小patch，如果尺寸小于512，使用非分块模式
    if width <= 512 and height <= 512:
        return None, None  # 不使用分块
    
    # 对于大一些的patch，计算合适的块大小
    def round_to_multiple_of_16(value, max_val):
        """将值舍入到16的倍数，但不超过max_val"""
        if value <= 16:
            return 16
        rounded = ((value + 15) // 16) * 16
        return min(rounded, max_val)
    
    # 计算块大小，但确保是16的倍数
    target_block_size = 256
    
    blockx = round_to_multiple_of_16(min(target_block_size, width), width)
    blocky = round_to_multiple_of_16(min(target_block_size, height), height)
    
    return blockx, blocky

# ─── 超时控制 ──────────────────────────────────────────────────────────────────
class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    """超时上下文管理器"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"操作超时 ({seconds}秒)")
    
    # 检查是否在主线程中（Unix信号只能在主线程中处理）
    import threading
    if threading.current_thread() is not threading.main_thread():
        # 如果不是主线程，只是yield而不设置信号
        yield
        return
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 恢复原信号处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ─── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    P = argparse.ArgumentParser("Small Patch Optimized Sentinel-2 L2A Processor")
    P.add_argument("--input_tiff",   required=True, help="ROI 掩膜或者模板栅格")
    P.add_argument("--start_date",   required=True, help="开始日期 (YYYY-MM-DD[THH:MM:SS]) - 包含该时间点")
    P.add_argument("--end_date",     required=True, help="结束日期 (YYYY-MM-DD[THH:MM:SS]) - 包含该时间点")
    P.add_argument("--output",       default="sentinel2_output", help="输出目录")
    P.add_argument("--max_cloud",    type=float, default=90, help="最大云量百分比")
    P.add_argument("--dask_workers", type=int,   default=1, help="本分区的 Dask worker 数")
    P.add_argument("--worker_memory",type=int,   default=4, help="每个 worker 内存 GB")
    P.add_argument("--chunksize",    type=int,   default=256, help="stackstac x/y chunk 大小")
    P.add_argument("--resolution",   type=int,   default=10, help="输出分辨率 (米)")
    P.add_argument("--overwrite",    action="store_true", help="覆盖已存在文件")
    P.add_argument("--debug",        action="store_true", help="输出调试日志")
    P.add_argument("--min_coverage", type=float, default=MIN_VALID_COVERAGE,
                   help="最小有效像素覆盖率 (百分比)")
    P.add_argument("--partition_id", default="unknown",
                   help="分区ID（用于日志标识）")
    P.add_argument("--temp_dir",     default=TEMP_DIR,
                   help="临时文件存储目录，默认使用系统临时目录")
    return P.parse_args()

# ─── logging ──────────────────────────────────────────────────────────────────
def setup_logging(debug: bool, out_dir: Path, partition_id: str):
    """设置日志，包含分区ID标识"""
    fmt = f"%(asctime)s [{partition_id}] [%(levelname)s] %(message)s"
    lvl = logging.DEBUG if debug else logging.INFO
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(lvl)
    
    # 清除现有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter(fmt)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（分区特定的日志文件）
    file_handler = logging.FileHandler(
        out_dir / f"s2_{partition_id}_detail.log", 
        "a", 
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_sys(partition_id: str):
    m = psutil.virtual_memory()
    logging.info(f"[{partition_id}] 系统信息 - CPU {os.cpu_count()} | "
                 f"RAM {m.total/1e9:.1f} GB (free {m.available/1e9:.1f} GB)")

def fmt_bbox(b):
    return f"{b[0]:.5f},{b[1]:.5f} ⇢ {b[2]:.5f},{b[3]:.5f}"

# ─── 简化的Dask客户端 ─────────────────────────────────────────────────────────────────
def make_simple_client(req_workers:int, req_mem:int, partition_id: str):
    """创建简化的Dask客户端，专为小patch优化"""
    # 对小patch，限制资源使用
    workers = min(req_workers, 2)  # 最多2个worker
    memory_gb = min(req_mem, 8)    # 最多8GB内存
    
    logging.info(f"[{partition_id}] 创建轻量级Dask客户端: {workers} workers, {memory_gb}GB memory")
    
    # 为不同分区自动分配dashboard端口，避免冲突
    port_base = 8900
    port_range = 200
    dashboard_port = port_base + (abs(hash(partition_id)) % port_range)
    
    # 小patch优化的Dask配置
    dask_config = {
        "distributed.worker.memory.target": 0.70,
        "distributed.worker.memory.spill": 0.80,   
        "distributed.worker.memory.pause": 0.90,
        "array.slicing.split_large_chunks": False,  # 小数据不需要分割
        "optimization.fuse.active": False,          # 简化优化
        "distributed.worker.daemon": False,
        "distributed.scheduler.work-stealing": False,  # 减少复杂性
    }
    
    dask.config.set(dask_config)
    
    try:
        cluster = LocalCluster(
            n_workers         = workers,
            threads_per_worker= 2,  # 减少线程数
            processes         = True,
            memory_limit      = f"{memory_gb}GB",
            dashboard_address = f":{dashboard_port}",
            silence_logs      = "ERROR",
        )
        
        cli = Client(cluster, asynchronous=False, timeout=30)
        logging.info(f"[{partition_id}] Dask客户端创建成功，dashboard: {cli.dashboard_link}")
        return cli
    except Exception as e:
        logging.error(f"[{partition_id}] Dask客户端创建失败: {e}")
        # 返回None，后续使用同步处理
        return None

# ─── ROI & 掩膜 ────────────────────────────────────────────────────────────────
def load_roi(tiff: Path, partition_id: str):
    """加载ROI数据，针对小patch优化"""
    with rasterio.open(tiff) as src:
        tpl = dict(crs=src.crs,
                   transform=src.transform,
                   width=src.width,
                   height=src.height)
        bbox_proj = src.bounds
        bbox_ll   = transform_bounds(src.crs, "EPSG:4326", *bbox_proj,
                                     densify_pts=21)
        
        # 读取掩膜并转换为1位数据以节省内存
        mask_np = (src.read(1) > 0).astype(np.uint8)
        
    # 检查ROI大小
    roi_size_kb = (mask_np.size * mask_np.itemsize) / 1024
    is_small_patch = tpl['width'] <= 512 and tpl['height'] <= 512  # 判断是否为小patch
    
    logging.info(f"[{partition_id}] ROI (CRS={tpl['crs']}): {tpl['width']}×{tpl['height']} ({roi_size_kb:.1f} KB)")
    logging.info(f"[{partition_id}] 小patch模式: {'是' if is_small_patch else '否'}")
    logging.info(f"[{partition_id}] ROI bbox proj: {fmt_bbox(bbox_proj)}")
    logging.info(f"[{partition_id}] ROI bbox lon/lat: {fmt_bbox(bbox_ll)}")
    
    return tpl, bbox_proj, bbox_ll, mask_np, is_small_patch

# ─── STAC ─────────────────────────────────────────────────────────────────────
def search_items(bbox_ll, date_range:str, max_cloud, partition_id: str):
    """搜索STAC项，针对小patch优化了重试逻辑"""
    # 解析开始和结束时间
    start_date, end_date = date_range.split("/")
    
    # 解析结束时间并添加一秒，确保包含结束时间点
    try:
        end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00').replace(' ', 'T'))
        end_dt_plus = end_dt + datetime.timedelta(seconds=1)
        search_date_range = f"{start_date}/{end_dt_plus.isoformat()}"
    except ValueError:
        logging.warning(f"[{partition_id}] 无法解析结束日期格式，使用原始范围: {date_range}")
        search_date_range = date_range
    
    logging.info(f"[{partition_id}] STAC 搜索日期范围: {search_date_range}")
    
    # 简化的重试逻辑
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
            logging.info(f"[{partition_id}] STAC 命中 {len(items)} item (云 < {max_cloud}%)")
            if items:
                b = np.array([it.bbox for it in items])
                union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
                logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
            return items
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logging.error(f"[{partition_id}] STAC搜索失败 (尝试{retries}/{max_retries+1}): {e}")
                raise
            
            retry_delay = min(30, retry_delay * RETRY_BACKOFF_FACTOR)
            jitter = random.uniform(0.8, 1.2)
            actual_delay = retry_delay * jitter
            
            logging.warning(f"[{partition_id}] STAC搜索失败 (尝试{retries}/{max_retries+1}): {e}, {actual_delay:.1f}秒后重试...")
            time.sleep(actual_delay)

def group_by_date(items, partition_id: str):
    """将items按日期分组"""
    g = {}
    for it in items:
        d = it.properties["datetime"][:10]
        g.setdefault(d, []).append(it)
    logging.info(f"[{partition_id}] ⇒ {len(g)} 观测日")
    return dict(sorted(g.items()))

# ─── baseline 校正 ─────────────────────────────────────────────────────────────
def harmonize_arr(arr: np.ndarray, date_key:str):
    """执行Baseline校正"""
    if datetime.datetime.strptime(date_key, "%Y-%m-%d") > BASELINE_CUTOFF:
        # 处理NaN值以避免警告
        valid_mask = ~np.isnan(arr) & (arr >= BASELINE_OFFSET)
        np.subtract(arr, BASELINE_OFFSET, out=arr, where=valid_mask)
    return arr

# ─── GeoTIFF 写出（修复版） ──────────────────────────────────────────────────────────────
def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    """写出GeoTIFF，针对小patch优化，修复了块大小问题"""
    # 处理NaN值
    if np.isnan(np_arr).any():
        np_arr = np.nan_to_num(np_arr, nan=0)
    
    # 计算最优块大小
    blockx, blocky = calculate_optimal_blocksize(tpl["width"], tpl["height"])
    
    # 基本profile
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
    
    # 根据是否需要分块来设置profile
    if blockx is not None and blocky is not None:
        # 大patch使用分块
        profile.update({
            "tiled": True,
            "blockxsize": blockx,
            "blockysize": blocky
        })
        logging.debug(f"使用分块模式: {blockx}x{blocky}")
    else:
        # 小patch不使用分块
        profile["tiled"] = False
        logging.debug(f"使用非分块模式（小patch {tpl['width']}x{tpl['height']}）")
    
    # 写入文件
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np_arr.astype(dtype, copy=False), 1)
        
        # 添加元数据
        if metadata:
            dst.update_tags(**metadata)

# ─── 验证TIFF ─────────────────────────────────────────────────────────────────
def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    """验证TIFF文件是否有效"""
    try:
        with rasterio.open(file_path) as src:
            # 检查基本属性
            if src.shape != expected_shape:
                logging.warning(f"验证失败: {file_path} 形状不匹配. 预期 {expected_shape}, 得到 {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logging.warning(f"验证失败: {file_path} CRS不匹配. 预期 {expected_crs}, 得到 {src.crs}")
                return False
            
            # 检查数据存在性（通过统计数据，避免读取整个数组）
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"验证失败: {file_path} 波段全为零")
                return False
            
            logging.debug(f"TIFF验证通过: {file_path}，形状={src.shape}")
            return True
            
    except Exception as e:
        logging.error(f"验证TIFF {file_path} 时出错: {e}")
        return False

# ─── SCL 质量评估（简化版）─────────────────────────────────────────────────────────────
def is_valid_scl(scl_arr):
    """判断SCL值是否为有效观测(非云/阴影/水体等)"""
    return ~np.isin(np.nan_to_num(scl_arr, nan=0), list(SCL_INVALID - {np.nan}))

def process_scl_simple(scl_arr, roi_mask, partition_id="unknown"):
    """
    简化的SCL处理，专为小patch优化，修复了DataArray问题
    """
    # 确保scl_arr是numpy数组
    scl_arr = ensure_numpy_array(scl_arr, "scl_arr")
    
    # 获取数组形状并处理单tile和多tile情况
    if len(scl_arr.shape) == 3:
        n_tiles, scl_height, scl_width = scl_arr.shape
    else:
        scl_height, scl_width = scl_arr.shape
        n_tiles = 1
        scl_arr = scl_arr.reshape(1, scl_height, scl_width)
    
    # 确保ROI掩膜是布尔类型
    roi_mask = roi_mask.astype(bool)
    roi_height, roi_width = roi_mask.shape
    
    # 处理形状不匹配
    if scl_height != roi_height or scl_width != roi_width:
        logging.debug(f"[{partition_id}] SCL形状调整: {(scl_height, scl_width)} -> {(roi_height, roi_width)}")
        use_height = min(scl_height, roi_height)
        use_width = min(scl_width, roi_width)
        scl_arr = scl_arr[:, :use_height, :use_width]
        roi_mask = roi_mask[:use_height, :use_width]
    
    # 计算有效掩膜
    valid_mask = is_valid_scl(scl_arr)
    
    # 计算覆盖率
    roi_pixel_count = np.sum(roi_mask)
    
    # 简化的tile选择：选择第一个有效tile
    tile_selection = np.full(roi_mask.shape, -1, dtype=np.int8)
    
    for tile_idx in range(n_tiles):
        current_valid = valid_mask[tile_idx] & roi_mask & (tile_selection < 0)
        tile_selection[current_valid] = tile_idx
    
    valid_pixel_count = np.sum(tile_selection >= 0)
    valid_pct = 100.0 * valid_pixel_count / roi_pixel_count if roi_pixel_count > 0 else 0.0
    
    logging.info(f"[{partition_id}] SCL处理结果: 有效像素 {valid_pixel_count}/{roi_pixel_count}, 覆盖率 {valid_pct:.2f}%")
    
    return valid_mask, tile_selection, valid_pct

def create_scl_mosaic_simple(scl_arr, tile_selection, roi_mask, target_shape, date_key=None, partition_id="unknown"):
    """
    简化的SCL镶嵌创建，修复了DataArray问题
    """
    try:
        # 确保scl_arr是numpy数组
        scl_arr = ensure_numpy_array(scl_arr, "scl_arr")
        
        if len(scl_arr.shape) == 3:
            n_tiles, arr_height, arr_width = scl_arr.shape
        else:
            arr_height, arr_width = scl_arr.shape
            n_tiles = 1
            scl_arr = scl_arr.reshape(1, arr_height, arr_width)
        
        target_height, target_width = target_shape
        result = np.zeros(target_shape, dtype=np.uint8)
        
        # 确定处理区域
        common_height = min(arr_height, roi_mask.shape[0], target_height, tile_selection.shape[0])
        common_width = min(arr_width, roi_mask.shape[1], target_width, tile_selection.shape[1])
        
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # 简化的镶嵌逻辑
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            y_coords, x_coords = np.where(valid_roi)
            tile_indices = tile_sel_crop[y_coords, x_coords]
            result[y_coords, x_coords] = scl_arr[tile_indices, y_coords, x_coords]
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] SCL镶嵌失败: {e}")
        raise

# ─── 简化的镶嵌 ────────────────────────────────────────────────────────────────
def smart_mosaic_simple(data_arr, tile_selection, roi_mask, partition_id="unknown"):
    """
    简化的智能镶嵌，专为小patch优化，修复了DataArray问题
    """
    try:
        # 确保data_arr是numpy数组
        data_arr = ensure_numpy_array(data_arr, "data_arr")
        
        # 单tile情况
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
        
        # 多tile情况
        n_tiles, data_height, data_width = data_arr.shape
        roi_mask = roi_mask.astype(bool)
        result = np.zeros(roi_mask.shape, dtype=data_arr.dtype)
        
        common_height = min(data_height, roi_mask.shape[0], tile_selection.shape[0])
        common_width = min(data_width, roi_mask.shape[1], tile_selection.shape[1])
        
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # 简化的镶嵌
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            y_coords, x_coords = np.where(valid_roi)
            tile_indices = tile_sel_crop[y_coords, x_coords]
            result[y_coords, x_coords] = data_arr[tile_indices, y_coords, x_coords]
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] 镶嵌失败: {e}")
        raise

# ─── 简化的波段处理 ───────────────────────────────────────────────────────────
def process_band_simple(items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                res, chunksize, out_path, is_small_patch, dask_client, partition_id="unknown", retries=2):
    """简化的波段处理，专为小patch优化，修复了DataArray问题"""
    t0 = time.time()
    
    if band_name == "SCL":
        logging.info(f"[{partition_id}]     波段 {band_name} 已在质量评估阶段处理，跳过")
        return True
    
    logging.info(f"[{partition_id}]     处理波段 {band_name}")
    
    # 检查输出路径是否已存在
    if out_path.exists():
        if validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]     {band_name} 已存在有效文件，跳过")
            return True
        else:
            logging.warning(f"[{partition_id}]     {band_name} 文件存在但无效，重新处理")
            out_path.unlink()
    
    # 重试循环
    for attempt in range(retries + 1):
        try:
            with timeout_handler(BAND_TIMEOUT):
                # 小patch的stackstac配置优化
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
                
                # 压平维度
                item_dim = None
                for dim in da.dims:
                    if dim not in ('band', 'x', 'y'):
                        if da.sizes[dim] > 1:
                            item_dim = dim
                        elif da.sizes[dim] == 1:
                            da = da.squeeze(dim, drop=True)
                
                band_da = da.sel(band=band_name)
                
                # 选择计算方式：对小patch使用同步计算，大patch使用dask
                if is_small_patch or dask_client is None:
                    # 同步计算，避免dask开销
                    try:
                        band_arr = band_da.compute()
                    except Exception as e:
                        logging.warning(f"[{partition_id}]     同步计算失败: {e}，尝试用dask")
                        if dask_client:
                            band_arr = ensure_numpy_array(band_da, "band_da")
                        else:
                            raise
                else:
                    # 使用dask计算
                    band_arr = ensure_numpy_array(band_da, "band_da")
                
                # 确保band_arr是numpy数组
                band_arr = ensure_numpy_array(band_arr, "band_arr")
                
                logging.debug(f"[{partition_id}]     {band_name} 数组形状: {band_arr.shape}, ROI形状: {mask_np.shape}")
                
                # 处理多tile情况
                if item_dim:
                    if tile_selection is not None:
                        arr = smart_mosaic_simple(band_arr, tile_selection, mask_np, partition_id)
                    else:
                        # 简单平均或选择第一个
                        if len(band_arr.shape) == 3:
                            arr = band_arr[0]  # 选择第一个tile
                        else:
                            arr = band_arr
                        
                        # 应用ROI掩膜
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
                    # 单tile情况
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
                
                # 应用基线校正
                harmonize_arr(arr, date_key)
                
                # 创建元数据
                metadata = {
                    "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    "DATE_ACQUIRED": date_key,
                    "BAND_NAME": band_name,
                    "ITEMS_COUNT": len(items)
                }
                
                # 写出GeoTIFF
                dtype = "uint16"
                write_tiff(arr, out_path, tpl, dtype, metadata)
                
                # 验证输出文件
                if not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                    logging.error(f"[{partition_id}]     ✗ 波段 {band_name} 验证失败")
                    if out_path.exists():
                        out_path.unlink()
                    continue
                
                logging.info(f"[{partition_id}]     ✓ {band_name:9s}  "
                            f"{os.path.getsize(out_path)/1e6:.2f} MB, 用时 {time.time()-t0:.1f}s")
                
                return True
                
        except TimeoutException as e:
            if attempt < retries:
                retry_delay = min(15, (attempt + 1) * 2)
                logging.warning(f"[{partition_id}]     波段 {band_name} 处理超时，{retry_delay}秒后重试 ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ 波段 {band_name} 处理超时: {e}")
                return False
                
        except Exception as e:
            if attempt < retries:
                retry_delay = min(15, (attempt + 1) * 2)
                logging.warning(f"[{partition_id}]     波段 {band_name} 处理错误: {e}, {retry_delay}秒后重试 ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ 波段 {band_name} 处理失败: {e}")
                return False
    
    return False

# ─── 简化的SCL评估和生成 ───────────────────────────────────────────────────────────
def process_scl_assessment_simple(items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                                       min_coverage, out_root, overwrite, is_small_patch, dask_client, partition_id="unknown"):
    """
    简化的SCL处理，专为小patch优化，修复了DataArray问题
    """
    t0 = time.time()
    logging.info(f"[{partition_id}]   处理SCL波段进行质量评估并生成SCL输出文件")
    
    # 构建SCL输出路径
    scl_out_name = BAND_MAPPING["SCL"]
    scl_dir = out_root / scl_out_name
    scl_dir.mkdir(parents=True, exist_ok=True)
    scl_out_path = scl_dir / f"{date_key}_mosaic.tiff"
    
    # 检查已存在文件
    if not overwrite and scl_out_path.exists():
        if validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]   SCL文件已存在且有效，跳过")
            # 生成简单的tile_selection
            tile_selection = np.zeros(mask_np.shape, dtype=np.int8)
            return True, 100.0, tile_selection
    
    # 检查SCL资产
    if not all('SCL' in item.assets for item in items):
        scl_items = [item for item in items if 'SCL' in item.assets]
        if not scl_items:
            logging.warning(f"[{partition_id}]   所有item均缺少SCL资产！")
            return False, 0.0, None
        items = scl_items
    
    try:
        with timeout_handler(SCL_BAND_TIMEOUT):
            # 小patch的stackstac配置
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
            
            # 压平维度
            item_dim = None
            for dim in da.dims:
                if dim not in ('band', 'x', 'y'):
                    if da.sizes[dim] > 1:
                        item_dim = dim
                    elif da.sizes[dim] == 1:
                        da = da.squeeze(dim, drop=True)
            
            scl_da = da.sel(band='SCL')
            
            # 选择计算方式并确保是numpy数组
            if is_small_patch or dask_client is None:
                try:
                    scl_arr = scl_da.compute()
                except Exception as e:
                    logging.warning(f"[{partition_id}]   SCL同步计算失败: {e}，尝试用dask")
                    scl_arr = ensure_numpy_array(scl_da, "scl_da")
            else:
                scl_arr = ensure_numpy_array(scl_da, "scl_da")
            
            # 确保scl_arr是numpy数组
            scl_arr = ensure_numpy_array(scl_arr, "scl_arr")
            
            # 简化的SCL处理
            valid_mask, tile_selection, valid_pct = process_scl_simple(scl_arr, mask_np, partition_id)
            
            if valid_pct < min_coverage:
                logging.warning(f"[{partition_id}]   ⚠️ {date_key} 有效覆盖率 {valid_pct:.2f}% < {min_coverage}%，跳过SCL生成")
                return False, valid_pct, None
            
            # 创建SCL镶嵌输出
            scl_output = create_scl_mosaic_simple(scl_arr, tile_selection, mask_np, (tpl["height"], tpl["width"]), date_key, partition_id)
            
            # 元数据
            metadata = {
                "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                "DATE_ACQUIRED": date_key,
                "BAND_NAME": "SCL",
                "ITEMS_COUNT": len(items),
                "VALID_COVERAGE_PCT": f"{valid_pct:.2f}"
            }
            
            # 写出SCL文件
            write_tiff(scl_output, scl_out_path, tpl, "uint8", metadata)
            
            if not validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                logging.error(f"[{partition_id}]   ✗ SCL文件验证失败")
                if scl_out_path.exists():
                    scl_out_path.unlink()
                return False, 0.0, None
            
            logging.info(f"[{partition_id}]   ✓ SCL 处理完成，有效率: {valid_pct:.2f}%, "
                        f"文件大小: {os.path.getsize(scl_out_path)/1e6:.2f} MB, 用时 {time.time()-t0:.1f}s")
            
            return True, valid_pct, tile_selection
            
    except TimeoutException as e:
        logging.error(f"[{partition_id}]   ✗ SCL处理超时: {e}")
        return False, 0.0, None
            
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ SCL处理失败: {e}")
        return False, 0.0, None

# ─── 简化的单日任务 ──────────────────────────────────────────────────────────────────
def process_day_simple(date_key:str, items, tpl, bbox_proj, mask_np,
                out_root:Path, res:int, chunksize:int,
                overwrite:bool, min_coverage:float, is_small_patch:bool, dask_client,
                partition_id:str="unknown") -> bool:
    """简化的单日处理，专为小patch优化"""
    logging.info(f"[{partition_id}] → {date_key} (item={len(items)})")
    t0 = time.time()
    
    try:
        with timeout_handler(DAY_TIMEOUT):
            # 创建波段输出目录
            for outname in BAND_MAPPING.values():
                band_dir = out_root / outname
                band_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查是否已全部处理完成
            if not overwrite:
                all_exist = True
                for band_name in S2_BANDS:
                    out_name = BAND_MAPPING[band_name]
                    out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                    if not out_path.exists() or not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        all_exist = False
                        break
                
                if all_exist:
                    logging.info(f"[{partition_id}]   所有波段已存在有效文件，跳过")
                    return True
            
            # 处理SCL波段
            scl_success, valid_pct, tile_selection = process_scl_assessment_simple(
                items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                min_coverage, out_root, overwrite, is_small_patch, dask_client, partition_id
            )
            
            if not scl_success:
                if valid_pct < min_coverage:
                    logging.warning(f"[{partition_id}]   {date_key} 有效覆盖率 {valid_pct:.2f}% < {min_coverage}%，跳过其他波段处理")
                    return True
                else:
                    logging.error(f"[{partition_id}]   {date_key} SCL处理失败，跳过该日期处理")
                    return False
            
            # 创建临时目录用于处理
            day_temp_dir = tempfile.mkdtemp(prefix=f"s2_{date_key}_", dir=TEMP_DIR)
            
            try:
                # 处理其他波段（不包括SCL）
                other_bands = [band for band in S2_BANDS if band != "SCL"]
                
                # 对小patch使用更少的并发
                max_workers = min(DEFAULT_MAX_WORKERS, len(other_bands)) if is_small_patch else min(4, len(other_bands))
                logging.info(f"[{partition_id}]   使用 {max_workers} 个线程处理 {len(other_bands)} 个波段")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    
                    for band_name in other_bands:
                        out_name = BAND_MAPPING[band_name]
                        out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                        
                        if not overwrite and out_path.exists() and validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                            logging.info(f"[{partition_id}]     波段 {band_name} 已存在有效文件，跳过")
                            continue
                        
                        temp_path = Path(day_temp_dir) / f"{band_name}_{date_key}.tiff"
                        
                        future = executor.submit(
                            process_band_simple, 
                            items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                            res, chunksize, temp_path, is_small_patch, dask_client, partition_id
                        )
                        futures[future] = (band_name, out_path, temp_path)
                    
                    # 处理结果
                    success_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        band_name, out_path, temp_path = futures[future]
                        try:
                            success = future.result()
                            if success:
                                if temp_path.exists() and validate_tiff(temp_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                                    shutil.copy2(temp_path, out_path)
                                    success_count += 1
                                    logging.info(f"[{partition_id}]     ✓ {band_name} 处理完成")
                                else:
                                    logging.error(f"[{partition_id}]     ✗ {band_name} 临时文件无效")
                            else:
                                logging.warning(f"[{partition_id}]     ✗ {band_name} 处理失败")
                        except Exception as e:
                            logging.error(f"[{partition_id}]     ✗ {band_name} 处理异常: {e}")
            finally:
                # 清理临时目录
                try:
                    shutil.rmtree(day_temp_dir)
                except:
                    pass
            
            # 记录处理结果
            total_other_bands = len(other_bands)
            total_bands = len(S2_BANDS)
            proc_time = time.time() - t0
            
            total_success = (1 if scl_success else 0) + success_count
            
            if total_success == total_bands:
                logging.info(f"[{partition_id}] ← {date_key} 全部波段处理成功 ({total_success}/{total_bands})，用时 {proc_time:.1f}s")
                return True
            elif total_success > 0:
                logging.warning(f"[{partition_id}] ← {date_key} 部分波段处理成功 ({total_success}/{total_bands})，用时 {proc_time:.1f}s")
                return True
            else:
                logging.error(f"[{partition_id}] ← {date_key} 所有波段处理失败，用时 {proc_time:.1f}s")
                return False
                
    except TimeoutException as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} 处理超时 ({proc_time:.1f}s): {e}")
        return False
    except Exception as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} 处理失败: {type(e).__name__} - {e}")
        return False

# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    a = get_args()
    out_dir = Path(a.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # 使用命令行指定的临时目录
    global TEMP_DIR
    TEMP_DIR = a.temp_dir
    
    setup_logging(a.debug, out_dir, a.partition_id)
    logging.info(f"[{a.partition_id}] ⚡ S2 Small Patch Processor 启动 (修复TIFF块大小问题)")
    log_sys(a.partition_id)
    logging.info(f"[{a.partition_id}] 处理超时设置: 总体 {PROCESS_TIMEOUT//60} 分钟, 单日 {DAY_TIMEOUT//60} 分钟")
    logging.info(f"[{a.partition_id}] 临时目录: {TEMP_DIR}")
    logging.info(f"[{a.partition_id}] 处理时间段: {a.start_date} → {a.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np, is_small_patch = load_roi(Path(a.input_tiff), a.partition_id)
    
    # 搜索STAC items
    search_date_range = f"{a.start_date}/{a.end_date}"
    
    items = search_items(bbox_ll, search_date_range, a.max_cloud, a.partition_id)
    if not items:
        logging.warning(f"[{a.partition_id}] 无满足条件的影像，退出")
        return

    # 按日期分组
    groups = group_by_date(items, a.partition_id)

    # 创建临时目录用于处理
    base_temp_dir = tempfile.mkdtemp(prefix=f"s2_proc_{a.partition_id}_", dir=TEMP_DIR)
    
    try:
        # 创建Dask客户端（可能失败，失败时用同步处理）
        dask_client = make_simple_client(a.dask_workers, a.worker_memory, a.partition_id)
        
        # 处理每一天的数据
        results = []
        for i, (d, its) in enumerate(groups.items()):
            # 对小patch，不需要复杂的垃圾回收和重启逻辑
            if i > 0 and not is_small_patch:
                gc.collect()
            
            # 处理当天数据
            try:
                success = process_day_simple(
                    d, its, tpl, bbox_proj, mask_np,
                    out_dir, a.resolution, a.chunksize,
                    a.overwrite, a.min_coverage, is_small_patch, dask_client, a.partition_id
                )
                results.append(success)
            except Exception as day_error:
                logging.error(f"[{a.partition_id}] 处理日期 {d} 时发生异常: {day_error}")
                results.append(False)
        
        # 关闭客户端
        if dask_client:
            try:
                dask_client.close(timeout=30)
            except:
                pass
        
        # 总结统计
        success_count = sum(results)
        total_count = len(results)
        
        logging.info(f"[{a.partition_id}] ✅ 分区处理完成: 成功 {success_count}/{total_count} 天")
        
        # 返回适当的退出码
        if success_count == 0 and total_count > 0:
            sys.exit(1)
        elif success_count < total_count:
            logging.warning(f"[{a.partition_id}] ⚠️  部分日期处理失败 ({total_count - success_count}/{total_count})")
            sys.exit(2)
        else:
            sys.exit(0)
    
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(base_temp_dir)
        except:
            pass

if __name__ == "__main__":
    main()