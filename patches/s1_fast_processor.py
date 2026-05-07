#!/usr/bin/env python3
"""
s1_fast_processor.py — Sentinel-1 RTC 快速下载 & ROI 拼接 (灵活并行处理版本)
更新：2025-05-20
支持灵活的并行分区处理，包含完善的错误处理和超时控制
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

# ▶ distributed 版本兼容
try:
    from distributed.comm.core import CommClosedError
except ImportError:
    from distributed import CommClosedError

warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask.core")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

# ─── 常量 ──────────────────────────────────────────────────────────────────────
# Sentinel-1 分辨率 (米)
SAR_RESOLUTION = 10.0

# 有效覆盖率阈值 (低于此值跳过处理)
MIN_VALID_COVERAGE = 10.0  # 百分比

# 超时设置（秒）
PROCESS_TIMEOUT = 120 * 60  # 总体超时
DAY_TIMEOUT = 40 * 60      # 单日处理超时
ITEM_TIMEOUT = 20 * 60      # 单个item处理超时

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
                   help="最小有效像素覆盖率 (百分比)")
    P.add_argument("--partition_id", default="unknown",
                   help="分区ID（用于日志标识）")
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
        out_dir / f"s1_{partition_id}_detail.log", 
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

# ─── Dask ─────────────────────────────────────────────────────────────────────
def make_client(req_workers:int, req_mem:int, partition_id: str):
    """创建Dask客户端，使用分区特定的dashboard端口"""
    total_mem = psutil.virtual_memory().total / 1e9
    workers = min(req_workers, os.cpu_count(),
                  max(1, int(total_mem // (req_mem*1.2))))
    if workers < req_workers:
        logging.warning(f"⚠️  worker 数 {req_workers}→{workers} (资源限制)")
    
    # 为不同分区自动分配dashboard端口
    # 使用分区ID的哈希值来确定端口，确保相同ID总是使用相同端口
    # 将端口限制在8700-8779之间
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

# ─── ROI & 掩膜 ────────────────────────────────────────────────────────────────
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

# ─── STAC ─────────────────────────────────────────────────────────────────────
def search_items(bbox_ll, date_range:str, orbit_state="both", partition_id="unknown"):
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)
    
    query = {"collections": ["sentinel-1-rtc"], "bbox": bbox_ll, "datetime": date_range}
    
    # 如果指定了轨道方向，添加过滤条件
    if orbit_state != "both":
        query["query"] = {"sat:orbit_state": {"eq": orbit_state}}
    
    q = cat.search(**query)
    items = list(q.get_items())
    logging.info(f"[{partition_id}] STAC 命中 {len(items)} item")
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
    logging.info(f"[{partition_id}] ⇒ {len(g)} 观测日-轨道组合")
    return dict(sorted(g.items()))

# ─── 振幅转dB ──────────────────────────────────────────────────────────────────
def amplitude_to_db(amp, mask=None):
    """
    振幅转换为dB值 (带偏移和缩放以适配int16存储)
    
    转换公式:
    dB = 20 * log10(amp)
    shifted = dB + 50     # 偏移避免负值
    scaled = shifted * 200 # 缩放保留精度
    clipped = np.clip(scaled, 0, 30000) # 裁剪到int16范围内
    """
    # 确保amp是numpy数组
    if hasattr(amp, 'values'):
        amp_array = amp.values
    elif hasattr(amp, 'compute'):
        amp_array = amp.compute()
    else:
        amp_array = np.asarray(amp)
    
    # 创建输出数组
    output = np.zeros_like(amp_array, dtype=np.int16)
    
    # 安全处理可能的无效值
    with np.errstate(invalid='ignore', divide='ignore'):
        # 确保amp是有限的(非NaN或inf)
        amp_finite = np.isfinite(amp_array)
        # 创建有效值掩膜 (> 0)
        valid_mask = amp_finite & (amp_array > 0)
    
    # 仅处理有效像素
    if np.any(valid_mask):
        # 仅计算有效像素的dB值
        with np.errstate(invalid='ignore', divide='ignore'):
            # 直接对有效位置进行计算，避免布尔索引问题
            valid_indices = np.where(valid_mask)
            valid_amp = amp_array[valid_indices]
            
            db = 20.0 * np.log10(valid_amp)
            db_shift = db + 50.0
            scaled = db_shift * 200.0
            clipped = np.clip(scaled, 0, 32767)  # 截断到int16范围
        
        # 赋值到输出数组
        output[valid_indices] = clipped.astype(np.int16)
    
    # 应用外部掩膜
    if mask is not None:
        # 确保mask是numpy数组
        if hasattr(mask, 'values'):
            mask_array = mask.values
        else:
            mask_array = np.asarray(mask)
        
        # 处理形状不匹配的情况
        if output.shape != mask_array.shape:
            # 计算共同区域
            common_shape = tuple(min(output.shape[i], mask_array.shape[i]) for i in range(len(output.shape)))
            
            # 创建裁剪后的数组
            if len(common_shape) == 2:
                output_cropped = output[:common_shape[0], :common_shape[1]]
                mask_cropped = mask_array[:common_shape[0], :common_shape[1]]
                output[:common_shape[0], :common_shape[1]] = np.where(mask_cropped > 0, output_cropped, 0)
                # 清零超出mask范围的区域
                if output.shape[0] > common_shape[0]:
                    output[common_shape[0]:, :] = 0
                if output.shape[1] > common_shape[1]:
                    output[:, common_shape[1]:] = 0
            else:
                # 如果不是2D，使用简单的裁剪
                output = np.where(mask_array > 0, output, 0)
        else:
            output = np.where(mask_array > 0, output, 0)
        
    return output

# ─── GeoTIFF 写出 ──────────────────────────────────────────────────────────────
def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    # 处理NaN值
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
        
        # 设置波段描述和元数据
        if metadata:
            dst.set_band_description(1, metadata.get("band_desc", ""))
            dst.update_tags(**metadata)

# ─── 验证TIFF ─────────────────────────────────────────────────────────────────
def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    """验证TIFF文件是否具有预期的属性"""
    try:
        with rasterio.open(file_path) as src:
            # 检查基本属性
            if src.shape != expected_shape:
                logging.warning(f"验证失败: {file_path} 形状不匹配. 预期 {expected_shape}, 得到 {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logging.warning(f"验证失败: {file_path} CRS不匹配. 预期 {expected_crs}, 得到 {src.crs}")
                return False
            
            # 检查转换矩阵
            if not np.allclose(np.array(src.transform)[:6], np.array(expected_transform)[:6], rtol=1e-05, atol=1e-08):
                logging.warning(f"验证失败: {file_path} 转换矩阵不匹配.")
                return False
            
            # 检查数据存在性
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"验证失败: {file_path} 波段全为零")
                return False
            
            # 检查文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            expected_size_mb = (src.width * src.height * src.count * 2) / (1024 * 1024)  # int16 = 2 bytes
            if file_size_mb < expected_size_mb * 0.05:  # 考虑压缩率，但不能太小
                logging.warning(f"验证失败: {file_path} 文件太小. 预期 ~{expected_size_mb:.2f}MB, 得到 {file_size_mb:.2f}MB")
                return False
            
            logging.debug(f"TIFF验证通过: {file_path}，形状={src.shape}, 大小={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logging.error(f"验证TIFF {file_path} 时出错: {e}")
        return False

# ─── 覆盖率分析 ────────────────────────────────────────────────────────────────
def analyze_coverage(data_arr, roi_mask, partition_id: str):
    """分析数据覆盖ROI的情况，处理形状不匹配的情况"""
    # 确保data_arr是numpy数组
    if hasattr(data_arr, 'values'):
        data_values = data_arr.values
    elif hasattr(data_arr, 'compute'):
        data_values = data_arr.compute()
    else:
        data_values = np.asarray(data_arr)
    
    # 创建有效值掩膜 (非零且有限)，抑制无效值警告
    with np.errstate(invalid='ignore'):
        valid_mask = (data_values > 0) & np.isfinite(data_values)
    
    # 检查形状是否匹配，如不匹配则调整
    if len(valid_mask.shape) == 2 and valid_mask.shape != roi_mask.shape:
        logging.info(f"[{partition_id}]     数据形状 {valid_mask.shape} 与ROI形状 {roi_mask.shape} 不匹配，裁剪到共同区域")
        # 计算共同的高度和宽度
        common_height = min(valid_mask.shape[0], roi_mask.shape[0])
        common_width = min(valid_mask.shape[1], roi_mask.shape[1])
        
        # 裁剪两个数组到共同区域
        valid_mask_cropped = valid_mask[:common_height, :common_width]
        roi_mask_cropped = roi_mask[:common_height, :common_width]
        
        # 使用裁剪后的掩膜继续分析，确保返回数值
        valid_count = int(np.sum(valid_mask_cropped & roi_mask_cropped))
        roi_count = int(np.sum(roi_mask_cropped))
        valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
        logging.info(f"[{partition_id}]     单个Tile: ROI内有效像素 {valid_count}/{roi_count} ({valid_pct:.2f}%)")
        
        return valid_mask, valid_pct
    
    # 处理3D数组
    elif len(valid_mask.shape) == 3:
        # 多个tile情况
        n_tiles = valid_mask.shape[0]
        tile_stats = []
        
        for i in range(n_tiles):
            # 检查当前tile的形状是否匹配ROI
            if valid_mask[i].shape != roi_mask.shape:
                logging.info(f"[{partition_id}]     Tile {i} 形状 {valid_mask[i].shape} 与ROI形状 {roi_mask.shape} 不匹配，裁剪到共同区域")
                # 计算共同区域
                common_height = min(valid_mask[i].shape[0], roi_mask.shape[0])
                common_width = min(valid_mask[i].shape[1], roi_mask.shape[1])
                
                # 裁剪当前tile和ROI掩膜
                tile_valid = valid_mask[i][:common_height, :common_width]
                roi_cropped = roi_mask[:common_height, :common_width]
            else:
                tile_valid = valid_mask[i]
                roi_cropped = roi_mask
            
            # 计算统计数据，确保返回数值
            valid_count = int(np.sum(tile_valid & roi_cropped))
            roi_count = int(np.sum(roi_cropped))
            valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
            logging.info(f"[{partition_id}]     Tile {i}: ROI内有效像素 {valid_count}/{roi_count} ({valid_pct:.2f}%)")
            tile_stats.append(valid_pct)
        
        # 取最大覆盖率作为总覆盖率（简化处理）
        if tile_stats:
            total_valid_pct = max(tile_stats)
            logging.info(f"[{partition_id}]     合并后: ROI内有效像素覆盖率 {total_valid_pct:.2f}%")
        else:
            total_valid_pct = 0
            logging.info(f"[{partition_id}]     无有效覆盖")
        
        return valid_mask, total_valid_pct
    
    else:
        # 单tile情况，形状匹配
        tile_valid = valid_mask & roi_mask
        valid_count = int(np.sum(tile_valid))
        roi_count = int(np.sum(roi_mask))
        valid_pct = 100 * valid_count / roi_count if roi_count > 0 else 0
        logging.info(f"[{partition_id}]     单个Tile: ROI内有效像素 {valid_count}/{roi_count} ({valid_pct:.2f}%)")
        
        return valid_mask, valid_pct

# ─── 处理单个item ────────────────────────────────────────────────────────────
def process_item(item, tpl, bbox_proj, mask_np, resolution, chunksize, temp_dir, min_coverage, partition_id, retries=2):
    """处理单个Sentinel-1 item并保存为VV/VH两个TIFF文件，包含重试机制
    
    Returns:
        tuple: (vv_path, vh_path, status) 
        status: "success", "skipped", "failed"
    """
    orbit_state = item.properties.get("sat:orbit_state", "unknown")
    date_str = item.properties.get("datetime").split("T")[0]
    item_id = item.id
    
    # 确保临时目录是Path对象
    temp_dir = Path(temp_dir)
    
    # 生成唯一的临时文件名
    uid = uuid.uuid4().hex[:8]
    vv_temp = temp_dir / f"{date_str}_vv_{orbit_state}_{uid}.tiff"
    vh_temp = temp_dir / f"{date_str}_vh_{orbit_state}_{uid}.tiff"
    
    logging.info(f"[{partition_id}]   处理item {item_id} ({date_str}_{orbit_state})")
    
    for attempt in range(retries + 1):
        try:
            # 使用超时控制处理单个item
            with timeout_handler(ITEM_TIMEOUT):
                # 使用stackstac加载数据
                ds = stackstac.stack(
                    [item], 
                    bounds=bbox_proj,
                    epsg=tpl["crs"].to_epsg(),
                    resolution=resolution,
                    chunksize=chunksize
                )
                
                # 检查波段是否存在
                if 'vv' not in ds.band.values or 'vh' not in ds.band.values:
                    logging.warning(f"[{partition_id}]   {date_str}_{orbit_state} 缺少必需的波段，跳过")
                    return None, None, "skipped"
                    
                # 提取VV和VH数据
                vv_data = ds.sel(band="vv").squeeze()
                vh_data = ds.sel(band="vh").squeeze()
                
                # 计算数据（触发实际的数据加载）
                try:
                    # 显式计算数据，这会触发实际的远程读取
                    vv_values = vv_data.compute()
                    vh_values = vh_data.compute()
                except Exception as compute_error:
                    if attempt < retries:
                        logging.warning(f"[{partition_id}]   尝试 {attempt+1}/{retries+1} 计算数据失败: {compute_error}，重试...")
                        time.sleep(2)  # 短暂等待后重试
                        continue
                    else:
                        raise
                
                # 获取数据的实际尺寸
                vv_shape = vv_values.shape
                logging.debug(f"[{partition_id}]   VV数据形状: {vv_shape}, ROI形状: {mask_np.shape}")
                
                # 分析覆盖率
                logging.info(f"[{partition_id}]   分析VV波段覆盖率")
                vv_valid_mask, vv_valid_pct = analyze_coverage(vv_values, mask_np, partition_id)
                
                logging.info(f"[{partition_id}]   分析VH波段覆盖率")
                vh_valid_mask, vh_valid_pct = analyze_coverage(vh_values, mask_np, partition_id)
                
                # 检查覆盖率是否达到阈值
                if vv_valid_pct < min_coverage and vh_valid_pct < min_coverage:
                    logging.warning(f"[{partition_id}]   ⚠️ {date_str}_{orbit_state} 有效覆盖率 VV={vv_valid_pct:.2f}%, VH={vh_valid_pct:.2f}% 均低于 {min_coverage}%，跳过")
                    return None, None, "skipped"
                
                # 处理VV数据: 振幅转dB并应用ROI掩膜
                if vv_valid_pct >= min_coverage:
                    logging.info(f"[{partition_id}]   处理VV波段")
                    
                    # 确保掩膜形状匹配
                    common_height = min(vv_values.shape[0], mask_np.shape[0])
                    common_width = min(vv_values.shape[1], mask_np.shape[1])
                    
                    # 裁剪数据和掩膜
                    vv_cropped = vv_values[:common_height, :common_width]
                    mask_cropped = mask_np[:common_height, :common_width]
                    
                    # 应用振幅转dB处理
                    vv_db = amplitude_to_db(vv_cropped, mask=mask_cropped)
                    
                    # 准备完整尺寸的输出数组
                    vv_final = np.zeros((tpl["height"], tpl["width"]), dtype=np.int16)
                    vv_final[:common_height, :common_width] = vv_db
                    
                    # 写出VV TIFF
                    vv_metadata = {
                        "band_desc": "VV polarization, amplitude to dB, +50 offset, scale=200",
                        "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                        "ORBIT_STATE": orbit_state,
                        "DATE_ACQUIRED": date_str,
                        "POLARIZATION": "VV",
                        "DESCRIPTION": "Sentinel-1 SAR data (VV). Values are amplitude converted to dB, shifted by +50, scaled by 200."
                    }
                    write_tiff(vv_final, vv_temp, tpl, "int16", vv_metadata)
                    
                    # 验证VV输出
                    if not validate_tiff(vv_temp, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        logging.error(f"[{partition_id}]   ✗ VV输出验证失败")
                        if vv_temp.exists():
                            vv_temp.unlink()
                        vv_temp = None
                    else:
                        logging.info(f"[{partition_id}]   ✓ VV: {os.path.getsize(vv_temp)/1e6:.2f} MB")
                else:
                    logging.warning(f"[{partition_id}]   ⚠️ VV覆盖率不足，跳过")
                    vv_temp = None
                
                # 处理VH数据: 振幅转dB并应用ROI掩膜
                if vh_valid_pct >= min_coverage:
                    logging.info(f"[{partition_id}]   处理VH波段")
                    
                    # 确保掩膜形状匹配
                    common_height = min(vh_values.shape[0], mask_np.shape[0])
                    common_width = min(vh_values.shape[1], mask_np.shape[1])
                    
                    # 裁剪数据和掩膜
                    vh_cropped = vh_values[:common_height, :common_width]
                    mask_cropped = mask_np[:common_height, :common_width]
                    
                    # 应用振幅转dB处理
                    vh_db = amplitude_to_db(vh_cropped, mask=mask_cropped)
                    
                    # 准备完整尺寸的输出数组
                    vh_final = np.zeros((tpl["height"], tpl["width"]), dtype=np.int16)
                    vh_final[:common_height, :common_width] = vh_db
                    
                    # 写出VH TIFF
                    vh_metadata = {
                        "band_desc": "VH polarization, amplitude to dB, +50 offset, scale=200",
                        "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                        "ORBIT_STATE": orbit_state,
                        "DATE_ACQUIRED": date_str,
                        "POLARIZATION": "VH",
                        "DESCRIPTION": "Sentinel-1 SAR data (VH). Values are amplitude converted to dB, shifted by +50, scaled by 200."
                    }
                    write_tiff(vh_final, vh_temp, tpl, "int16", vh_metadata)
                    
                    # 验证VH输出
                    if not validate_tiff(vh_temp, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        logging.error(f"[{partition_id}]   ✗ VH输出验证失败")
                        if vh_temp.exists():
                            vh_temp.unlink()
                        vh_temp = None
                    else:
                        logging.info(f"[{partition_id}]   ✓ VH: {os.path.getsize(vh_temp)/1e6:.2f} MB")
                else:
                    logging.warning(f"[{partition_id}]   ⚠️ VH覆盖率不足，跳过")
                    vh_temp = None
                
                # 检查最终状态
                if vv_temp or vh_temp:
                    # 成功完成，跳出重试循环
                    return vv_temp, vh_temp, "success"
                else:
                    # 没有生成任何文件，但这是覆盖率不足导致的，算作跳过
                    return None, None, "skipped"
                
        except TimeoutException as e:
            if attempt < retries:
                logging.warning(f"[{partition_id}]   尝试 {attempt+1}/{retries+1} 处理 {date_str}_{orbit_state} 超时，重试...")
                time.sleep(5)  # 等待5秒后重试
                continue
            else:
                logging.error(f"[{partition_id}]   ⚠️ 处理 {date_str}_{orbit_state} 超时: {e}")
                return None, None, "failed"
        except (RuntimeError, Exception) as e:
            # 检查是否是网络/IO错误
            error_msg = str(e).lower()
            is_retriable_error = any(keyword in error_msg for keyword in [
                'rasterio', 'read', 'tiff', 'network', 'timeout', 'connection', 'io'
            ])
            
            if is_retriable_error and attempt < retries:
                logging.warning(f"[{partition_id}]   尝试 {attempt+1}/{retries+1} 处理 {date_str}_{orbit_state} 出错: {type(e).__name__} - {e}，重试...")
                time.sleep(3)  # 等待3秒后重试
                continue
            else:
                logging.error(f"[{partition_id}]   ✗ 处理 {date_str}_{orbit_state} 出错: {type(e).__name__} - {e}")
                return None, None, "failed"
    
    # 如果所有重试都失败了
    logging.error(f"[{partition_id}]   ✗ 处理 {date_str}_{orbit_state} 全部重试失败")
    return None, None, "failed"

# ─── 镶嵌多个TIFF ───────────────────────────────────────────────────────────
def mosaic_tiffs(tiff_paths, output_path, tpl, date_str, orbit_state, polarization, partition_id):
    """镶嵌多个TIFF为一个输出TIFF"""
    try:
        # 确保输出路径是Path对象
        output_path = Path(output_path)
        
        # 打开所有源TIFF
        src_files = []
        for path in tiff_paths:
            if path and os.path.exists(path):
                try:
                    src = rasterio.open(path)
                    src_files.append(src)
                except Exception as e:
                    logging.warning(f"[{partition_id}]   打开 {path} 用于镶嵌失败: {e}")
        
        if not src_files:
            logging.warning(f"[{partition_id}]   没有有效文件用于镶嵌 {date_str}_{polarization}_{orbit_state}")
            return None
        
        # 执行镶嵌操作
        logging.info(f"[{partition_id}]   镶嵌 {len(src_files)} 个 {polarization} 文件 ({date_str}_{orbit_state})")
        mosaic_data, out_transform = merge(src_files, nodata=0)
        
        # 关闭所有源文件
        for src in src_files:
            src.close()
        
        # 检查镶嵌数据结构
        if mosaic_data.shape[0] < 1:
            logging.error(f"[{partition_id}]   镶嵌数据结构不正确 ({date_str}_{polarization}_{orbit_state})")
            return None
        
        # 创建元数据
        metadata = {
            "band_desc": f"{polarization} polarization, amplitude to dB, +50 offset, scale=200",
            "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "ORBIT_STATE": orbit_state,
            "DATE_ACQUIRED": date_str,
            "POLARIZATION": polarization,
            "MOSAIC_SOURCE_COUNT": len(src_files),
            "DESCRIPTION": f"Mosaicked Sentinel-1 SAR data ({polarization}). Values are amplitude converted to dB, shifted by +50, scaled by 200."
        }
        
        # 写出镶嵌TIFF
        write_tiff(mosaic_data[0], output_path, tpl, "int16", metadata)
        
        # 验证输出文件
        if not validate_tiff(output_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.error(f"[{partition_id}]   ✗ 镶嵌TIFF验证失败 ({date_str}_{polarization}_{orbit_state})")
            if Path(output_path).exists():
                Path(output_path).unlink()
            return None
        
        # 记录成功完成和文件大小
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"[{partition_id}]   ✓ 成功创建镶嵌 {output_path} ({file_size_mb:.2f} MB)")
        
        return output_path
    
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ 创建镶嵌 {date_str}_{polarization}_{orbit_state} 时出错: {e}")
        return None

# ─── 处理单日观测 ───────────────────────────────────────────────────────────
def process_day_orbit(key, items, tpl, bbox_proj, mask_np, out_dir, resolution, chunksize, min_coverage, partition_id, overwrite=False):
    """处理同一日期和轨道状态的所有items，包含20分钟超时控制"""
    date_str, orbit_state = key.split("_")
    logging.info(f"[{partition_id}] → {key} (item={len(items)})")
    t0 = time.time()
    
    try:
        # 使用20分钟超时控制单日处理
        with timeout_handler(DAY_TIMEOUT):
            # 确保输出目录是Path对象
            out_dir = Path(out_dir)
            
            # 创建输出目录
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查输出文件是否已存在 (直接输出到目标目录，不创建子文件夹)
            vv_out = out_dir / f"{date_str}_vv_{orbit_state}.tiff"
            vh_out = out_dir / f"{date_str}_vh_{orbit_state}.tiff"
            
            # 如果文件已存在且不覆盖，则跳过
            if not overwrite and vv_out.exists() and vh_out.exists():
                # 验证现有文件
                vv_valid = validate_tiff(vv_out, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"])
                vh_valid = validate_tiff(vh_out, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"])
                
                if vv_valid and vh_valid:
                    logging.info(f"[{partition_id}]   已存在有效文件，跳过")
                    return True
                else:
                    logging.warning(f"[{partition_id}]   文件存在但验证失败，重新处理")
                    # 删除无效文件
                    if not vv_valid and vv_out.exists():
                        vv_out.unlink()
                    if not vh_valid and vh_out.exists():
                        vh_out.unlink()
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix=f"s1_{date_str}_{orbit_state}_")
            logging.debug(f"[{partition_id}]   临时目录: {temp_dir}")
            
            try:
                # 处理每个item，允许部分失败
                vv_temp_files = []
                vh_temp_files = []
                processed_count = 0  # 成功处理的数量
                failed_count = 0     # 真正失败的数量
                skipped_count = 0    # 跳过的数量
                
                for i, item in enumerate(items):
                    item_start_time = time.time()
                    logging.info(f"[{partition_id}]   处理item {i+1}/{len(items)}")
                    
                    vv_path, vh_path, status = process_item(item, tpl, bbox_proj, mask_np, resolution, chunksize, temp_dir, min_coverage, partition_id)
                    
                    if status == "success":
                        processed_count += 1
                        if vv_path:
                            vv_temp_files.append(str(vv_path))
                        if vh_path:
                            vh_temp_files.append(str(vh_path))
                        
                        item_duration = time.time() - item_start_time
                        logging.info(f"[{partition_id}]   item {i+1} 处理成功，用时 {item_duration:.1f}s")
                    elif status == "skipped":
                        skipped_count += 1
                        item_duration = time.time() - item_start_time
                        logging.info(f"[{partition_id}]   item {i+1} 跳过（覆盖率不足或缺少波段），用时 {item_duration:.1f}s")
                    else:  # status == "failed"
                        failed_count += 1
                        item_duration = time.time() - item_start_time
                        logging.warning(f"[{partition_id}]   item {i+1} 处理失败，用时 {item_duration:.1f}s")
                
                # 记录处理统计
                logging.info(f"[{partition_id}]   items处理统计: 成功 {processed_count}, 跳过 {skipped_count}, 失败 {failed_count} (总共 {len(items)})")
                
                # 如果没有有效文件，根据原因给出不同的消息
                if not vv_temp_files and not vh_temp_files:
                    if processed_count == 0 and skipped_count > 0:
                        logging.info(f"[{partition_id}]   {key} 所有items均因覆盖率不足或缺少波段而跳过")
                        return True  # 跳过不算失败
                    else:
                        logging.warning(f"[{partition_id}]   没有为 {key} 生成任何有效文件")
                        return False
                
                # 处理VV文件
                vv_success = False
                if vv_temp_files:
                    # 如果只有一个VV文件，直接使用
                    if len(vv_temp_files) == 1:
                        logging.info(f"[{partition_id}]   只有一个有效的VV文件，直接使用")
                        shutil.copy2(vv_temp_files[0], vv_out)
                        vv_success = True
                    # 多个VV文件需要镶嵌
                    else:
                        vv_mosaic = mosaic_tiffs(vv_temp_files, vv_out, tpl, date_str, orbit_state, "VV", partition_id)
                        vv_success = vv_mosaic is not None
                
                # 处理VH文件
                vh_success = False
                if vh_temp_files:
                    # 如果只有一个VH文件，直接使用
                    if len(vh_temp_files) == 1:
                        logging.info(f"[{partition_id}]   只有一个有效的VH文件，直接使用")
                        shutil.copy2(vh_temp_files[0], vh_out)
                        vh_success = True
                    # 多个VH文件需要镶嵌
                    else:
                        vh_mosaic = mosaic_tiffs(vh_temp_files, vh_out, tpl, date_str, orbit_state, "VH", partition_id)
                        vh_success = vh_mosaic is not None
                
                # 输出处理结果
                total_duration = time.time() - t0
                if vv_success and vh_success:
                    logging.info(f"[{partition_id}] ← {key} 成功处理VV和VH，用时 {total_duration:.1f}s")
                    return True
                elif vv_success:
                    logging.info(f"[{partition_id}] ← {key} 只成功处理VV，用时 {total_duration:.1f}s")
                    return True
                elif vh_success:
                    logging.info(f"[{partition_id}] ← {key} 只成功处理VH，用时 {total_duration:.1f}s")
                    return True
                else:
                    logging.error(f"[{partition_id}] ← {key} 处理失败，用时 {total_duration:.1f}s")
                    return False
                    
            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(temp_dir)
                    logging.debug(f"[{partition_id}]   已清理临时目录: {temp_dir}")
                except Exception as e:
                    logging.warning(f"[{partition_id}]   清理临时目录失败: {e}")
                    
    except TimeoutException as e:
        total_duration = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {key} 处理超时 ({total_duration:.1f}s > {DAY_TIMEOUT}s): {e}")
        return False
    except Exception as e:
        total_duration = time.time() - t0
        logging.error(f"[{partition_id}] ✗ 处理 {key} 出错 (用时 {total_duration:.1f}s): {type(e).__name__} - {e}")
        return False

# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.debug, out_dir, args.partition_id)
    logging.info(f"[{args.partition_id}] ⚡ S1 Fast Processor 启动 (灵活并行版本)"); 
    log_sys(args.partition_id)
    logging.info(f"[{args.partition_id}] 处理超时设置: 总体{PROCESS_TIMEOUT//60}分钟, 单日{DAY_TIMEOUT//60}分钟, 单item{ITEM_TIMEOUT//60}分钟")
    logging.info(f"[{args.partition_id}] 处理时间段: {args.start_date} → {args.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np = load_roi(Path(args.input_tiff), args.partition_id)
    
    # 基于轨道状态搜索items
    if args.orbit_state == "both":
        logging.info(f"[{args.partition_id}] 搜索上升和下降轨道数据")
        items = search_items(bbox_ll, f"{args.start_date}/{args.end_date}", partition_id=args.partition_id)
    else:
        logging.info(f"[{args.partition_id}] 搜索 {args.orbit_state} 轨道数据")
        items = search_items(bbox_ll, f"{args.start_date}/{args.end_date}", args.orbit_state, args.partition_id)
    
    if not items:
        logging.warning(f"[{args.partition_id}] 无满足条件的影像，退出")
        return

    # 按日期和轨道状态分组
    groups = group_by_date_orbit(items, args.partition_id)

    with make_client(args.dask_workers, args.worker_memory, args.partition_id):
        report_path = out_dir / f"dask-report-{args.partition_id}.html"
        with performance_report(filename=report_path):
            # 创建线程池处理多个日期
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                # 提交所有任务
                future_to_group = {}
                for key, group_items in groups.items():
                    future = executor.submit(
                        process_day_orbit, key, group_items, tpl, bbox_proj, mask_np,
                        str(out_dir), SAR_RESOLUTION, args.chunksize, args.min_coverage,
                        args.partition_id, args.overwrite
                    )
                    future_to_group[future] = key
                
                # 处理结果
                results = []
                for future in concurrent.futures.as_completed(future_to_group):
                    key = future_to_group[future]
                    try:
                        success = future.result()
                        results.append(success)
                    except Exception as e:
                        logging.error(f"[{args.partition_id}] 处理 {key} 时发生异常: {e}")
                        results.append(False)
    
    success_count = sum(results)
    total_count = len(results)
    
    logging.info(f"[{args.partition_id}] ✅ 分区处理完成: 成功 {success_count}/{total_count} 天")
    logging.info(f"[{args.partition_id}] 📊 Dask 性能报告已保存: {report_path}")
    
    # 返回适当的退出码
    if success_count == 0:
        sys.exit(1)  # 全部失败
    elif success_count < total_count:
        logging.warning(f"[{args.partition_id}] ⚠️  部分日期处理失败 ({total_count - success_count}/{total_count})")
        sys.exit(2)  # 部分失败
    else:
        sys.exit(0)  # 全部成功

if __name__ == "__main__":
    main()