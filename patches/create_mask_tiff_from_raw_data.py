#!/usr/bin/env python3
"""
TIFF文件处理脚本 - 并行生成10m分辨率mask
将TIFF文件转换为值为1、分辨率为10m的mask，并根据年份重命名
"""

import os
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import numpy as np
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 配置日志
def setup_logging():
    """设置详细的日志配置"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('tiff_processing.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_year_mapping(csv_path):
    """
    加载CSV文件并创建文件名到年份的映射
    
    Args:
        csv_path (str): CSV文件路径
        
    Returns:
        dict: 文件名到年份的映射字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"正在加载年份映射文件: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"成功读取CSV文件，共{len(df)}条记录")
        
        # 创建文件名到年份的映射
        year_mapping = dict(zip(df['new_fname'], df['year']))
        logger.info(f"创建年份映射完成，共{len(year_mapping)}个映射")
        
        return year_mapping
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        raise

def get_tiff_files(directory):
    """
    获取目录下所有TIFF文件
    
    Args:
        directory (str): TIFF文件目录
        
    Returns:
        list: TIFF文件路径列表
    """
    logger = logging.getLogger(__name__)
    logger.info(f"正在扫描TIFF文件目录: {directory}")
    
    tiff_files = []
    for ext in ['*.tif', '*.tiff']:
        tiff_files.extend(Path(directory).glob(ext))
    
    logger.info(f"找到{len(tiff_files)}个TIFF文件")
    return [str(f) for f in tiff_files]

def calculate_10m_transform_and_shape(bounds, original_crs):
    """
    计算10m分辨率的transform和shape
    
    Args:
        bounds: 原始边界 (left, bottom, right, top)
        original_crs: 原始坐标系
        
    Returns:
        tuple: (transform, width, height)
    """
    left, bottom, right, top = bounds
    
    # 计算10m分辨率下的像素数量
    width = int((right - left) / 10)
    height = int((top - bottom) / 10)
    
    # 创建新的transform
    transform = from_bounds(left, bottom, right, top, width, height)
    
    return transform, width, height

def process_single_tiff(args):
    """
    处理单个TIFF文件
    
    Args:
        args (tuple): (tiff_path, year_mapping, output_dir)
        
    Returns:
        tuple: (success, filename, message)
    """
    tiff_path, year_mapping, output_dir = args
    logger = logging.getLogger(__name__)
    
    try:
        filename = os.path.basename(tiff_path)
        logger.debug(f"开始处理文件: {filename}")
        
        # 检查文件是否在年份映射中
        if filename not in year_mapping:
            error_msg = f"文件 {filename} 在年份映射中未找到"
            logger.warning(error_msg)
            return False, filename, error_msg
        
        year = year_mapping[filename]
        
        # 生成输出文件名
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{year}_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 读取原始TIFF文件
        with rasterio.open(tiff_path, 'r') as src:
            # 获取原始属性
            original_crs = src.crs
            bounds = src.bounds
            original_nodata = src.nodata
            
            logger.debug(f"原始文件信息 - CRS: {original_crs}, 边界: {bounds}")
            
            # 计算10m分辨率的transform和尺寸
            new_transform, new_width, new_height = calculate_10m_transform_and_shape(bounds, original_crs)
            
            logger.debug(f"新分辨率信息 - 宽度: {new_width}, 高度: {new_height}")
            
            # 创建输出数组（全部填充为1）
            output_data = np.ones((new_height, new_width), dtype=np.uint8)
            
            # 写入新的TIFF文件
            profile = {
                'driver': 'GTiff',
                'dtype': np.uint8,
                'nodata': None,  # mask文件通常不需要nodata值
                'width': new_width,
                'height': new_height,
                'count': 1,
                'crs': original_crs,
                'transform': new_transform,
                'compress': 'lzw',  # 使用压缩以节省空间
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512
            }
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(output_data, 1)
        
        success_msg = f"成功处理 {filename} -> {output_filename}"
        logger.debug(success_msg)
        return True, filename, success_msg
        
    except Exception as e:
        error_msg = f"处理文件 {filename} 时出错: {str(e)}"
        logger.error(error_msg)
        return False, filename, error_msg

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("开始TIFF文件批量处理")
    logger.info("=" * 60)
    
    # 配置路径
    base_dir = "/scratch/zf281/create_d-pixels_biomassters/data"
    tiff_dir = os.path.join(base_dir, "train_agbm")
    csv_path = os.path.join(base_dir, "train_agbm_with_year.csv")
    output_dir = os.path.join(base_dir, "train_agbm_masks_10m")
    
    logger.info(f"TIFF文件目录: {tiff_dir}")
    logger.info(f"CSV文件路径: {csv_path}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录已创建: {output_dir}")
    
    # 加载年份映射
    try:
        year_mapping = load_year_mapping(csv_path)
    except Exception as e:
        logger.error(f"加载年份映射失败，程序退出: {e}")
        return
    
    # 获取所有TIFF文件
    tiff_files = get_tiff_files(tiff_dir)
    if not tiff_files:
        logger.error("未找到任何TIFF文件，程序退出")
        return
    
    # 检查可用的CPU核心数
    available_cores = cpu_count()
    use_cores = min(90, available_cores)  # 使用指定的96核心或系统可用核心数
    logger.info(f"系统可用CPU核心: {available_cores}, 使用核心数: {use_cores}")
    
    # 准备处理参数
    process_args = [(tiff_path, year_mapping, output_dir) for tiff_path in tiff_files]
    
    # 开始并行处理
    logger.info(f"开始并行处理 {len(tiff_files)} 个文件...")
    start_time = time.time()
    
    successful_files = []
    failed_files = []
    
    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor(max_workers=use_cores) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_tiff, args): args[0] 
            for args in process_args
        }
        
        # 使用tqdm显示进度条
        with tqdm(total=len(tiff_files), desc="处理进度", ncols=100) as pbar:
            for future in as_completed(future_to_file):
                try:
                    success, filename, message = future.result()
                    if success:
                        successful_files.append(filename)
                        pbar.set_postfix({'状态': '成功', '文件': filename[:20]})
                    else:
                        failed_files.append((filename, message))
                        pbar.set_postfix({'状态': '失败', '文件': filename[:20]})
                    
                    pbar.update(1)
                    
                except Exception as e:
                    file_path = future_to_file[future]
                    filename = os.path.basename(file_path)
                    error_msg = f"处理过程中出现异常: {str(e)}"
                    failed_files.append((filename, error_msg))
                    logger.error(f"文件 {filename} 处理异常: {e}")
                    pbar.update(1)
    
    # 处理完成，输出统计信息
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("处理完成！")
    logger.info("=" * 60)
    logger.info(f"总处理时间: {processing_time:.2f}秒")
    logger.info(f"成功处理文件数: {len(successful_files)}")
    logger.info(f"失败文件数: {len(failed_files)}")
    logger.info(f"处理速度: {len(tiff_files) / processing_time:.2f} 文件/秒")
    
    if failed_files:
        logger.warning("以下文件处理失败:")
        for filename, error in failed_files:
            logger.warning(f"  - {filename}: {error}")
    
    logger.info(f"所有生成的mask文件已保存到: {output_dir}")
    logger.info("处理完成！")

if __name__ == "__main__":
    main()