#!/usr/bin/env python3
"""
TIFF processing script - generate 10 m resolution masks in parallel
Converts each TIFF into a 10 m mask filled with the value 1 and renames it by year
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
import paths

def setup_logging():
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
    """Load the CSV and return a {filename: year} mapping."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading year mapping file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Read CSV file, {len(df)} records")
        
        year_mapping = dict(zip(df['new_fname'], df['year']))
        logger.info(f"Year mapping built, {len(year_mapping)} entries")
        
        return year_mapping
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise

def get_tiff_files(directory):
    """List the TIFF files in a directory, as path strings."""
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning TIFF directory: {directory}")
    
    tiff_files = []
    for ext in ['*.tif', '*.tiff']:
        tiff_files.extend(Path(directory).glob(ext))
    
    logger.info(f"Found {len(tiff_files)} TIFF files")
    return [str(f) for f in tiff_files]

def calculate_10m_transform_and_shape(bounds, original_crs):
    """Transform and shape of a 10 m grid over ``bounds``.

    Returns (transform, width, height).
    """
    left, bottom, right, top = bounds
    
    width = int((right - left) / 10)
    height = int((top - bottom) / 10)
    
    transform = from_bounds(left, bottom, right, top, width, height)
    
    return transform, width, height

def process_single_tiff(args):
    """Process one TIFF.

    ``args`` is (tiff_path, year_mapping, output_dir); returns
    (success, filename, message).
    """
    tiff_path, year_mapping, output_dir = args
    logger = logging.getLogger(__name__)
    
    try:
        filename = os.path.basename(tiff_path)
        logger.debug(f"Processing file: {filename}")
        
        if filename not in year_mapping:
            error_msg = f"File {filename} not found in the year mapping"
            logger.warning(error_msg)
            return False, filename, error_msg
        
        year = year_mapping[filename]
        
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{year}_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        with rasterio.open(tiff_path, 'r') as src:
            original_crs = src.crs
            bounds = src.bounds
            original_nodata = src.nodata
            
            logger.debug(f"Source file - CRS: {original_crs}, bounds: {bounds}")
            
            new_transform, new_width, new_height = calculate_10m_transform_and_shape(bounds, original_crs)
            
            logger.debug(f"New grid - width: {new_width}, height: {new_height}")
            
            output_data = np.ones((new_height, new_width), dtype=np.uint8)
            
            profile = {
                'driver': 'GTiff',
                'dtype': np.uint8,
                'nodata': None,  # mask files do not need a nodata value
                'width': new_width,
                'height': new_height,
                'count': 1,
                'crs': original_crs,
                'transform': new_transform,
                'compress': 'lzw',  # compress to save space
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512
            }
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(output_data, 1)
        
        success_msg = f"Processed {filename} -> {output_filename}"
        logger.debug(success_msg)
        return True, filename, success_msg
        
    except Exception as e:
        error_msg = f"Error processing file {filename}: {str(e)}"
        logger.error(error_msg)
        return False, filename, error_msg

def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting batch TIFF processing")
    logger.info("=" * 60)
    
    base_dir = paths.ACQ_DATA_ROOT
    tiff_dir = os.path.join(base_dir, "train_agbm")
    csv_path = os.path.join(base_dir, "train_agbm_with_year.csv")
    output_dir = os.path.join(base_dir, "train_agbm_masks_10m")
    
    logger.info(f"TIFF directory: {tiff_dir}")
    logger.info(f"CSV path: {csv_path}")
    logger.info(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")
    
    try:
        year_mapping = load_year_mapping(csv_path)
    except Exception as e:
        logger.error(f"Failed to load the year mapping, exiting: {e}")
        return
    
    tiff_files = get_tiff_files(tiff_dir)
    if not tiff_files:
        logger.error("No TIFF files found, exiting")
        return
    
    available_cores = cpu_count()
    use_cores = min(90, available_cores)  # configured cap or the system core count, whichever is smaller
    logger.info(f"Available CPU cores: {available_cores}, using: {use_cores}")
    
    process_args = [(tiff_path, year_mapping, output_dir) for tiff_path in tiff_files]
    
    logger.info(f"Processing {len(tiff_files)} files in parallel...")
    start_time = time.time()
    
    successful_files = []
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=use_cores) as executor:
        future_to_file = {
            executor.submit(process_single_tiff, args): args[0] 
            for args in process_args
        }
        
        with tqdm(total=len(tiff_files), desc="Progress", ncols=100) as pbar:
            for future in as_completed(future_to_file):
                try:
                    success, filename, message = future.result()
                    if success:
                        successful_files.append(filename)
                        pbar.set_postfix({'status': 'success', 'file': filename[:20]})
                    else:
                        failed_files.append((filename, message))
                        pbar.set_postfix({'status': 'failed', 'file': filename[:20]})
                    
                    pbar.update(1)
                    
                except Exception as e:
                    file_path = future_to_file[future]
                    filename = os.path.basename(file_path)
                    error_msg = f"Exception during processing: {str(e)}"
                    failed_files.append((filename, error_msg))
                    logger.error(f"Exception processing file {filename}: {e}")
                    pbar.update(1)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("Processing complete")
    logger.info("=" * 60)
    logger.info(f"Total time: {processing_time:.2f} s")
    logger.info(f"Files processed successfully: {len(successful_files)}")
    logger.info(f"Files failed: {len(failed_files)}")
    logger.info(f"Throughput: {len(tiff_files) / processing_time:.2f} files/s")
    
    if failed_files:
        logger.warning("The following files failed:")
        for filename, error in failed_files:
            logger.warning(f"  - {filename}: {error}")
    
    logger.info(f"All generated mask files saved to: {output_dir}")
    logger.info("Done")

if __name__ == "__main__":
    main()