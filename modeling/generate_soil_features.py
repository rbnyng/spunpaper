import pandas as pd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path
from tqdm import tqdm
import logging
import time
from owslib.wcs import WebCoverageService
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- CONFIGURATION ---
BIODIVERSITY_CSV_PATHS = [
    "/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv",
    "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"
]
SOIL_CACHE_DIR = Path("/scratch/ray25/soil_features_cache")
SOIL_VARS_TO_FETCH = {
    # --- Physical and Chemical ---
    'phh2o':    ['phh2o_0-5cm_mean', 'phh2o_5-15cm_mean'],
    'soc':      ['soc_0-5cm_mean', 'soc_5-15cm_mean'],
    'nitrogen': ['nitrogen_0-5cm_mean', 'nitrogen_5-15cm_mean'],
    'clay':     ['clay_0-5cm_mean', 'clay_5-15cm_mean'],
    'silt':     ['silt_0-5cm_mean', 'silt_5-15cm_mean'],
    'sand':     ['sand_0-5cm_mean', 'sand_5-15cm_mean'],
    'cec':      ['cec_0-5cm_mean', 'cec_5-15cm_mean'],
    'bdod':     ['bdod_0-5cm_mean', 'bdod_5-15cm_mean'],

    # --- Water Availability ---
    # Water at Field Capacity (how much it holds)
    'wv0033':   ['wv0033_0-5cm_mean', 'wv0033_5-15cm_mean'],
    # Water at Permanent Wilting Point (how much is unavailable)
    'wv1500':   ['wv1500_0-5cm_mean', 'wv1500_5-15cm_mean'],
    
    # --- Soil Classification (Categorical) ---
    # This layer has no depth, it's a single classification for the profile.
    'wrb':      ['MostProbable'] 
}

# Define the CRS we will use for all requests, as per your working example
CRS_WGS84_URI = 'http://www.opengis.net/def/crs/EPSG/0/4326'

# Number of parallel threads to use
MAX_WORKERS = 16

def _fetch_single_layer(wcs, coverage_id, subsets, lon, lat):
    """
    Helper function to fetch data for a single coverage_id.
    This is the function that each thread will execute.
    """
    try:
        response = wcs.getCoverage(
            identifier=[coverage_id],
            subsets=subsets,
            crs=CRS_WGS84_URI,
            subsettingCrs=CRS_WGS84_URI,
            format='image/tiff'
        )
        with MemoryFile(response.read()) as memfile:
            with memfile.open() as dataset:
                row, col = dataset.index(lon, lat)
                nodata_val = dataset.nodatavals[0]
                raw_val = dataset.read(1)[row, col]
                if raw_val != nodata_val:
                    return coverage_id, float(raw_val)
    except Exception as e:
        # If any error occurs for this layer, log it and return None
        logging.debug(f"Sub-thread failed for {coverage_id} at ({lat},{lon}): {e}")
    
    return coverage_id, np.nan # Return NaN on failure
    
def fetch_soil_features_for_point(
    lat: float, 
    lon: float, 
    wcs_services: dict
    ) -> dict:
    features = {}
    bbox_size = 0.05
    half_size = bbox_size / 2.0
    subsets = [('X', lon - half_size, lon + half_size), ('Y', lat - half_size, lat + half_size)]

    layer_results = {}
    # Use a ThreadPoolExecutor to run all layer requests concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary to map a running "future" back to its coverage ID
        future_to_cid = {}
        for var_name, coverage_ids in SOIL_VARS_TO_FETCH.items():
            for cid in coverage_ids:
                wcs = wcs_services[var_name]
                # Submit the job to the pool and store the future
                future = executor.submit(_fetch_single_layer, wcs, cid, subsets, lon, lat)
                future_to_cid[future] = cid

        # As each future completes, process its result
        for future in as_completed(future_to_cid):
            cid = future_to_cid[future]
            try:
                # result() will re-raise exceptions from the thread, or return the value
                _, pixel_val = future.result()
                layer_results[cid] = pixel_val
            except Exception as e:
                logging.warning(f"A fetch job failed for {cid} with error: {e}")
                layer_results[cid] = np.nan
    
    # --- Now, process the collected results sequentially (this part is super fast) ---
    for var_name, coverage_ids in SOIL_VARS_TO_FETCH.items():
        if len(coverage_ids) == 1:
            cid = coverage_ids[0]
            pixel_val = layer_results.get(cid, np.nan)
            features[f"soil_{var_name}"] = pixel_val
        else:
            depth_values = []
            for cid in coverage_ids:
                pixel_val = layer_results.get(cid, np.nan)
                scaled_val = pixel_val
                if not np.isnan(pixel_val):
                    if var_name == 'nitrogen': scaled_val /= 100.0
                    elif var_name == 'phh2o': scaled_val /= 10.0
                    elif var_name in ['soc', 'clay', 'silt', 'sand', 'cec', 'bdod', 'wv0033', 'wv1500']: scaled_val /= 10.0
                depth_values.append(scaled_val)
                depth_str = cid.split('_')[1].replace('-', '_')
                features[f"soil_{var_name}_{depth_str}"] = scaled_val
            
            if len(depth_values) == 2 and not np.isnan(depth_values[0]) and not np.isnan(depth_values[1]):
                features[f"soil_{var_name}_gradient"] = depth_values[0] - depth_values[1]
            else:
                features[f"soil_{var_name}_gradient"] = np.nan

    return features
    
def main():
    logging.info("Starting SoilGrids feature pre-processing (Small Box Method).")
    SOIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df_list = [pd.read_csv(p) for p in BIODIVERSITY_CSV_PATHS]
    biodiversity_df = pd.concat(df_list, ignore_index=True)
    biodiversity_df.dropna(subset=['latitude', 'longitude', 'sample_id'], inplace=True)
    
    # --- Setup: Connect to each WCS service once for efficiency ---
    logging.info("Initializing WCS service connections...")
    wcs_services = {}
    #unique_map_names = set(cid.split('_')[0] for v in SOIL_VARS_TO_FETCH.values() for cid in v)
    unique_map_names = set(SOIL_VARS_TO_FETCH.keys())
    for map_name in tqdm(unique_map_names, desc="Connecting to services"):
        service_url = f'https://maps.isric.org/mapserv?map=/map/{map_name}.map'
        wcs_services[map_name] = WebCoverageService(service_url, version='2.0.1')
        time.sleep(0.01) # Be polite
    
    # --- Main processing loop ---
    for _, row in tqdm(biodiversity_df.iterrows(), total=len(biodiversity_df), desc="Processing points"):
        sample_id, lat, lon = row['sample_id'], row['latitude'], row['longitude']
        
        cache_file = SOIL_CACHE_DIR / f"{sample_id}.npy"
        if cache_file.exists():
            continue

        soil_features = fetch_soil_features_for_point(lat, lon, wcs_services)
        
        if soil_features:
            np.save(cache_file, soil_features)

    logging.info("SoilGrids feature pre-processing complete.")
    logging.info(f"Cached data saved in: {SOIL_CACHE_DIR}")

if __name__ == "__main__":
    main()