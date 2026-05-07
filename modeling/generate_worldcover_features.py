import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import rasterio
from pathlib import Path
from tqdm import tqdm
import logging
import time
from shapely.geometry import Point

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- CONFIGURATION ---

# Point to the same CSV files your main training script uses
BIODIVERSITY_CSV_PATHS = [
    "/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv",
    "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"
]

# A directory to store the downloaded WorldCover GeoTIFF tiles
WORLDCOVER_TILES_DIR = Path("/scratch/ray25/worldcover_tiles")

# The final output file: a clean CSV mapping sample IDs to land cover classes
OUTPUT_CSV_PATH = Path("/scratch/ray25/worldcover_features.csv")

# ESA WorldCover specific settings (2021 is recommended)
YEAR = 2021
VERSION = {2020: 'v100', 2021: 'v200'}[YEAR]
S3_URL_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# --- END CONFIGURATION ---

def main():
    """
    Main function to run the ESA WorldCover pre-processing pipeline.
    """
    logging.info("Starting ESA WorldCover feature pre-processing.")
    WORLDCOVER_TILES_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV_PATH.exists():
        logging.info(f"Output file {OUTPUT_CSV_PATH} already exists. To re-run, please delete it first.")
        return

    # 1. Load all sample points
    try:
        df_list = [pd.read_csv(p) for p in BIODIVERSITY_CSV_PATHS]
        points_df = pd.concat(df_list, ignore_index=True)
        points_df.dropna(subset=['latitude', 'longitude', 'sample_id'], inplace=True)
        logging.info(f"Loaded {len(points_df)} total points from CSVs.")
    except Exception as e:
        logging.error(f"Failed to load biodiversity data: {e}"); return
        
    # Convert points to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(points_df['longitude'], points_df['latitude'])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs="EPSG:4326")

    # 2. Load the ESA WorldCover grid
    logging.info("Loading ESA WorldCover grid file...")
    try:
        grid_url = f'{S3_URL_PREFIX}/v100/2020/esa_worldcover_2020_grid.geojson'
        grid_gdf = gpd.read_file(grid_url)
    except Exception as e:
        logging.error(f"Failed to load ESA grid file: {e}"); return
    
    # Ensure CRSs match
    grid_gdf = grid_gdf.to_crs(points_gdf.crs)

    # 3. Assign a tile name to each point using a spatial join
    logging.info("Assigning points to WorldCover tiles...")
    joined_gdf = gpd.sjoin(points_gdf, grid_gdf, how='inner', predicate='intersects')
    logging.info(f"{len(joined_gdf)} points were successfully located within the grid.")

    # 4. Process points tile by tile
    unique_tiles_needed = joined_gdf['ll_tile'].unique()
    logging.info(f"Found {len(unique_tiles_needed)} unique tiles to process.")
    
    all_results = []

    for tile_name in tqdm(unique_tiles_needed, desc="Processing Tiles"):
        tile_path = WORLDCOVER_TILES_DIR / f"ESA_WorldCover_10m_{YEAR}_{VERSION}_{tile_name}_Map.tif"

        # Download the tile if we don't have it
        if not tile_path.exists():
            logging.info(f"Downloading tile: {tile_name}")
            url = f"{S3_URL_PREFIX}/{VERSION}/{YEAR}/map/ESA_WorldCover_10m_{YEAR}_{VERSION}_{tile_name}_Map.tif"
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                with open(tile_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                logging.error(f"Failed to download {tile_name}: {e}")
                continue # Skip to the next tile

        # Extract values for all points in this tile
        try:
            points_in_this_tile = joined_gdf[joined_gdf['ll_tile'] == tile_name]
            coords = [(p.x, p.y) for p in points_in_this_tile.geometry]
            
            with rasterio.open(tile_path) as src:
                # The 'sample' method is highly efficient for this
                sampled_values = [val[0] for val in src.sample(coords)]
            
            # Store results
            for i, row_index in enumerate(points_in_this_tile.index):
                all_results.append({
                    'sample_id': points_in_this_tile.loc[row_index, 'sample_id'],
                    'land_cover_class': sampled_values[i]
                })

        except Exception as e:
            logging.error(f"Failed to process points for tile {tile_name}: {e}")

    # 5. Save final output
    if not all_results:
        logging.warning("No land cover features were extracted.")
        return

    output_df = pd.DataFrame(all_results)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logging.info(f"Successfully extracted features for {len(output_df)} samples.")
    logging.info(f"Output saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()