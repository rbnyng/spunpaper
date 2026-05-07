import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS, Transformer
import os
import math

# --- Configuration ---
CSV_FILE = '/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv'
OUTPUT_DIR = 'ECM_asia_tifs_patches'
PIXEL_SIZE = 10  # meters
PATCH_SIZE = 3   # creates a 3x3 pixel grid
# List of columns
COLUMNS_TO_RASTERIZE = ['rarefied']

# --- Helper Function to get the correct UTM zone ---
def get_utm_crs(lon, lat):
    """Calculates the UTM zone for a given lat/lon and returns a pyproj.CRS object."""
    utm_band = str(int((lon + 180) / 6) + 1)
    if lat >= 0:
        epsg_code = '326' + utm_band.zfill(2)
    else:
        epsg_code = '327' + utm_band.zfill(2)
    return CRS(f"EPSG:{epsg_code}")

# --- Main Script ---

# 1. Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Read the CSV data
try:
    df = pd.read_csv(CSV_FILE)
    # Remove any extra whitespace from column names
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE}' was not found.")
    exit()

print(f"Processing {len(df)} samples from '{CSV_FILE}'...")

# 3. Loop through each row in the DataFrame
for index, row in df.iterrows():
    sample_id = row['sample_id']
    lon = row['longitude']
    lat = row['latitude']

    if pd.isna(lon) or pd.isna(lat):
        print(f"WARNING: Skipping sample_id '{sample_id}' (CSV row {index + 2}) due to missing longitude/latitude.")
        continue
        
    # --- Coordinate Projection ---
    # Define the source CRS (WGS84) and target CRS (the local UTM zone)
    source_crs = CRS("EPSG:4326")
    target_crs = get_utm_crs(lon, lat)

    # Create a transformer to convert from WGS84 to UTM
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Project the center coordinate of our sample
    center_x, center_y = transformer.transform(lon, lat)

    # --- Calculate Georeference Information ---
    # Total width/height of our 3x3 patch in meters
    total_dimension = PATCH_SIZE * PIXEL_SIZE
    half_dimension = total_dimension / 2.0

    # The top-left corner is the origin for the geotransform
    # In raster coordinates, the origin is the top-left corner of the top-left pixel
    x_min = center_x - half_dimension
    y_max = center_y + half_dimension

    # Create the affine transformation matrix
    # This matrix maps pixel coordinates to the CRS coordinates
    transform = from_origin(x_min, y_max, PIXEL_SIZE, PIXEL_SIZE)

    # 4. Loop through the specified columns to create a TIF for each one
    for col_name in COLUMNS_TO_RASTERIZE:
        # Also check if the data value itself is NaN
        value = row[col_name]
        if pd.isna(value):
            print(f"  -> WARNING: Skipping column '{col_name}' for sample '{sample_id}' because its value is missing.")
            continue
            
        # --- Create Data Array ---
        # Create a 3x3 NumPy array and fill it with the value
        data_array = np.full((PATCH_SIZE, PATCH_SIZE), value, dtype=np.float64)

        # --- Write the GeoTIFF ---
        output_filename = f"2021_{sample_id}_{col_name}.tif"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"  Creating '{output_path}' with value {value}...")

        # Define the metadata for the new raster file
        profile = {
            'driver': 'GTiff',
            'height': PATCH_SIZE,
            'width': PATCH_SIZE,
            'count': 1, # Number of bands
            'dtype': data_array.dtype,
            'crs': target_crs,
            'transform': transform,
            'nodata': -9999  # A value to represent no data, if needed
        }

        # Write the data to the file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data_array, 1) # Write to the first band

print("\nProcessing complete.")
print(f"All TIF files have been saved in the '{OUTPUT_DIR}' directory.")