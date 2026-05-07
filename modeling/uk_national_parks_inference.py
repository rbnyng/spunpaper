"""
UK National Parks Time Series Inference

Generates mycorrhizal richness prediction TIFs for UK national parks
using Tessera embeddings and a pre-trained SSL model.

Parks: Cairngorms, Lake District, Yorkshire Dales
Years: 2017, 2024 (extensible to full 2017-2024 range)

Usage:
    python uk_national_parks_inference.py [--workers N]
"""

import subprocess
import pickle
import json
import zipfile
import tempfile
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from rasterio.merge import merge
import rasterio
from rasterio.transform import from_origin
from skimage.util import view_as_windows
import geopandas as gpd

# Global variables for worker processes (initialized once per worker)
_worker_evaluator = None
_worker_model = None


def _init_worker(evaluator_path: Path, model_path: Path):
    """Initialize worker process by loading model once."""
    global _worker_evaluator, _worker_model
    with open(evaluator_path, 'rb') as f:
        _worker_evaluator = pickle.load(f)
    with open(model_path, 'rb') as f:
        _worker_model = pickle.load(f)


class Config:
    # === Training/Model Paths (existing infrastructure) ===
    TRAINING_BIODIVERSITY_CSVS = [
        "/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv",
        "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"
    ]
    TRAINING_REPRESENTATIONS_DIR = "/maps-priv/maps/ray25/data/ecm_representations"
    MODEL_OUTPUT_DIR = Path("./model")
    # Use PCA model (much faster than UMAP: 0.5s vs 1236s per tile)
    EVALUATOR_SAVE_PATH = MODEL_OUTPUT_DIR / "evaluator_ssl_pca.pkl"
    MODEL_SSL_ONLY_SAVE_PATH = MODEL_OUTPUT_DIR / "model_ssl_pca.pkl"
    SATELLITE_DIM_REDUCTION = 'pca'
    DIM_REDUCTION_COMPONENTS = 256
    USE_BIOME_FILTER = False

    # === UK National Parks Shapefiles ===
    NATIONAL_PARKS_DIR = Path("/maps-priv/maps/ray25/nationalparks")

    # Map of park names to their shapefile zip paths
    # Adjust these filenames to match your actual files
    PARK_SHAPEFILES = {
        "cairngorms": NATIONAL_PARKS_DIR / "cairngorms.zip",
        "lake_district": NATIONAL_PARKS_DIR / "lake_district.zip",
        "yorkshire_dales": NATIONAL_PARKS_DIR / "yorkshire_dales.zip",
    }

    # === Inference Configuration ===
    YEARS = list(range(2017, 2025))  # Can extend to list(range(2017, 2025)) for all years

    # === Output Paths ===
    OUTPUT_BASE_DIR = Path("./data/uk_national_parks")

    @classmethod
    def get_park_output_dir(cls, park_name: str) -> Path:
        return cls.OUTPUT_BASE_DIR / park_name

    @classmethod
    def get_embeddings_dir(cls, park_name: str, year: int) -> Path:
        return cls.get_park_output_dir(park_name) / f"embeddings_{year}"

    @classmethod
    def get_mosaic_path(cls, park_name: str, year: int) -> Path:
        return cls.get_park_output_dir(park_name) / f"mosaic_{year}.tif"

    @classmethod
    def get_prediction_path(cls, park_name: str, year: int) -> Path:
        return cls.get_park_output_dir(park_name) / f"prediction_{year}.tif"

    @classmethod
    def get_aoi_geojson_path(cls, park_name: str) -> Path:
        return cls.get_park_output_dir(park_name) / "aoi.geojson"


def load_park_boundary(shapefile_path: Path) -> gpd.GeoDataFrame:
    """
    Load national park boundary from a zipped shapefile.

    Handles both:
    - Direct .zip files containing shapefiles
    - Directories containing .shp files
    """
    if shapefile_path.suffix == '.zip':
        # Read directly from zip
        gdf = gpd.read_file(f"zip://{shapefile_path}")
    elif shapefile_path.suffix == '.shp':
        gdf = gpd.read_file(shapefile_path)
    elif shapefile_path.is_dir():
        # Find .shp file in directory
        shp_files = list(shapefile_path.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No .shp file found in {shapefile_path}")
        gdf = gpd.read_file(shp_files[0])
    else:
        raise ValueError(f"Unsupported shapefile format: {shapefile_path}")

    # Ensure WGS84 for geotessera
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def gdf_to_bbox_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """
    Convert a GeoDataFrame to a bounding box GeoJSON FeatureCollection.

    Uses the total bounds of all geometries to create a simple bbox polygon.
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny]
                ]]
            }
        }]
    }


def ensure_model_exists(config: Config):
    """
    Check that the pre-trained SSL model exists.
    If not, provide instructions for training.
    """
    if not config.MODEL_SSL_ONLY_SAVE_PATH.exists():
        print("ERROR: Pre-trained SSL model not found!")
        print(f"Expected path: {config.MODEL_SSL_ONLY_SAVE_PATH}")
        print("\nPlease run the training step first using 10m_map.py or time_series_diff.py")
        print("Or copy the existing model files to the ./model/ directory")
        exit(1)

    if not config.EVALUATOR_SAVE_PATH.exists():
        print("ERROR: Evaluator object not found!")
        print(f"Expected path: {config.EVALUATOR_SAVE_PATH}")
        exit(1)

    print("✓ Pre-trained model and evaluator found")


def download_embeddings_for_park(config: Config, park_name: str, year: int, aoi_geojson_path: Path):
    """
    Download Tessera embeddings for a given park and year.
    """
    embeddings_dir = config.get_embeddings_dir(park_name, year)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # geotessera nests files: global_0.1_degree_representation/{year}/grid_*/grid_*_{year}.tiff
    repr_dir = embeddings_dir / 'global_0.1_degree_representation' / str(year)
    if repr_dir.exists() and any(repr_dir.glob('*/*.tiff')):
        print(f"  Embeddings for {park_name} {year} already exist. Skipping download.")
        return

    print(f"  Downloading Tessera embeddings for {park_name} {year}...")
    command = [
        'geotessera', 'download',
        '--region-file', str(aoi_geojson_path),
        '--year', str(year),
        '--format', 'tiff',
        '--output', str(embeddings_dir)
    ]

    try:
        subprocess.run(command, check=True)
        print(f"  Download complete for {park_name} {year}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to download embeddings for {park_name} {year}")
        print(f"  Command: {' '.join(command)}")
        raise


def process_single_tile_str(tile_path_str: str, predictions_dir_str: str):
    """Wrapper that takes strings (for multiprocessing pickling)."""
    return process_single_tile(Path(tile_path_str), Path(predictions_dir_str))


def process_single_tile(tile_path: Path, predictions_dir: Path):
    """
    Worker function to process a single tile. Designed for parallel execution.
    Uses global _worker_evaluator and _worker_model (loaded once per worker via initializer).
    """
    global _worker_evaluator, _worker_model

    pred_tile_path = predictions_dir / f"{tile_path.stem}_pred.tif"

    # Skip if already exists
    if pred_tile_path.exists():
        return "skipped"

    try:
        with rasterio.open(tile_path) as src:
            tile_data = src.read().transpose(1, 2, 0)  # (H, W, C)
            profile = src.profile

        h, w, c = tile_data.shape

        # Create 3x3 windows for spatial context
        padded = np.pad(tile_data, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=np.nan)
        windows = view_as_windows(padded, (3, 3, c), step=1)
        feature_vectors = windows.reshape(h * w, -1)

        # Filter valid pixels
        valid_mask = ~np.isnan(feature_vectors).any(axis=1)
        valid_features = feature_vectors[valid_mask]

        if valid_features.shape[0] == 0:
            return "empty"

        # Apply model pipeline
        scaled_pixels = _worker_evaluator.scaler.transform(valid_features)
        reduced_pixels = _worker_evaluator.dim_reduction_model.transform(scaled_pixels)
        predictions_flat = _worker_model.predict(reduced_pixels)

        # Reconstruct prediction map
        final_predictions = np.full(h * w, np.nan, dtype=np.float32)
        final_predictions[valid_mask] = predictions_flat
        prediction_map = final_predictions.reshape(h, w)

        # Save prediction tile
        profile.update(count=1, dtype='float32', nodata=np.nan, compress='LZW')
        with rasterio.open(pred_tile_path, 'w', **profile) as dst:
            dst.write(prediction_map, 1)

        return "processed"

    except Exception as e:
        return f"error: {e}"


def run_inference_per_tile(config: Config, park_name: str, year: int, evaluator, model, n_workers: int = 1):
    """
    Run SSL model inference on each embedding tile individually.
    Saves per-tile predictions to a subdirectory.

    Args:
        n_workers: Number of parallel workers (1 = sequential)
    """
    embeddings_dir = config.get_embeddings_dir(park_name, year)
    predictions_dir = config.get_park_output_dir(park_name) / f"predictions_tiles_{year}"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Find embedding tiles
    repr_dir = embeddings_dir / 'global_0.1_degree_representation' / str(year)
    embedding_files = list(repr_dir.glob('*/*.tiff')) + list(repr_dir.glob('*/*.tif'))

    if not embedding_files:
        print(f"  ERROR: No embedding tiles found for {park_name} {year}")
        return False

    # Filter to only tiles that need processing
    tiles_to_process = []
    skipped = 0
    for tile_path in embedding_files:
        pred_tile_path = predictions_dir / f"{tile_path.stem}_pred.tif"
        if pred_tile_path.exists():
            skipped += 1
        else:
            tiles_to_process.append(tile_path)

    print(f"  Running inference on {len(embedding_files)} tiles ({skipped} already done, {len(tiles_to_process)} to process)...")

    if not tiles_to_process:
        print(f"  All tiles already processed!")
        return True

    processed = 0
    errors = 0

    if n_workers == 1:
        # Sequential processing - initialize globals in main process
        _init_worker(config.EVALUATOR_SAVE_PATH, config.MODEL_SSL_ONLY_SAVE_PATH)

        for i, tile_path in enumerate(tiles_to_process):
            result = process_single_tile(tile_path, predictions_dir)
            if result == "processed":
                processed += 1
            elif result.startswith("error"):
                errors += 1
                print(f"    ERROR: {tile_path.name}: {result}")

            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(tiles_to_process)} tiles...")
    else:
        # Parallel processing with multiprocessing.Pool (true parallelism)
        print(f"  Using {n_workers} parallel processes...")

        # Convert paths to strings for pickling, create args list
        args_list = [(str(tile), str(predictions_dir)) for tile in tiles_to_process]

        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(config.EVALUATOR_SAVE_PATH, config.MODEL_SSL_ONLY_SAVE_PATH)
        ) as pool:
            results = pool.starmap(process_single_tile_str, args_list, chunksize=1)

        for result in results:
            if result == "processed":
                processed += 1
            elif result.startswith("error"):
                errors += 1

        print(f"    Completed {len(results)} tiles")

    print(f"  Inference complete: {processed} processed, {skipped} skipped, {errors} errors")
    return True


def merge_prediction_tiles(config: Config, park_name: str, year: int):
    """
    Merge per-tile predictions into a single prediction GeoTIFF.
    Only merges single-band predictions (much lower memory than 128-band embeddings).
    """
    predictions_dir = config.get_park_output_dir(park_name) / f"predictions_tiles_{year}"
    final_prediction_path = config.get_prediction_path(park_name, year)

    if final_prediction_path.exists():
        print(f"  Final prediction for {park_name} {year} already exists. Skipping merge.")
        return

    pred_tiles = list(predictions_dir.glob('*_pred.tif'))
    if not pred_tiles:
        print(f"  ERROR: No prediction tiles found to merge for {park_name} {year}")
        return

    print(f"  Merging {len(pred_tiles)} prediction tiles...")

    # Open all prediction tiles
    src_files = [rasterio.open(fp) for fp in pred_tiles]
    mosaic_data, out_transform = merge(src_files)

    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic_data.shape[1],
        "width": mosaic_data.shape[2],
        "transform": out_transform,
        "compress": "LZW",
        "BIGTIFF": "IF_SAFER"
    })

    with rasterio.open(final_prediction_path, "w", **out_meta) as dest:
        dest.write(mosaic_data)

    for src in src_files:
        src.close()

    print(f"  Final prediction saved: {final_prediction_path}")

    # Print stats
    valid_preds = mosaic_data[~np.isnan(mosaic_data)]
    if valid_preds.size > 0:
        print(f"    Richness range: {valid_preds.min():.1f} - {valid_preds.max():.1f}")
        print(f"    Richness mean: {valid_preds.mean():.1f}")


def process_park(config: Config, park_name: str, evaluator, model, n_workers: int = 1):
    """
    Full processing pipeline for a single national park.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {park_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")

    # Setup output directory
    output_dir = config.get_park_output_dir(park_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load park boundary
    shapefile_path = config.PARK_SHAPEFILES[park_name]
    print(f"\nLoading boundary from: {shapefile_path}")

    try:
        gdf = load_park_boundary(shapefile_path)
        print(f"  Loaded {len(gdf)} feature(s)")
        print(f"  Total bounds: {gdf.total_bounds}")
    except Exception as e:
        print(f"  ERROR loading shapefile: {e}")
        return

    # Create AOI GeoJSON (bounding box)
    aoi_geojson = gdf_to_bbox_geojson(gdf)
    aoi_path = config.get_aoi_geojson_path(park_name)
    with open(aoi_path, 'w') as f:
        json.dump(aoi_geojson, f)
    print(f"  AOI GeoJSON saved: {aoi_path}")

    # Process each year
    for year in config.YEARS:
        print(f"\n--- Year {year} ---")
        download_embeddings_for_park(config, park_name, year, aoi_path)
        if run_inference_per_tile(config, park_name, year, evaluator, model, n_workers=n_workers):
            merge_prediction_tiles(config, park_name, year)


def main():
    parser = argparse.ArgumentParser(description="UK National Parks Mycorrhizal Richness Prediction")
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of parallel workers for inference (default: 1)')
    args = parser.parse_args()

    print("="*60)
    print("UK National Parks - Mycorrhizal Richness Prediction")
    print("="*60)

    config = Config()

    # Verify model exists
    print("\nChecking for pre-trained model...")
    ensure_model_exists(config)

    # Load model and evaluator once (for main process, workers load their own)
    print("\nLoading model and evaluator...")
    with open(config.EVALUATOR_SAVE_PATH, 'rb') as f:
        evaluator = pickle.load(f)
    with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")

    # Process each park
    parks_to_process = list(config.PARK_SHAPEFILES.keys())
    print(f"\nParks to process: {parks_to_process}")
    print(f"Years: {config.YEARS}")
    print(f"Workers: {args.workers}")

    for park_name in parks_to_process:
        try:
            process_park(config, park_name, evaluator, model, n_workers=args.workers)
        except Exception as e:
            print(f"\nERROR processing {park_name}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {config.OUTPUT_BASE_DIR}")
    print("\nGenerated files:")

    for park_name in parks_to_process:
        park_dir = config.get_park_output_dir(park_name)
        if park_dir.exists():
            print(f"\n{park_name}/")
            for f in sorted(park_dir.glob("prediction_*.tif")):
                print(f"  {f.name}")


if __name__ == "__main__":
    main()
