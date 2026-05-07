"""
Profile tile inference to identify bottlenecks.

Usage:
    python profile_tile_inference.py [tile_path]
"""

import time
import pickle
import sys
import numpy as np
import rasterio
from pathlib import Path
from skimage.util import view_as_windows

# Default tile path
DEFAULT_TILE = "data/uk_national_parks/cairngorms/embeddings_2017/global_0.1_degree_representation/2017/grid_-3.55_57.35/grid_-3.55_57.35_2017.tiff"

def profile_tile(tile_path: str):
    print("=" * 60)
    print("Tile Inference Profiling")
    print("=" * 60)
    print(f"\nTile: {tile_path}\n")

    # Load model
    print("Loading model and evaluator...")
    t0 = time.time()
    with open('model/evaluator_ssl_only.pkl', 'rb') as f:
        evaluator = pickle.load(f)
    with open('model/model_ssl_only.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"  Model load time: {time.time()-t0:.1f}s\n")

    # Step 1: Read tile
    print("Step 1: Reading tile from disk...")
    t0 = time.time()
    with rasterio.open(tile_path) as src:
        tile_data = src.read().transpose(1, 2, 0)
    read_time = time.time() - t0
    print(f"  Time: {read_time:.1f}s")
    print(f"  Shape: {tile_data.shape} (H x W x Bands)")
    print(f"  Size: {tile_data.nbytes / 1e9:.2f} GB\n")

    # Step 2: Create windowed features
    print("Step 2: Creating 3x3 window features...")
    t0 = time.time()
    h, w, c = tile_data.shape
    padded = np.pad(tile_data, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=np.nan)
    windows = view_as_windows(padded, (3, 3, c), step=1)
    features = windows.reshape(h * w, -1)
    valid_mask = ~np.isnan(features).any(axis=1)
    valid_features = features[valid_mask]
    window_time = time.time() - t0
    print(f"  Time: {window_time:.1f}s")
    print(f"  Total pixels: {h * w:,}")
    print(f"  Valid pixels: {valid_features.shape[0]:,}")
    print(f"  Feature dim: {valid_features.shape[1]} (3x3x{c})\n")

    # Step 3: Scaler transform
    print("Step 3: StandardScaler transform...")
    t0 = time.time()
    scaled = evaluator.scaler.transform(valid_features)
    scaler_time = time.time() - t0
    print(f"  Time: {scaler_time:.1f}s\n")

    # Step 4: UMAP/dim reduction transform
    print("Step 4: Dimensionality reduction transform...")
    print(f"  Model type: {type(evaluator.dim_reduction_model).__name__}")
    t0 = time.time()
    reduced = evaluator.dim_reduction_model.transform(scaled)
    umap_time = time.time() - t0
    print(f"  Time: {umap_time:.1f}s")
    print(f"  Output dim: {reduced.shape[1]}\n")

    # Step 5: LightGBM predict
    print("Step 5: LightGBM prediction...")
    t0 = time.time()
    preds = model.predict(reduced)
    predict_time = time.time() - t0
    print(f"  Time: {predict_time:.1f}s")
    print(f"  Predictions: {preds.shape[0]:,}")
    print(f"  Range: {preds.min():.1f} - {preds.max():.1f}\n")

    # Summary
    total_time = read_time + window_time + scaler_time + umap_time + predict_time
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Step':<30} {'Time':>10} {'%':>8}")
    print("-" * 50)
    print(f"{'1. Read tile':<30} {read_time:>9.1f}s {100*read_time/total_time:>7.1f}%")
    print(f"{'2. Window features':<30} {window_time:>9.1f}s {100*window_time/total_time:>7.1f}%")
    print(f"{'3. Scaler transform':<30} {scaler_time:>9.1f}s {100*scaler_time/total_time:>7.1f}%")
    print(f"{'4. Dim reduction (UMAP)':<30} {umap_time:>9.1f}s {100*umap_time/total_time:>7.1f}%")
    print(f"{'5. LightGBM predict':<30} {predict_time:>9.1f}s {100*predict_time/total_time:>7.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<30} {total_time:>9.1f}s {100:>7.1f}%")
    print()

    # Bottleneck analysis
    times = {
        'Read tile': read_time,
        'Window features': window_time,
        'Scaler': scaler_time,
        'Dim reduction': umap_time,
        'LightGBM': predict_time
    }
    bottleneck = max(times, key=times.get)
    print(f"BOTTLENECK: {bottleneck} ({times[bottleneck]:.1f}s, {100*times[bottleneck]/total_time:.0f}% of total)")


if __name__ == "__main__":
    tile_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TILE

    if not Path(tile_path).exists():
        print(f"ERROR: Tile not found: {tile_path}")
        print("\nUsage: python profile_tile_inference.py [tile_path]")
        sys.exit(1)

    profile_tile(tile_path)
