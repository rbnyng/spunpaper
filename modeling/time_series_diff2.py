import subprocess
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from rasterio.merge import merge
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling 
from skimage.util import view_as_windows
from spun_train_patch import CombinedPatchClimateEvaluator, filter_by_biome, lgb
import contextily as ctx
from matplotlib.colors import LogNorm, TwoSlopeNorm
import pyproj

class Config:
    TRAINING_BIODIVERSITY_CSVS = ["/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv", "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"]
    TRAINING_REPRESENTATIONS_DIR = "/maps-priv/maps/ray25/data/ecm_representations"
    MODEL_OUTPUT_DIR = Path("./model")
    EVALUATOR_SAVE_PATH = MODEL_OUTPUT_DIR / "evaluator_ssl_only.pkl"
    MODEL_SSL_ONLY_SAVE_PATH = MODEL_OUTPUT_DIR / "model_ssl_only.pkl"
    SATELLITE_DIM_REDUCTION = 'umap'
    DIM_REDUCTION_COMPONENTS = 256
    USE_BIOME_FILTER = True
    
    INFERENCE_ROI_GEOJSON = {
      "type": "FeatureCollection",
      "name": "CairngormsNationalPark",
      "features": [
        {
          "type": "Feature",
          "properties": {},
          "geometry": {
            "coordinates": [
              [
                [
                  -2.9453050333798103,
                  57.01221843324177
                ],
                [
                  -3.044386001519854,
                  57.01221843324177
                ],
                [
                  -3.044386001519854,
                  56.958547694672745
                ],
                [
                  -2.9453050333798103,
                  56.958547694672745
                ],
                [
                  -2.9453050333798103,
                  57.01221843324177
                ]
              ]
            ],
            "type": "Polygon"
          }
        }
      ]
    }
    INFERENCE_DATA_DIR = Path("./data/inference2")
    AOI_GEOJSON_PATH = INFERENCE_DATA_DIR / "aoi.geojson"
    
    # 2017 data paths
    EMBEDDINGS_DIR_2017 = INFERENCE_DATA_DIR / "embeddings_10m_2017"
    MOSAIC_10M_GEOTIFF_PATH_2017 = INFERENCE_DATA_DIR / "embeddings_mosaic_10m_2017.tif"
    PREDICTION_10M_GEOTIFF_PATH_2017 = INFERENCE_DATA_DIR / "prediction_ssl_10m_2017.tif"
    
    # 2024 data paths
    EMBEDDINGS_DIR_2024 = INFERENCE_DATA_DIR / "embeddings_10m_2024"
    MOSAIC_10M_GEOTIFF_PATH_2024 = INFERENCE_DATA_DIR / "embeddings_mosaic_10m_2024.tif"
    PREDICTION_10M_GEOTIFF_PATH_2024 = INFERENCE_DATA_DIR / "prediction_ssl_10m_2024.tif"
    
    YEARS_TO_ANALYZE = list(range(2017, 2025))
    YEARLY_PREDICTIONS_DIR = INFERENCE_DATA_DIR / "yearly_predictions"
    YEARLY_EMBEDDINGS_DIR = INFERENCE_DATA_DIR / "yearly_embeddings"
    LINE_PLOT_FIGURE_PATH = Path("./figures/Temporal_Line_Plot_2017_2024.png")
    
    CHANGE_MAP_PATH = INFERENCE_DATA_DIR / "change_2017_to_2024.tif"
    
    FINAL_FIGURE_PATH = Path("./figures/Temporal_Comparison_2017_2024.png")

def train_production_model(config: Config):
    print("\n--- TRAINING PRODUCTION MODEL ---")
    config.MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    if config.MODEL_SSL_ONLY_SAVE_PATH.exists():
        print("SSL-only model already exists. Skipping training.")
        return

    print("Loading and preparing training data...")
    df_list = [pd.read_csv(csv_path) for csv_path in config.TRAINING_BIODIVERSITY_CSVS]
    biodiversity_df = pd.concat(df_list, ignore_index=True).dropna(subset=['latitude', 'longitude', 'rarefied', 'sample_id'])
    if config.USE_BIOME_FILTER: biodiversity_df = filter_by_biome(biodiversity_df)

    print("\nTraining SSL-Only Model...")
    evaluator_ssl = CombinedPatchClimateEvaluator(climate_data_path=None, use_climate_cache=False, climate_features_cache_dir=None, soil_features_cache_dir=None)
    X_ssl, y_ssl, _, _, _, _ = evaluator_ssl.prepare_dataset(biodiversity_df, representations_dir_path=config.TRAINING_REPRESENTATIONS_DIR, use_satellite=True, use_climate=False, use_soil=False, use_worldcover=False, satellite_dim_reduction=config.SATELLITE_DIM_REDUCTION, dim_reduction_components=config.DIM_REDUCTION_COMPONENTS, random_state=42)
    X_ssl_features_only = X_ssl[:, :-2]
    
    print(f"Training LightGBM model on {X_ssl_features_only.shape[0]} samples...")
    model_ssl = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, n_jobs=-1, device='cuda')
    model_ssl.fit(X_ssl_features_only, y_ssl)
    
    print(f"Saving fitted SSL evaluator object to {config.EVALUATOR_SAVE_PATH}")
    with open(config.EVALUATOR_SAVE_PATH, 'wb') as f: pickle.dump(evaluator_ssl, f)
    print(f"Saving trained SSL-only model to {config.MODEL_SSL_ONLY_SAVE_PATH}")
    with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'wb') as f: pickle.dump(model_ssl, f)

    print("--- Model training complete. ---")

    
def generate_yearly_line_plot(config: Config):
    print("\n--- GENERATING YEARLY TIME-SERIES LINE PLOT ---")
    config.YEARLY_PREDICTIONS_DIR.mkdir(exist_ok=True)
    config.YEARLY_EMBEDDINGS_DIR.mkdir(exist_ok=True)

    yearly_mean_richness = {}
    yearly_q25 = {}  # 25th percentile
    yearly_q75 = {}  # 75th percentile

    with open(config.EVALUATOR_SAVE_PATH, 'rb') as f: evaluator = pickle.load(f)
    with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'rb') as f: model_ssl = pickle.load(f)

    for year in config.YEARS_TO_ANALYZE:
        print(f"\n--- Processing Year: {year} ---")
        prediction_path = config.YEARLY_PREDICTIONS_DIR / f"prediction_{year}.tif"
        
        if not prediction_path.exists():
            embeddings_dir = config.YEARLY_EMBEDDINGS_DIR / str(year)
            embeddings_dir.mkdir(exist_ok=True)
            if not any(embeddings_dir.iterdir()):
                print(f"Downloading Tessera embeddings for {year}...")
                command = ['geotessera', 'download', '--region-file', str(config.AOI_GEOJSON_PATH), '--year', str(year), '--format', 'tiff', '--output', str(embeddings_dir)]
                subprocess.run(command, check=True)
            
            embedding_files = list(embeddings_dir.glob('*.tif'))
            if not embedding_files:
                print(f"WARNING: No embedding tiles found for {year}. Skipping.")
                yearly_mean_richness[year] = np.nan
                yearly_q25[year] = np.nan
                yearly_q75[year] = np.nan
                continue
            
            src_files_to_mosaic = [rasterio.open(fp) for fp in embedding_files]
            mosaic, out_transform = merge(src_files_to_mosaic)
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform})

            print(f"Running inference for {year}...")
            mosaic_np = mosaic.transpose(1, 2, 0)
            h, w, c = mosaic_np.shape
            padded_mosaic = np.pad(mosaic_np, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=np.nan)
            windows = view_as_windows(padded_mosaic, (3, 3, c), step=1)
            feature_vectors = windows.reshape(h * w, -1)
            
            valid_mask = ~np.isnan(feature_vectors).any(axis=1)
            valid_features = feature_vectors[valid_mask]
            
            scaled_pixels = evaluator.scaler.transform(valid_features)
            reduced_pixels = evaluator.dim_reduction_model.transform(scaled_pixels)
            predictions_flat = model_ssl.predict(reduced_pixels)
            
            final_predictions = np.full(h * w, np.nan, dtype=np.float32)
            final_predictions[valid_mask] = predictions_flat
            prediction_map = final_predictions.reshape(h, w)
            
            out_meta.update(count=1, dtype='float32', nodata=np.nan, compress='LZW')
            with rasterio.open(prediction_path, 'w', **out_meta) as dst: dst.write(prediction_map, 1)
            print(f"Prediction map for {year} saved to {prediction_path}")
        else:
            print(f"Prediction map for {year} already exists. Loading...")
            with rasterio.open(prediction_path) as src:
                prediction_map = src.read(1)

        mean_richness = np.nanmean(prediction_map)
        q25_richness = np.nanpercentile(prediction_map, 25)
        q75_richness = np.nanpercentile(prediction_map, 75)
        
        yearly_mean_richness[year] = mean_richness
        yearly_q25[year] = q25_richness
        yearly_q75[year] = q75_richness
        
        print(f"Mean predicted richness for {year}: {mean_richness:.2f} (IQR: {q25_richness:.2f} - {q75_richness:.2f})")

    years = list(yearly_mean_richness.keys())
    mean_values = np.array(list(yearly_mean_richness.values()))
    q25_values = np.array(list(yearly_q25.values()))
    q75_values = np.array(list(yearly_q75.values()))
    
    # Calculate error bar lengths
    lower_error = np.abs(mean_values - q25_values)
    upper_error = np.abs(q75_values - mean_values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(years, mean_values, yerr=[lower_error, upper_error], 
                marker='o', linestyle='-', capsize=5, capthick=2, 
                label='Mean (25th-75th percentile)', linewidth=2, markersize=8)
    
    ax.set_title('Average Predicted Mycorrhizal Richness in Cairngorms (2017-2024)', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mean Predicted Richness (Rarefied)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xticks(years)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    fig.tight_layout()
    
    plt.savefig(config.LINE_PLOT_FIGURE_PATH, dpi=600)
    print(f"--- Yearly line plot saved to {config.LINE_PLOT_FIGURE_PATH} ---")
    
def acquire_and_prepare_inference_data(config: Config):
    print("\n--- ACQUIRING & PREPARING INFERENCE DATA ---")
    config.INFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.AOI_GEOJSON_PATH, 'w') as f: json.dump(config.INFERENCE_ROI_GEOJSON, f)

    for year, embeddings_dir, mosaic_path in [
        (2017, config.EMBEDDINGS_DIR_2017, config.MOSAIC_10M_GEOTIFF_PATH_2017),
        (2024, config.EMBEDDINGS_DIR_2024, config.MOSAIC_10M_GEOTIFF_PATH_2024)
    ]:
        if not mosaic_path.exists():
            print(f"\nPreparing 10m analysis-ready mosaic for {year}...")
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            if not any(embeddings_dir.iterdir()):
                print(f"Downloading Tessera embeddings for AOI (year {year})...")
                command = ['geotessera', 'download', '--region-file', str(config.AOI_GEOJSON_PATH), '--year', str(year), '--format', 'tiff', '--output', str(embeddings_dir)]
                subprocess.run(command, check=True)
            
            embedding_files = list(embeddings_dir.glob('*.tif'))
            if not embedding_files: print(f"ERROR: No embedding tiles found for {year}."); exit()
            
            src_files_to_mosaic = [rasterio.open(fp) for fp in embedding_files]
            mosaic, out_transform = merge(src_files_to_mosaic)
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform, "compress": "LZW"})
            
            with rasterio.open(mosaic_path, "w", **out_meta) as dest: dest.write(mosaic)
            for src in src_files_to_mosaic: src.close()
            print(f"Mosaic for {year} saved.")
        else:
            print(f"\n10m analysis-ready mosaic for {year} already exists. Skipping.")

    print("--- Inference data preparation complete. ---")
    
def run_all_inference(config: Config):
    print("\n--- RUNNING ALL INFERENCE ---")
    
    with open(config.EVALUATOR_SAVE_PATH, 'rb') as f: evaluator = pickle.load(f)
    with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'rb') as f: model_ssl = pickle.load(f)
    
    for year, mosaic_path, prediction_path in [
        (2017, config.MOSAIC_10M_GEOTIFF_PATH_2017, config.PREDICTION_10M_GEOTIFF_PATH_2017),
        (2024, config.MOSAIC_10M_GEOTIFF_PATH_2024, config.PREDICTION_10M_GEOTIFF_PATH_2024)
    ]:
        if not prediction_path.exists():
            print(f"\nRunning 10m SSL model inference for {year}...")
            
            with rasterio.open(mosaic_path) as src:
                mosaic_np = src.read().transpose(1, 2, 0)
                profile = src.profile
            
            h, w, c = mosaic_np.shape
            padded_mosaic = np.pad(mosaic_np, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=np.nan)
            windows = view_as_windows(padded_mosaic, (3, 3, c), step=1)
            feature_vectors = windows.reshape(h * w, -1)
            
            valid_mask = ~np.isnan(feature_vectors).any(axis=1)
            valid_features = feature_vectors[valid_mask]
            
            scaled_pixels = evaluator.scaler.transform(valid_features)
            reduced_pixels = evaluator.dim_reduction_model.transform(scaled_pixels)
            predictions_flat = model_ssl.predict(reduced_pixels)
            
            final_predictions = np.full(h * w, np.nan, dtype=np.float32)
            final_predictions[valid_mask] = predictions_flat
            prediction_map = final_predictions.reshape(h, w)
            
            profile.update(count=1, dtype='float32', nodata=np.nan)
            with rasterio.open(prediction_path, 'w', **profile) as dst: dst.write(prediction_map, 1)
            print(f"10m SSL prediction map for {year} saved.")
        else:
            print(f"\n10m prediction map for {year} already exists. Skipping inference.")
    
    if not config.CHANGE_MAP_PATH.exists():
        print("\nCalculating change map (2024 - 2017)...")
        
        with rasterio.open(config.PREDICTION_10M_GEOTIFF_PATH_2017) as src:
            pred_2017 = src.read(1)
            profile = src.profile
        
        with rasterio.open(config.PREDICTION_10M_GEOTIFF_PATH_2024) as src:
            pred_2024 = src.read(1)
        
        change_map = pred_2024 - pred_2017
        
        profile.update(count=1, dtype='float32', nodata=np.nan)
        with rasterio.open(config.CHANGE_MAP_PATH, 'w', **profile) as dst: 
            dst.write(change_map, 1)
        print("Change map saved.")
    else:
        print("\nChange map already exists. Skipping calculation.")
    
    print("--- All inference complete. ---")

def create_final_visualizations(config: Config):
    print("\n--- CREATING FINAL VISUALIZATION ---")
    config.FINAL_FIGURE_PATH.parent.mkdir(exist_ok=True, parents=True)

    with rasterio.open(config.PREDICTION_10M_GEOTIFF_PATH_2017) as src:
        pred_2017_data = src.read(1)
        src_bounds = src.bounds
    
    with rasterio.open(config.PREDICTION_10M_GEOTIFF_PATH_2024) as src:
        pred_2024_data = src.read(1)
    
    with rasterio.open(config.CHANGE_MAP_PATH) as src:
        change_data = src.read(1)

    fig, axes = plt.subplots(1, 3, figsize=(22, 12))
    
    # Shared normalization for 2017 and 2024
    #vmin = np.nanmin([np.nanmin(pred_2017_data), np.nanmin(pred_2024_data)])
    vmin = 10
    vmax = np.nanmax([np.nanmax(pred_2017_data), np.nanmax(pred_2024_data)])
    norm = LogNorm(vmin=max(vmin, 1), vmax=vmax)
    cmap = 'viridis'

    ax1 = axes[0]
    im1 = ax1.imshow(pred_2017_data, cmap=cmap, norm=norm, 
                     extent=(src_bounds.left, src_bounds.right, src_bounds.bottom, src_bounds.top))
    ax1.set_title('A) 2017 Prediction', fontsize=18)
    ax1.set_aspect('equal')
    ax1.set_axis_off()

    ax2 = axes[1]
    im2 = ax2.imshow(pred_2024_data, cmap=cmap, norm=norm, 
                     extent=(src_bounds.left, src_bounds.right, src_bounds.bottom, src_bounds.top))
    ax2.set_title('B) 2024 Prediction', fontsize=18)
    ax2.set_aspect('equal')
    ax2.set_axis_off()

    ax3 = axes[2]
    change_abs_max = np.nanmax(np.abs(change_data))
    change_norm = TwoSlopeNorm(vmin=-change_abs_max, vcenter=0, vmax=change_abs_max)
    im3 = ax3.imshow(change_data, cmap='RdBu_r', norm=change_norm, 
                     extent=(src_bounds.left, src_bounds.right, src_bounds.bottom, src_bounds.top))
    ax3.set_title('C) Change', fontsize=18)
    ax3.set_aspect('equal')
    ax3.set_axis_off()

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Colorbar for predictions (2017 & 2024)
    cbar_ax1 = fig.add_axes([0.93, 0.55, 0.015, 0.35])
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax1)
    linear_ticks = [10, 25, 50, 75, 100, 125, 150]
    actual_ticks = [t for t in linear_ticks if max(vmin, 1) <= t <= vmax]
    cbar1.set_ticks(actual_ticks)
    cbar1.set_ticklabels([f"{int(t)}" for t in actual_ticks])
    cbar1.set_label('Predicted Richness (Rarefied)', fontsize=12)
    
    # Colorbar for change map
    cbar_ax2 = fig.add_axes([0.93, 0.1, 0.015, 0.35])
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(norm=change_norm, cmap='RdBu_r'), cax=cbar_ax2)
    cbar2.set_label('Change in Richness (2024 - 2017)', fontsize=12)
    
    plt.savefig(config.FINAL_FIGURE_PATH, dpi=300, bbox_inches='tight')
    print(f"--- Final figure saved to {config.FINAL_FIGURE_PATH} ---")  

    
if __name__ == "__main__":
    cfg = Config()
    train_production_model(cfg)
    acquire_and_prepare_inference_data(cfg)
    run_all_inference(cfg)
    create_final_visualizations(cfg)
    generate_yearly_line_plot(cfg)
    print("\n\nPipeline finished successfully!")