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
from spun_train_patch import CombinedPatchClimateEvaluator, filter_by_biome, lgb, ClimateExtractor
import contextily as ctx
from matplotlib.colors import PowerNorm, LogNorm

class Config:
    TRAINING_BIODIVERSITY_CSVS = ["/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv", "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"]
    TRAINING_REPRESENTATIONS_DIR = "/maps-priv/maps/ray25/data/ecm_representations"
    CLIMATE_DATA_DIR = "/maps-priv/maps/ray25/data/worldclim/data"
    CLIMATE_CACHE_DIR = Path("/scratch/ray25/climate_features_cache")
    MODEL_OUTPUT_DIR = Path("./model")
    EVALUATOR_SAVE_PATH = MODEL_OUTPUT_DIR / "evaluator_ssl_only.pkl"
    MODEL_SSL_ONLY_SAVE_PATH = MODEL_OUTPUT_DIR / "model_ssl_only.pkl"
    MODEL_CLIMATE_ONLY_SAVE_PATH = MODEL_OUTPUT_DIR / "model_climate_only.pkl"
    SATELLITE_DIM_REDUCTION = 'umap'
    DIM_REDUCTION_COMPONENTS = 256
    USE_BIOME_FILTER = True
    
    INFERENCE_AOI_GEOJSON = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {},
          "geometry": {
            "coordinates": [
              [
                [
                  8.885467583699409,
                  46.345958140034384
                ],
                [
                  8.749107035791724,
                  46.345958140034384
                ],
                [
                  8.749107035791724,
                  46.25185428184591
                ],
                [
                  8.885467583699409,
                  46.25185428184591
                ],
                [
                  8.885467583699409,
                  46.345958140034384
                ]
              ]
            ],
            "type": "Polygon"
          }
        }
      ]
    }
    INFERENCE_DATA_DIR = Path("./data/inference")
    AOI_GEOJSON_PATH = INFERENCE_DATA_DIR / "aoi.geojson"
    
    EMBEDDINGS_DIR = INFERENCE_DATA_DIR / "embeddings_10m"
    MOSAIC_10M_GEOTIFF_PATH = INFERENCE_DATA_DIR / "embeddings_mosaic_10m.tif"
    PREDICTION_10M_GEOTIFF_PATH = INFERENCE_DATA_DIR / "prediction_ssl_10m.tif"
    CLIMATE_FEATURES_1KM_RASTER_PATH = INFERENCE_DATA_DIR / "climate_features_1km.tif"
    PREDICTION_1KM_GEOTIFF_PATH = INFERENCE_DATA_DIR / "prediction_climate_1km.tif"
    
    # WorldCover masking
    WORLDCOVER_10M_PATH = INFERENCE_DATA_DIR / "worldcover_10m.tif"
    # ESA WorldCover classes to keep (vegetated surfaces where EcM predictions make sense)
    # 10=Tree cover, 20=Shrubland, 30=Grassland, 40=Cropland, 90=Herbaceous wetland, 95=Mangroves, 100=Moss and lichen
    WORLDCOVER_VALID_CLASSES = [10, 20, 30, 40, 90, 95, 100]
    
    FINAL_FIGURE_PATH = Path("./figures/3_Panel_Comparison.png")

def train_production_models(config: Config):
    print("\n--- PHASE 1: TRAINING PRODUCTION MODELS ---")
    config.MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    if config.MODEL_SSL_ONLY_SAVE_PATH.exists() and config.MODEL_CLIMATE_ONLY_SAVE_PATH.exists():
        print("Both SSL-only and Climate-only models already exist. Skipping training.")
        return

    print("Loading and preparing training data...")
    df_list = [pd.read_csv(csv_path) for csv_path in config.TRAINING_BIODIVERSITY_CSVS]
    biodiversity_df = pd.concat(df_list, ignore_index=True).dropna(subset=['latitude', 'longitude', 'rarefied', 'sample_id'])
    if config.USE_BIOME_FILTER: biodiversity_df = filter_by_biome(biodiversity_df)

    # --- Train SSL-Only Model ---
    if not config.MODEL_SSL_ONLY_SAVE_PATH.exists():
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
    else:
        print("\nSSL-Only model already exists.")

    # --- Train Climate-Only Model ---
    if not config.MODEL_CLIMATE_ONLY_SAVE_PATH.exists():
        print("\nTraining Climate-Only Model...")
        
        climate_extractor = ClimateExtractor(config.CLIMATE_DATA_DIR, use_cache=True)
        if not config.CLIMATE_CACHE_DIR.exists():
            print(f"Existing cache directory not found at {config.CLIMATE_CACHE_DIR}")
        
        all_climate_features = []
        desc = "Extracting Climate Features (from existing cache)"
        for _, row in tqdm(biodiversity_df.iterrows(), total=len(biodiversity_df), desc=desc):
            sample_id = str(row['sample_id'])
            cache_file = config.CLIMATE_CACHE_DIR / f"{sample_id}.npy"

            if cache_file.exists():
                feats = np.load(cache_file, allow_pickle=True).item()
            else:
                feats = climate_extractor.extract_climate_features(row['latitude'], row['longitude'])
                config.CLIMATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, feats)
            
            feats['sample_id'] = sample_id
            all_climate_features.append(feats)
            
        climate_df = pd.DataFrame(all_climate_features).dropna()
        merged_df = pd.merge(biodiversity_df[['sample_id', 'rarefied']], climate_df, on='sample_id')
        
        feature_names = [col for col in merged_df.columns if col not in ['sample_id', 'rarefied']]
        X_clim = merged_df[feature_names].values
        y_clim = merged_df['rarefied'].values

        print(f"Training LightGBM model on {X_clim.shape[0]} samples...")
        model_clim = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, n_jobs=-1, device='cuda')
        model_clim.fit(X_clim, y_clim)
        
        print(f"Saving trained Climate-only model to {config.MODEL_CLIMATE_ONLY_SAVE_PATH}")
        with open(config.MODEL_CLIMATE_ONLY_SAVE_PATH, 'wb') as f: pickle.dump((model_clim, feature_names), f)
    else:
        print("\nClimate-Only model already exists.")

    print("--- Model training complete. ---")

def acquire_and_prepare_inference_data(config: Config):
    print("\n--- PHASE 2: ACQUIRING & PREPARING INFERENCE DATA ---")
    config.INFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.AOI_GEOJSON_PATH, 'w') as f: json.dump(config.INFERENCE_AOI_GEOJSON, f)

    # --- Prepare 10m SSL Data ---
    if not config.MOSAIC_10M_GEOTIFF_PATH.exists():
        print("Preparing 10m analysis-ready mosaic...")
        config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        if not any(config.EMBEDDINGS_DIR.iterdir()):
            print(f"Downloading Tessera embeddings for AOI...")
            command = ['geotessera', 'download', '--region-file', str(config.AOI_GEOJSON_PATH), '--year', '2024', '--format', 'tiff', '--output', str(config.EMBEDDINGS_DIR)]
            subprocess.run(command, check=True)
        
        embedding_files = list(config.EMBEDDINGS_DIR.glob('*.tif'))
        if not embedding_files: print("ERROR: No embedding tiles found."); exit()
        
        src_files_to_mosaic = [rasterio.open(fp) for fp in embedding_files]
        mosaic, out_transform = merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform, "compress": "LZW"})
        
        with rasterio.open(config.MOSAIC_10M_GEOTIFF_PATH, "w", **out_meta) as dest: dest.write(mosaic)
        for src in src_files_to_mosaic: src.close()
    else:
        print(f"10m analysis-ready mosaic already exists. Skipping.")

    # --- Acquire and Reproject WorldCover Mask ---
    if not config.WORLDCOVER_10M_PATH.exists():
        print("\nAcquiring and reprojecting ESA WorldCover for AOI...")
        
        # Download WorldCover tile covering AOI
        # ESA WorldCover 2021 is available via several APIs. Here we use
        # the Planetary Computer STAC catalog, but you can also download
        # tiles directly from https://worldcover2021.esa.int/
        # For simplicity, assume the tile is pre-downloaded or use a CLI tool.
        # Alternative: use the same bounds to crop from a global/regional WorldCover file.
        
        with rasterio.open(config.MOSAIC_10M_GEOTIFF_PATH) as ref:
            ref_profile = ref.profile
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_height = ref.height
            ref_width = ref.width
            ref_bounds = ref.bounds
        
        # Try downloading via planetary computer stac
        try:
            import planetary_computer as pc
            import pystac_client
            
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace,
            )
            
            search = catalog.search(
                collections=["esa-worldcover"],
                bbox=[ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top],
            )
            items = list(search.items())
            
            if items:
                worldcover_href = items[0].assets["map"].href
                with rasterio.open(worldcover_href) as wc_src:
                    # Reproject and crop to match our 10m prediction grid exactly
                    wc_data = np.empty((ref_height, ref_width), dtype=np.uint8)
                    reproject(
                        source=rasterio.band(wc_src, 1),
                        destination=wc_data,
                        src_transform=wc_src.transform,
                        src_crs=wc_src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest,  # Nearest for categorical data
                        dst_nodata=0
                    )
                
                wc_profile = ref_profile.copy()
                wc_profile.update(count=1, dtype='uint8', nodata=0)
                with rasterio.open(config.WORLDCOVER_10M_PATH, 'w', **wc_profile) as dst:
                    dst.write(wc_data, 1)
                print("WorldCover mask saved.")
            else:
                print("WARNING: No WorldCover tiles found for AOI. Proceeding without mask.")
                
        except ImportError:
            print("WARNING: planetary_computer/pystac_client not available.")
            print("Please download ESA WorldCover 2021 tile for your AOI manually")
            print("and save it to:", config.WORLDCOVER_10M_PATH)
            print("Proceeding without mask.")
    else:
        print("WorldCover mask already exists. Skipping.")

    # --- Prepare 1km Climate Data ---
    if not config.CLIMATE_FEATURES_1KM_RASTER_PATH.exists():
        print("\nGenerating 1km climate feature raster for AOI (Full Statistical Method)...")
        
        with rasterio.open(config.MOSAIC_10M_GEOTIFF_PATH) as src:
            bounds = src.bounds
            target_crs = src.crs

        resolution = 1000
        width = int((bounds.right - bounds.left) / resolution)
        height = int((bounds.top - bounds.bottom) / resolution)
        transform = from_origin(bounds.left, bounds.top, resolution, resolution)
        
        with open(config.MODEL_CLIMATE_ONLY_SAVE_PATH, 'rb') as f: _, feature_names = pickle.load(f)
        
        climate_extractor = ClimateExtractor(config.CLIMATE_DATA_DIR, use_cache=True)
        band_data_dict = {}

        monthly_vars = ['prec', 'tavg', 'tmin', 'tmax', 'srad', 'vapr', 'wind']
        
        for var in tqdm(monthly_vars, desc="Processing Monthly Variables"):
            monthly_rasters = []
            for month in range(1, 13):
                source_path = climate_extractor.climate_metadata[var][month]['path']
                with rasterio.open(source_path) as src:
                    destination = np.empty((height, width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1), destination=destination,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=transform, dst_crs=target_crs,
                        resampling=Resampling.bilinear, dst_nodata=np.nan
                    )
                    monthly_rasters.append(destination)
            
            stack = np.stack(monthly_rasters, axis=0)
            band_data_dict[f'{var}_annual_mean'] = np.nanmean(stack, axis=0)
            band_data_dict[f'{var}_annual_min'] = np.nanmin(stack, axis=0)
            band_data_dict[f'{var}_annual_max'] = np.nanmax(stack, axis=0)
            band_data_dict[f'{var}_annual_std'] = np.nanstd(stack, axis=0)

        static_features = [f for f in feature_names if not any(v in f for v in monthly_vars)]
        for feature_name in tqdm(static_features, desc="Reprojecting Static Rasters"):
            source_path = climate_extractor.climate_metadata[feature_name]['path']
            with rasterio.open(source_path) as src:
                destination = np.empty((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=destination,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=target_crs,
                    resampling=Resampling.bilinear, dst_nodata=np.nan
                )
                band_data_dict[feature_name] = destination
        
        final_band_stack = []
        for name in feature_names:
            if name in band_data_dict:
                final_band_stack.append(band_data_dict[name])
            else:
                print(f"FATAL: Could not generate data for required feature '{name}'."); exit()
        
        climate_raster_data = np.stack(final_band_stack, axis=0)
        
        profile = { 'driver': 'GTiff', 'count': len(feature_names), 'dtype': 'float32', 'nodata': np.nan,
                    'width': width, 'height': height, 'crs': target_crs, 'transform': transform, 'compress': 'LZW'}
        with rasterio.open(config.CLIMATE_FEATURES_1KM_RASTER_PATH, 'w', **profile) as dst:
            dst.write(climate_raster_data)
        print("1km climate feature raster saved.")
    else:
        print("1km climate feature raster already exists. Skipping.")
    print("--- Inference data preparation complete. ---")
    
def run_all_inference(config: Config):
    print("\n--- PHASE 3: RUNNING ALL INFERENCE ---")
    
    # --- Run 10m Inference ---
    if not config.PREDICTION_10M_GEOTIFF_PATH.exists():
        print("Running 10m SSL model inference...")
        with open(config.EVALUATOR_SAVE_PATH, 'rb') as f: evaluator = pickle.load(f)
        with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'rb') as f: model_ssl = pickle.load(f)
            
        with rasterio.open(config.MOSAIC_10M_GEOTIFF_PATH) as src:
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
        with rasterio.open(config.PREDICTION_10M_GEOTIFF_PATH, 'w', **profile) as dst: dst.write(prediction_map, 1)
        print("10m SSL prediction map saved.")
    else:
        print("10m prediction map already exists. Skipping inference.")

    # --- Run 1km Inference ---
    if not config.PREDICTION_1KM_GEOTIFF_PATH.exists():
        print("\nRunning 1km Climate model inference...")
        with open(config.MODEL_CLIMATE_ONLY_SAVE_PATH, 'rb') as f: model_clim, feature_names = pickle.load(f)
            
        with rasterio.open(config.CLIMATE_FEATURES_1KM_RASTER_PATH) as src:
            clim_data = src.read(); profile = src.profile; h, w = src.height, src.width
            
        feature_matrix = clim_data.reshape(len(feature_names), -1).T
        valid_mask = ~np.isnan(feature_matrix).any(axis=1)
        valid_features = feature_matrix[valid_mask]
        
        predictions_flat = model_clim.predict(valid_features)
        
        final_predictions = np.full(h * w, np.nan, dtype=np.float32)
        final_predictions[valid_mask] = predictions_flat
        prediction_map = final_predictions.reshape(h, w)
        
        profile.update(count=1, dtype='float32', nodata=np.nan)
        with rasterio.open(config.PREDICTION_1KM_GEOTIFF_PATH, 'w', **profile) as dst: dst.write(prediction_map, 1)
        print("1km climate prediction map saved.")
    else:
        print("1km prediction map already exists. Skipping inference.")
    print("--- All inference complete. ---")

def create_final_visualizations(config: Config):
    print("\n--- PHASE 4: CREATING FINAL VISUALIZATION ---")
    config.FINAL_FIGURE_PATH.parent.mkdir(exist_ok=True, parents=True)

    with rasterio.open(config.PREDICTION_10M_GEOTIFF_PATH) as src:
        pred_10m_data = src.read(1); src_crs = src.crs; src_bounds = src.bounds; src_transform = src.transform
    with rasterio.open(config.PREDICTION_1KM_GEOTIFF_PATH) as src:
        pred_1km_data = src.read(1); pred_1km_bounds = src.bounds

    # Apply WorldCover vegetation mask at display time
    if config.WORLDCOVER_10M_PATH.exists():
        print("Applying WorldCover vegetation mask to 10m predictions...")
        with rasterio.open(config.WORLDCOVER_10M_PATH) as src:
            wc_data = src.read(1)
        veg_mask = np.isin(wc_data, config.WORLDCOVER_VALID_CLASSES)
        pred_10m_data[~veg_mask] = np.nan
        pct_masked = 100 * (~veg_mask).sum() / veg_mask.size
        print(f"  Masked {pct_masked:.1f}% of pixels (non-vegetated)")
    else:
        print("WARNING: No WorldCover mask found. Displaying all pixels.")

    fig, axes = plt.subplots(1, 3, figsize=(24, 12))
    
    valid_10m = pred_10m_data[~np.isnan(pred_10m_data)]
    vmin = max(np.nanmin(valid_10m), 1)
    vmax = np.nanmax(valid_10m)
    norm = LogNorm(vmin=5, vmax=200)
    #norm = PowerNorm(gamma=0.5, vmin=10, vmax=vmax)
    cmap = 'viridis'
    
    # Use light grey for NaN / masked pixels (non-vegetated areas)
    NODATA_COLOR = '#d9d9d9'
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=NODATA_COLOR)

    # --- Basemap Overlay ---
    ax2 = axes[0]
    dst_crs = "EPSG:3857"
    dst_transform, dst_width, dst_height = calculate_default_transform(src_crs, dst_crs, pred_10m_data.shape[1], pred_10m_data.shape[0], *src_bounds)
    dst_data = np.empty((dst_height, dst_width), dtype=pred_10m_data.dtype)
    reproject(source=pred_10m_data, destination=dst_data, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.bilinear)
    dst_extent = (dst_transform.c, dst_transform.c + dst_transform.a * dst_width, dst_transform.f + dst_transform.e * dst_height, dst_transform.f)
    ax2.imshow(dst_data, cmap=cmap, norm=norm, alpha=0.65, extent=dst_extent)
    ctx.add_basemap(ax2, crs=dst_crs, source=ctx.providers.Esri.WorldImagery, attribution=False)
    ax2.set_title('A) Satellite Basemap', fontsize=18); ax2.set_axis_off()
    
    # --- Panel B: 10m SSL Prediction ---
    ax_ssl = axes[1]
    masked_10m = np.ma.masked_invalid(pred_10m_data)
    ax_ssl.imshow(
        masked_10m, cmap=cmap_obj, norm=norm,
        extent=(src_bounds.left, src_bounds.right, src_bounds.bottom, src_bounds.top)
    )
    ax_ssl.set_title('B) 10m Prediction (SSL)', fontsize=18)
    ax_ssl.set_aspect('equal')
    ax_ssl.set_axis_off()

    # --- Panel C: 1km Climate Prediction ---
    ax_clim = axes[2]
    masked_1km = np.ma.masked_invalid(pred_1km_data)
    ax_clim.imshow(
        masked_1km, cmap=cmap_obj, norm=norm,
        extent=(pred_1km_bounds.left, pred_1km_bounds.right, pred_1km_bounds.bottom, pred_1km_bounds.top)
    )
    ax_clim.set_title('C) 1km Prediction', fontsize=18)
    ax_clim.set_aspect('equal')
    ax_clim.set_axis_off()

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    # --- Colorbar with linear ticks ---
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        cax=cbar_ax
    )
    linear_ticks = [5, 10, 25, 50, 75, 100, 150, 200]
    actual_ticks = [t for t in linear_ticks if vmin <= t <= vmax]
    cbar.set_ticks(actual_ticks)
    cbar.set_ticklabels([f"{int(t)}" for t in actual_ticks])
    cbar.set_label('Predicted Mycorrhizal Richness (Species/100m²)', fontsize=14)
    
    # Add explicit nodata patch directly below the colorbar
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.transforms as mtransforms
    
    # Get colorbar position to align below it
    cbar_pos = cbar_ax.get_position()
    # Small square patch centered under colorbar
    patch_size = 0.02
    patch_x = cbar_pos.x0
    patch_y = cbar_pos.y0 - 0.06
    
    patch_ax = fig.add_axes([patch_x, patch_y, patch_size, patch_size])
    patch_ax.set_facecolor(NODATA_COLOR)
    patch_ax.set_xticks([]); patch_ax.set_yticks([])
    for spine in patch_ax.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(0.5)
    
    fig.text(patch_x + patch_size + 0.005, patch_y + patch_size / 2,
             'Non-vegetated\n/ No data', fontsize=10, va='center', ha='left')
    
    plt.savefig(config.FINAL_FIGURE_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"--- Final figure saved to {config.FINAL_FIGURE_PATH} ---")  

    
if __name__ == "__main__":
    cfg = Config()
    train_production_models(cfg)
    acquire_and_prepare_inference_data(cfg)
    run_all_inference(cfg)
    create_final_visualizations(cfg)
    print("\n\nPipeline finished successfully!")