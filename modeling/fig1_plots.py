import subprocess
import pickle
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import rasterio
from rasterio.merge import merge
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from spun_train_patch import CombinedPatchClimateEvaluator, filter_by_biome, lgb
from rasterio.plot import plotting_extent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AssetConfig:
    TRAINING_BIODIVERSITY_CSVS = ["/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv", "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"]
    TRAINING_REPRESENTATIONS_DIR = "/maps-priv/maps/ray25/data/ecm_representations"
    EVALUATOR_SAVE_PATH = Path("./model/evaluator_ssl_only.pkl")
    MODEL_SSL_ONLY_SAVE_PATH = Path("./model/model_ssl_only.pkl")
    
    ASSETS_DIR = Path("figures")
    
    SAMPLES_MAP_PATH = ASSETS_DIR / "sample_distribution_map.png"
    SCATTER_PLOT_PATH = ASSETS_DIR / "performance_scatterplot.png"
    
    ROI_GEOJSON = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {},
          "geometry": {
            "type": "Polygon",
            "coordinates": [[
              [-3.8, 57.2], [-3.2, 57.2], [-3.2, 56.8], [-3.8, 56.8], [-3.8, 57.2]
            ]]
          }
        }
      ]
    }
    
    ROI_DIR = ASSETS_DIR / "roi_data"
    ROI_GEOJSON_PATH = ROI_DIR / "roi.geojson"
    ROI_EMBEDDINGS_DIR = ROI_DIR / "embeddings"
    ROI_EMBEDDINGS_MOSAIC_PATH = ROI_DIR / "embeddings_mosaic.tif"
    
    ROI_SATELLITE_BASEMAP_PATH = ASSETS_DIR / "satellite_basemap.png"
    ROI_PCA_SIGNATURE_PATH = ASSETS_DIR / "embeddings_map.png"
    ROI_PREDICTION_MAP_PATH = ASSETS_DIR / "predicted_biodiversity.png"
    
    SATELLITE_DIM_REDUCTION = 'umap'
    DIM_REDUCTION_COMPONENTS = 256
    USE_BIOME_FILTER = True

def generate_samples_map(config: AssetConfig):
    if config.SAMPLES_MAP_PATH.exists():
        logging.info("Sample distribution map already exists. Skipping.")
        return

    logging.info("Generating sample distribution map...")
    df_list = [pd.read_csv(csv_path) for csv_path in config.TRAINING_BIODIVERSITY_CSVS]
    df = pd.concat(df_list, ignore_index=True).dropna(subset=['latitude', 'longitude'])

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    logging.info("Downloading world basemap...")
    try:
        world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(world_url)
    except Exception as e:
        logging.error(f"Could not download or read the world basemap. Skipping spatial plots. Error: {e}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.plot(ax=ax, color='#e0e0e0', edgecolor='black', linewidth=0.5)
    gdf.plot(ax=ax, marker='o', color='crimson', markersize=25, alpha=0.6)
    
    ax.set_title('Geographic Distribution of Training Samples', fontsize=18)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_xlim(-20, 160)
    ax.set_ylim(0, 80)
    
    fig.savefig(config.SAMPLES_MAP_PATH, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Distribution map saved to {config.SAMPLES_MAP_PATH}")

def generate_performance_scatterplot(config: AssetConfig):
    if config.SCATTER_PLOT_PATH.exists():
        logging.info("Performance scatterplot already exists. Skipping.")
        return

    logging.info("Generating model performance scatterplot...")
    
    logging.info("Loading and preparing training data for scatterplot...")
    df_list = [pd.read_csv(csv_path) for csv_path in config.TRAINING_BIODIVERSITY_CSVS]
    biodiversity_df = pd.concat(df_list, ignore_index=True).dropna(subset=['latitude', 'longitude', 'rarefied', 'sample_id'])
    if config.USE_BIOME_FILTER: biodiversity_df = filter_by_biome(biodiversity_df)

    evaluator_ssl = CombinedPatchClimateEvaluator(climate_data_path=None, use_climate_cache=False, climate_features_cache_dir=None, soil_features_cache_dir=None)
    X_ssl, y_ssl, _, _, _, _ = evaluator_ssl.prepare_dataset(biodiversity_df, representations_dir_path=config.TRAINING_REPRESENTATIONS_DIR, use_satellite=True, use_climate=False, use_soil=False, use_worldcover=False, satellite_dim_reduction=config.SATELLITE_DIM_REDUCTION, dim_reduction_components=config.DIM_REDUCTION_COMPONENTS, random_state=42)
    X_ssl_features_only = X_ssl[:, :-2]

    X_train, X_test, y_train, y_test = train_test_split(X_ssl_features_only, y_ssl, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, n_jobs=-1, device='cuda')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
    
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='1:1 Line')
    
    ax.set_xlabel('True Richness', fontsize=14)
    ax.set_ylabel('Predicted Richness', fontsize=14)
    ax.set_title('Model Performance on Hold-Out Test Set', fontsize=16)
    ax.legend()
    
    fig.savefig(config.SCATTER_PLOT_PATH, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Performance scatterplot saved to {config.SCATTER_PLOT_PATH}")

def generate_roi_visuals(config: AssetConfig):
    logging.info(f"--- Generating Visuals for ROI: {config.ROI_DIR} ---")
    config.ROI_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ROI_GEOJSON_PATH, 'w') as f: json.dump(config.ROI_GEOJSON, f)

    if not config.ROI_EMBEDDINGS_MOSAIC_PATH.exists():
        logging.info("Acquiring and mosaicing Tessera embeddings for ROI...")
        config.ROI_EMBEDDINGS_DIR.mkdir(exist_ok=True)
        if not any(config.ROI_EMBEDDINGS_DIR.iterdir()):
            command = ['geotessera', 'download', '--region-file', str(config.ROI_GEOJSON_PATH), '--year', '2024', '--format', 'tiff', '--output', str(config.ROI_EMBEDDINGS_DIR)]
            subprocess.run(command, check=True)
        
        src_files = [rasterio.open(fp) for fp in config.ROI_EMBEDDINGS_DIR.glob('*.tif')]
        if not src_files: raise FileNotFoundError("No embedding tiles found for the ROI.")
        mosaic, transform = merge(src_files)
        meta = src_files[0].meta.copy()
        meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})
        
        with rasterio.open(config.ROI_EMBEDDINGS_MOSAIC_PATH, "w", **meta) as dest: dest.write(mosaic)
        for src in src_files: src.close()
    else:
        logging.info("Embeddings mosaic already exists.")

    if not config.ROI_PCA_SIGNATURE_PATH.exists():
        logging.info("Generating PCA visualization of embeddings...")
        with rasterio.open(config.ROI_EMBEDDINGS_MOSAIC_PATH) as src:
            embeddings = src.read()
            profile = src.profile
        
        c, h, w = embeddings.shape
        pixels = embeddings.reshape(c, -1).T
        
        valid_mask = ~np.isnan(pixels).any(axis=1)
        valid_pixels = pixels[valid_mask]
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(valid_pixels)
        
        pca_scaled = np.zeros_like(pca_result)
        for i in range(3):
            chan = pca_result[:, i]
            pca_scaled[:, i] = (chan - chan.min()) / (chan.max() - chan.min())
            
        pca_image_flat = np.full((h * w, 3), np.nan, dtype=np.float32)
        pca_image_flat[valid_mask] = pca_scaled
        pca_image = pca_image_flat.reshape(h, w, 3)

        plt.imsave(config.ROI_PCA_SIGNATURE_PATH, pca_image, dpi=600)
        logging.info(f"PCA signature image saved to {config.ROI_PCA_SIGNATURE_PATH}")

    if not config.ROI_PREDICTION_MAP_PATH.exists():
        logging.info("Generating predicted biodiversity map for ROI...")
        
        from skimage.util import view_as_windows
        with open(config.EVALUATOR_SAVE_PATH, 'rb') as f: evaluator = pickle.load(f)
        with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'rb') as f: model = pickle.load(f)
        
        with rasterio.open(config.ROI_EMBEDDINGS_MOSAIC_PATH) as src:
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
        predictions_flat = model.predict(reduced_pixels)
        
        final_predictions = np.full(h * w, np.nan, dtype=np.float32)
        final_predictions[valid_mask] = predictions_flat
        prediction_map = final_predictions.reshape(h, w)
        
        pred_geotiff_path = config.ROI_DIR / "prediction.tif"
        profile.update(count=1, dtype='float32', nodata=np.nan)
        with rasterio.open(pred_geotiff_path, 'w', **profile) as dst: dst.write(prediction_map, 1)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(prediction_map, cmap='viridis', vmin=0, vmax=np.nanpercentile(prediction_map, 98))
        ax.set_axis_off()
        plt.colorbar(im, ax=ax, label="Predicted Richness", shrink=0.7)
        fig.savefig(config.ROI_PREDICTION_MAP_PATH, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        logging.info(f"Prediction map image saved to {config.ROI_PREDICTION_MAP_PATH}")

    if not config.ROI_SATELLITE_BASEMAP_PATH.exists():
        logging.info("Generating satellite basemap for ROI...")
        pred_geotiff_path = config.ROI_DIR / "prediction.tif"
        with rasterio.open(pred_geotiff_path) as src:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_axis_off()
            ax.imshow(np.zeros(src.shape), extent=plotting_extent(src), alpha=0)
            ctx.add_basemap(ax, crs=src.crs.to_string(), source=ctx.providers.Esri.WorldImagery, attribution="")
            fig.savefig(config.ROI_SATELLITE_BASEMAP_PATH, dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            logging.info(f"Satellite basemap saved to {config.ROI_SATELLITE_BASEMAP_PATH}")

if __name__ == "__main__":
    config = AssetConfig()
    config.ASSETS_DIR.mkdir(exist_ok=True)
    
    generate_samples_map(config)
    generate_performance_scatterplot(config)
    generate_roi_visuals(config)

    logging.info("\n--- Asset generation complete. ---")
    logging.info(f"All files saved in: {config.ASSETS_DIR.resolve()}")