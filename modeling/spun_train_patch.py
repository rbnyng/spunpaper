import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import json
import richdem as rd 
import geopandas
from shapely.geometry import Point
import argparse
import sys 

# --- Model Imports ---
import xgboost as xgb
import lightgbm as lgb

import time
import pickle
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BiomeFilter:
    ECOREGIONS_URL = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "ecoregions_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ecoregions_gdf = None

    def _download_ecoregions(self) -> Path:
        zip_path = self.cache_dir / "Ecoregions2017.zip"
        extract_dir = self.cache_dir / "Ecoregions2017"

        # Check if already downloaded and extracted
        shapefile_path = extract_dir / "Ecoregions2017.shp"
        if shapefile_path.exists():
            logging.info(f"Using cached ecoregions data at {shapefile_path}")
            return extract_dir

        # Download the shapefile
        logging.info(f"Downloading RESOLVE Ecoregions from {self.ECOREGIONS_URL}...")
        try:
            response = requests.get(self.ECOREGIONS_URL, stream=True, timeout=300)
            response.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"Downloaded to {zip_path}")

            # Extract the zipfile
            logging.info("Extracting shapefile...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            logging.info(f"Extracted to {extract_dir}")
            return extract_dir

        except Exception as e:
            logging.error(f"Failed to download/extract ecoregions: {e}")
            raise

    def load_ecoregions(self) -> geopandas.GeoDataFrame:
        if self.ecoregions_gdf is not None:
            return self.ecoregions_gdf

        extract_dir = self._download_ecoregions()
        shapefile_path = extract_dir / "Ecoregions2017.shp"

        logging.info(f"Loading ecoregions shapefile from {shapefile_path}...")
        try:
            self.ecoregions_gdf = geopandas.read_file(shapefile_path)
            logging.info(f"Loaded {len(self.ecoregions_gdf)} ecoregions")

            # Log available columns for reference
            logging.info(f"Available columns: {self.ecoregions_gdf.columns.tolist()}")

            # Ensure CRS is WGS84 (EPSG:4326) for lat/lon matching
            if self.ecoregions_gdf.crs is None:
                logging.warning("Ecoregions GeoDataFrame has no CRS, assuming EPSG:4326")
                self.ecoregions_gdf.set_crs(epsg=4326, inplace=True)
            elif self.ecoregions_gdf.crs.to_epsg() != 4326:
                logging.info(f"Reprojecting from {self.ecoregions_gdf.crs} to EPSG:4326")
                self.ecoregions_gdf = self.ecoregions_gdf.to_crs(epsg=4326)

            return self.ecoregions_gdf

        except Exception as e:
            logging.error(f"Failed to load ecoregions shapefile: {e}")
            raise

    def assign_biomes(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Assigning biomes to samples...")

        # Load ecoregions if not already loaded
        if self.ecoregions_gdf is None:
            self.load_ecoregions()

        # Create GeoDataFrame from sample locations
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        samples_gdf = geopandas.GeoDataFrame(
            df.copy(),
            geometry=geometry,
            crs="EPSG:4326"
        )

        # Spatial join to assign biomes
        # Use 'left' join to keep all samples, even if they don't intersect any ecoregion
        logging.info("Performing spatial join with ecoregions...")
        joined = geopandas.sjoin(
            samples_gdf,
            self.ecoregions_gdf[['BIOME_NAME', 'BIOME_NUM', 'geometry']],
            how='left',
            predicate='within'
        )

        # Drop the geometry column and index_right from the join
        result_df = pd.DataFrame(joined.drop(columns=['geometry', 'index_right']))

        # Log statistics
        total_samples = len(result_df)
        assigned_samples = result_df['BIOME_NAME'].notna().sum()
        logging.info(f"Assigned biomes to {assigned_samples}/{total_samples} samples")

        if assigned_samples < total_samples:
            logging.warning(
                f"{total_samples - assigned_samples} samples could not be assigned to a biome "
                "(they may be in areas not covered by the ecoregions dataset)"
            )

        return result_df

    def filter_by_biome_outliers(
        self,
        df: pd.DataFrame,
        richness_column: str = 'rarefied',
        iqr_multiplier: float = 5.0,
        remove_unassigned: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(
            f"Filtering samples by biome outliers "
            f"(threshold: median + {iqr_multiplier} * IQR)..."
        )

        # Ensure biomes are assigned
        if 'BIOME_NAME' not in df.columns or 'BIOME_NUM' not in df.columns:
            logging.info("Biome columns not found, assigning biomes first...")
            df = self.assign_biomes(df)

        initial_count = len(df)

        # Handle samples without biome assignment
        unassigned_mask = df['BIOME_NAME'].isna()
        unassigned_count = unassigned_mask.sum()

        if unassigned_count > 0:
            if remove_unassigned:
                logging.info(f"Removing {unassigned_count} samples without biome assignment")
                df = df[~unassigned_mask].copy()
            else:
                logging.warning(
                    f"Keeping {unassigned_count} samples without biome assignment "
                    "(they will not be filtered by biome statistics)"
                )

        # Calculate biome-level statistics
        logging.info("Calculating biome-level statistics...")
        biome_stats = df.groupby('BIOME_NAME')[richness_column].agg([
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std')
        ]).reset_index()

        # Calculate IQR and threshold
        biome_stats['iqr'] = biome_stats['q75'] - biome_stats['q25']
        biome_stats['upper_threshold'] = (
            biome_stats['median'] + iqr_multiplier * biome_stats['iqr']
        )

        # Log biome statistics
        logging.info("\nBiome statistics:")
        for _, row in biome_stats.iterrows():
            logging.info(
                f"  {row['BIOME_NAME']}: "
                f"n={int(row['count'])}, "
                f"median={row['median']:.2f}, "
                f"IQR={row['iqr']:.2f}, "
                f"threshold={row['upper_threshold']:.2f}"
            )

        # Merge threshold back to main dataframe
        df = df.merge(
            biome_stats[['BIOME_NAME', 'upper_threshold']],
            on='BIOME_NAME',
            how='left'
        )

        # Identify outliers
        # For unassigned samples (if kept), upper_threshold will be NaN, so they won't be filtered
        outlier_mask = df[richness_column] > df['upper_threshold']
        outliers = df[outlier_mask].copy()
        filtered_df = df[~outlier_mask].copy()

        # Clean up temporary column
        filtered_df = filtered_df.drop(columns=['upper_threshold'])
        if len(outliers) > 0:
            outliers = outliers.drop(columns=['upper_threshold'])

        removed_count = len(outliers)
        logging.info(
            f"\nFiltering complete: "
            f"Removed {removed_count} outliers "
            f"({removed_count/initial_count*100:.1f}% of initial samples)"
        )
        logging.info(f"Remaining samples: {len(filtered_df)}/{initial_count}")

        # Log outliers by biome
        if removed_count > 0:
            outliers_by_biome = outliers.groupby('BIOME_NAME').size()
            logging.info("\nOutliers removed per biome:")
            for biome, count in outliers_by_biome.items():
                logging.info(f"  {biome}: {count}")

        return filtered_df, outliers

    def apply_filter(
        self,
        df: pd.DataFrame,
        richness_column: str = 'rarefied',
        iqr_multiplier: float = 5.0,
        remove_unassigned: bool = True,
        save_removed: Optional[Path] = None
    ) -> pd.DataFrame:
        # Assign biomes
        df_with_biomes = self.assign_biomes(df)

        # Filter outliers
        filtered_df, removed_df = self.filter_by_biome_outliers(
            df_with_biomes,
            richness_column=richness_column,
            iqr_multiplier=iqr_multiplier,
            remove_unassigned=remove_unassigned
        )

        # Optionally save removed samples
        if save_removed and len(removed_df) > 0:
            removed_df.to_csv(save_removed, index=False)
            logging.info(f"Saved {len(removed_df)} removed samples to {save_removed}")

        return filtered_df

def filter_by_biome(
    df: pd.DataFrame,
    richness_column: str = 'rarefied',
    iqr_multiplier: float = 5.0,
    remove_unassigned: bool = True,
    cache_dir: Optional[Path] = None,
    save_removed: Optional[Path] = None
) -> pd.DataFrame:
    biome_filter = BiomeFilter(cache_dir=cache_dir)
    return biome_filter.apply_filter(
        df,
        richness_column=richness_column,
        iqr_multiplier=iqr_multiplier,
        remove_unassigned=remove_unassigned,
        save_removed=save_removed
    )

class ClimateExtractor:
    def __init__(self, climate_data_path, use_cache=True):
        self.climate_data_path = Path(climate_data_path)
        self.use_cache = use_cache
        
        self.monthly_vars = ['prec', 'tavg', 'tmin', 'tmax', 'srad', 'vapr', 'wind']
        self.static_vars = [f'bio_{i}' for i in range(1, 20)] + ['elev', 'slope', 'aspect']
        self.cache_file = Path('/scratch/ray25/climate_metadata_cache_v3.pkl')
        
        self.climate_metadata = {} 
        self.data_loaded = False
        
        self._generate_topography_layers()
        self._load_climate_metadata()

    def _generate_topography_layers(self):
        logging.info("Checking for topography layers (slope, aspect)...")
        elev_path = self.climate_data_path / "wc2.1_30s_elev.tif"
        slope_path = self.climate_data_path / "wc2.1_30s_slope.tif"
        aspect_path = self.climate_data_path / "wc2.1_30s_aspect.tif"

        if not elev_path.exists():
            logging.warning(f"Elevation file not found at {elev_path}. Cannot generate topography layers.")
            return

        # Check if both files already exist
        if slope_path.exists() and aspect_path.exists():
            logging.info("Slope and Aspect layers already exist. Skipping generation.")
            return

        try:
            logging.info("Generating slope and aspect layers... This may take a few minutes.")
            
            # Load the elevation data using richdem
            elev_dem = rd.LoadGDAL(str(elev_path))
            
            # Calculate slope and aspect
            slope = rd.TerrainAttribute(elev_dem, attrib='slope_riserun')
            aspect = rd.TerrainAttribute(elev_dem, attrib='aspect')
            
            # Save the new layers as GeoTIFFs, preserving the geo-referencing info
            if not slope_path.exists():
                logging.info(f"Saving slope layer to {slope_path}")
                rd.SaveGDAL(str(slope_path), slope)
            
            if not aspect_path.exists():
                logging.info(f"Saving aspect layer to {aspect_path}")
                rd.SaveGDAL(str(aspect_path), aspect)
            
            logging.info("Topography layers generated successfully.")

        except ImportError:
            logging.error("`richdem` library not found. Please run `pip install richdem`.")
        except Exception as e:
            logging.error(f"An error occurred during topography generation: {e}", exc_info=True)

    def _load_climate_metadata(self):
        if self.use_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.climate_metadata = pickle.load(f)
                self.data_loaded = True
                logging.info(f"Successfully loaded climate metadata from cache: {self.cache_file}")
                return
            except Exception as e:
                logging.warning(f"Failed to load metadata cache: {e}. Re-generating.")
        
        logging.info("Scanning all climate data files for metadata...")
        temp_metadata = {}
        # Loop for monthly variables
        for var in self.monthly_vars:
            monthly_meta = {}
            for month in range(1, 13):
                filepath = self.climate_data_path / f"wc2.1_30s_{var}_{month:02d}.tif"
                if filepath.exists():
                    try:
                        with rasterio.open(filepath) as src:
                            monthly_meta[month] = {'path': str(filepath), 'transform': src.transform, 'nodata': src.nodata, 'bounds': src.bounds, 'shape': src.shape}
                    except Exception as e: logging.error(f"Error reading metadata for {filepath}: {e}")
            if monthly_meta: temp_metadata[var] = monthly_meta
        # Loop for static variables (now includes slope and aspect)
        for var in self.static_vars:
            filepath = self.climate_data_path / f"wc2.1_30s_{var}.tif"
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        temp_metadata[var] = {'path': str(filepath), 'transform': src.transform, 'nodata': src.nodata, 'bounds': src.bounds, 'shape': src.shape}
                except Exception as e: logging.error(f"Error reading metadata for {filepath}: {e}")
        
        self.climate_metadata = temp_metadata
        if self.climate_metadata:
            self.data_loaded = True
            if self.use_cache:
                with open(self.cache_file, 'wb') as f: pickle.dump(self.climate_metadata, f)
        else: logging.error("Failed to load any climate metadata!")

    def extract_climate_features(self, lat, lon):
        features = {}
        if not (-90 <= lat <= 90 and -180 <= lon <= 180): return {}
        # Process Monthly Variables (code is unchanged)
        for var in self.monthly_vars:
            if var not in self.climate_metadata: continue
            monthly_values = []
            for month in range(1, 13):
                month_meta = self.climate_metadata[var].get(month)
                if not month_meta:
                    monthly_values.append(np.nan)
                    continue
                try:
                    if not (month_meta['bounds'].left <= lon <= month_meta['bounds'].right and month_meta['bounds'].bottom <= lat <= month_meta['bounds'].top):
                        monthly_values.append(np.nan); continue
                    row_idx, col_idx = rowcol(month_meta['transform'], lon, lat)
                    if not (0 <= row_idx < month_meta['shape'][0] and 0 <= col_idx < month_meta['shape'][1]):
                        monthly_values.append(np.nan); continue
                    with rasterio.open(month_meta['path']) as src:
                        pixel_val = src.read(1, window=Window(col_idx, row_idx, 1, 1))[0, 0]
                    monthly_values.append(float(pixel_val) if pixel_val != month_meta['nodata'] else np.nan)
                except Exception: monthly_values.append(np.nan)
            valid_values = [v for v in monthly_values if not np.isnan(v)]
            if valid_values:
                features[f"{var}_annual_mean"] = np.mean(valid_values)
                features[f"{var}_annual_min"], features[f"{var}_annual_max"] = np.min(valid_values), np.max(valid_values)
        
        # Process Static Variables (this logic now automatically handles slope and aspect)
        for var in self.static_vars:
            if var not in self.climate_metadata: continue
            var_meta = self.climate_metadata[var]
            pixel_val = np.nan
            try:
                if (var_meta['bounds'].left <= lon <= var_meta['bounds'].right and var_meta['bounds'].bottom <= lat <= var_meta['bounds'].top):
                    row_idx, col_idx = rowcol(var_meta['transform'], lon, lat)
                    if (0 <= row_idx < var_meta['shape'][0] and 0 <= col_idx < var_meta['shape'][1]):
                        with rasterio.open(var_meta['path']) as src:
                            val = src.read(1, window=Window(col_idx, row_idx, 1, 1))[0, 0]
                        if val != var_meta['nodata']: pixel_val = float(val)
            except Exception: pass
            features[var] = pixel_val
        return features
        
class CombinedPatchClimateEvaluator:
    def __init__(self, climate_data_path, use_climate_cache, climate_features_cache_dir, soil_features_cache_dir):
        self.climate_extractor = None
        if climate_data_path:
            self.climate_extractor = ClimateExtractor(climate_data_path, use_cache=use_climate_cache)
        self.climate_features_cache_dir = None
        if climate_features_cache_dir:
            self.climate_features_cache_dir = Path(climate_features_cache_dir)
            self.climate_features_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store dimensionality reduction models for consistent transform
        self.dim_reduction_model = None
        self.scaler = None
        
        self.soil_features_cache_dir = soil_features_cache_dir
        if self.soil_features_cache_dir:
            logging.info(f"Using pre-processed soil feature cache at: {self.soil_features_cache_dir}")
            
    def apply_dimensionality_reduction(self, X_satellite, method='none', n_components=None, random_state=42, 
                                     is_training=True):
        if method == 'none' or X_satellite is None:
            return X_satellite, [f"patch_{i}" for i in range(X_satellite.shape[1])] if X_satellite is not None else []
        
        if is_training:
            # Fit the scaler and dimensionality reduction model
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_satellite)
            
            if n_components is None:
                # Auto-select number of components
                if method == 'pca':
                    n_components = min(50, X_satellite.shape[1] // 10, X_satellite.shape[0] // 2)
                elif method == 'umap':
                    n_components = min(20, X_satellite.shape[1] // 20, X_satellite.shape[0] // 5)
                n_components = max(2, n_components)  # Ensure at least 2 components
            
            logging.info(f"Applying {method.upper()} with {n_components} components to satellite features")
            
            if method == 'pca':
                self.dim_reduction_model = PCA(n_components=n_components, random_state=random_state)
                X_reduced = self.dim_reduction_model.fit_transform(X_scaled)
                feature_names = [f"pca_{i+1}" for i in range(n_components)]
                
                # Log explained variance for PCA
                explained_var = self.dim_reduction_model.explained_variance_ratio_
                total_var = np.sum(explained_var)
                logging.info(f"PCA explained variance: {total_var:.3f} ({explained_var[:5]}...)")
                
            elif method == 'umap':
                # UMAP parameters optimized for this use case
                self.dim_reduction_model = umap.UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=min(15, X_satellite.shape[0] // 3),
                    min_dist=0.1,
                    metric='euclidean'
                )
                X_reduced = self.dim_reduction_model.fit_transform(X_scaled)
                feature_names = [f"umap_{i+1}" for i in range(n_components)]
                
        else:
            # Transform test data using fitted models
            if self.scaler is None or self.dim_reduction_model is None:
                raise ValueError("Models not fitted. Cannot transform test data.")
            
            X_scaled = self.scaler.transform(X_satellite)
            X_reduced = self.dim_reduction_model.transform(X_scaled)
            feature_names = ([f"pca_{i+1}" for i in range(X_reduced.shape[1])] if method == 'pca' 
                           else [f"umap_{i+1}" for i in range(X_reduced.shape[1])])
        
        return X_reduced, feature_names
    
    def prepare_dataset(self, 
                        biodiversity_df: pd.DataFrame, 
                        representations_dir_path: Optional[Union[str, Path]] = None,
                        worldcover_features_path: Optional[Union[str, Path]] = None,
                        use_satellite: bool = True,
                        use_climate: bool = True,
                        use_soil: bool = True,
                        use_worldcover: bool = True,
                        satellite_dim_reduction: str = 'none',
                        dim_reduction_components: Optional[int] = None,
                        random_state: int = 42
                        ) -> Tuple[
                            Optional[np.ndarray], Optional[np.ndarray], Optional[List[Dict]], 
                            Optional[List[Dict]], Optional[Tuple[int, int, int]], Optional[List[str]]]:
        all_skipped_locations = []
        biodiversity_df['sample_id'] = biodiversity_df['sample_id'].astype(str)
        current_merged_df = biodiversity_df[['sample_id', 'latitude', 'longitude', 'rarefied']].copy()
        
        X_patch_flat, patch_feature_names_list, patch_dims = None, [], None
        X_climate, climate_feature_names = None, []
        X_soil, soil_feature_names = None, []
                
        if use_satellite:
            if not representations_dir_path:
                logging.error("use_satellite is True, but representations_dir_path is not provided.")
                return None, None, [], [], None, None
            logging.info(f"Loading PATCH representations from directory: {representations_dir_path}")
            repr_dir = Path(representations_dir_path)
            if not repr_dir.is_dir():
                logging.error(f"Representations directory not found: {repr_dir}")
                return None, None, [], [{'id': 'N/A', 'reason': f'Directory not found: {repr_dir}', 'stage': 'patch_load'}], None, None

            all_repr_dict, skipped_locations_from_npy, first_valid_patch_read = {}, [], False
            npy_files = list(repr_dir.glob('*.npy'))
            logging.info(f"Found {len(npy_files)} .npy files to process.")

            for npy_path in tqdm(npy_files, desc="Loading .npy representations"):
                try:
                    # Extract the sample_id from the filename
                    filename_stem = npy_path.stem
                    parts = filename_stem.split('_')
                    if len(parts) >= 2:
                        sample_id = parts[1] 
                    else:
                        sample_id = filename_stem

                    patch_tensor = np.load(npy_path)
                    if isinstance(patch_tensor, np.ndarray) and patch_tensor.ndim == 3:
                        P1, P2, C = patch_tensor.shape
                        if not first_valid_patch_read:
                            patch_dims = (P1, P2, C)
                            logging.info(f"Detected patch dimensions (P1, P2, C): {patch_dims} from {npy_path.name}")
                            first_valid_patch_read = True
                        if patch_dims and (P1, P2, C) != patch_dims:
                            skipped_locations_from_npy.append({'id': sample_id, 'reason': f'Inconsistent patch dimensions', 'stage': 'patch_load'})
                            continue
                        all_repr_dict[sample_id] = patch_tensor
                    else:
                        skipped_locations_from_npy.append({'id': sample_id, 'reason': 'Invalid format', 'stage': 'patch_load'})
                except Exception as e:
                    skipped_locations_from_npy.append({'id': npy_path.name, 'reason': str(e), 'stage': 'patch_load'})
            
            all_skipped_locations.extend(skipped_locations_from_npy)
            if not all_repr_dict:
                logging.error("No valid patch representations found.")
                return None, None, [], all_skipped_locations, None, None

            repr_df = pd.DataFrame([{'sample_id': sid, 'representation_patch': patch} for sid, patch in all_repr_dict.items()])
            logging.info(f"Total unique valid patch representations collected: {len(repr_df)}")
            current_merged_df = pd.merge(current_merged_df, repr_df, on='sample_id', how='inner')
            
            if current_merged_df.empty:
                logging.error("No samples common between bio data and patch representations.")
                return None, None, [], all_skipped_locations, patch_dims, None
            logging.info(f"Successfully merged {len(current_merged_df)} samples with patch data.")
            
            patch_stack = np.stack(current_merged_df['representation_patch'].values).astype(np.float32)
            X_patch_flat = patch_stack.reshape(len(current_merged_df), -1)
            
            if np.isnan(X_patch_flat).any():
                logging.warning("NaNs found in PATCH features. Imputing with column means.")
                X_patch_flat = np.nan_to_num(X_patch_flat, nan=np.nanmean(X_patch_flat, axis=0))
            
            # Apply dimensionality reduction to satellite features
            X_patch_reduced, patch_feature_names_list = self.apply_dimensionality_reduction(
                X_patch_flat, method=satellite_dim_reduction, 
                n_components=dim_reduction_components, 
                random_state=random_state, 
                is_training=True
            )
            X_patch_flat = X_patch_reduced
        
        if use_climate:
            if not self.climate_extractor or not self.climate_extractor.data_loaded:
                logging.error("ClimateExtractor not ready. Cannot extract climate features.")
                return None, None, [], all_skipped_locations, patch_dims, None

            all_climate_features = []
            skipped_climate_extraction = []
            points_for_climate = current_merged_df[['latitude', 'longitude', 'sample_id']].copy()
            
            logging.info(f"Processing {len(points_for_climate)} locations for climate features (using cache)...")
            
            for _, row in tqdm(points_for_climate.iterrows(), total=len(points_for_climate), desc="Climate Features"):
                sample_id, lat, lon = row['sample_id'], row['latitude'], row['longitude']
                cache_file = self.climate_features_cache_dir / f"{sample_id}.npy" if self.climate_features_cache_dir else None
                
                try:
                    # Check cache first
                    if cache_file and cache_file.exists():
                        climate_feats_dict = np.load(cache_file, allow_pickle=True).item()
                    else:
                        # If not in cache, extract fresh
                        climate_feats_dict = self.climate_extractor.extract_climate_features(lat, lon)
                        # And save to cache if successful
                        if climate_feats_dict and cache_file:
                            np.save(cache_file, climate_feats_dict)
                    
                    if climate_feats_dict:
                        climate_feats_dict['sample_id'] = sample_id
                        all_climate_features.append(climate_feats_dict)
                    else:
                        skipped_climate_extraction.append({'id': sample_id, 'reason': 'No climate features extracted'})
                except Exception as e:
                    logging.warning(f"Error processing climate for {sample_id}: {e}")
                    skipped_climate_extraction.append({'id': sample_id, 'reason': str(e)})

            all_skipped_locations.extend([dict(item, stage='climate_extraction') for item in skipped_climate_extraction])
            
            if not all_climate_features:
                logging.error("No climate features could be extracted or loaded for any point.")
                if not use_satellite: return None, None, [], all_skipped_locations, patch_dims, None
                logging.warning("Proceeding with SATELLITE-ONLY data as no climate features were found.")
            else:
                extracted_climate_df = pd.DataFrame(all_climate_features)
                climate_feature_names = [col for col in extracted_climate_df.columns if col != 'sample_id']
                logging.info(f"Successfully loaded/extracted {len(climate_feature_names)} climate features for {len(extracted_climate_df)} points.")
                
                # Merge climate data back
                current_merged_df = pd.merge(current_merged_df, extracted_climate_df, on='sample_id', how='inner')
                if current_merged_df.empty:
                    logging.error("No samples remaining after merging with climate data.")
                    return None, None, [], all_skipped_locations, patch_dims, None
                logging.info(f"{len(current_merged_df)} samples remain after climate merge.")
                
                X_climate = current_merged_df[climate_feature_names].values.astype(np.float32)
                if np.isnan(X_climate).any():
                    logging.warning("NaNs found in CLIMATE features. Imputing with column means.")
                    X_climate = np.nan_to_num(X_climate, nan=np.nanmean(X_climate, axis=0))
                    
        if use_soil and self.soil_features_cache_dir:
            logging.info("Loading soil features from pre-processed cache")
            all_soil_features = []
            points_for_soil = current_merged_df[['latitude', 'longitude', 'sample_id']].copy()
            for _, row in tqdm(points_for_soil.iterrows(), total=len(points_for_soil), desc="Loading Soil Features"):
                cache_file = self.soil_features_cache_dir / f"{row['sample_id']}.npy"
                if cache_file.exists():
                    all_soil_features.append(np.load(cache_file, allow_pickle=True).item())
                else:
                    all_soil_features.append({})
            soil_df = pd.DataFrame(all_soil_features).set_index(points_for_soil.index)

            # ONE-HOT ENCODING FOR WRB
            wrb_col = 'soil_wrb'
            if wrb_col in soil_df.columns:
                logging.info(f"Performing one-hot encoding on '{wrb_col}' column...")
                
                # Ensure the column doesn't have NaNs from failed lookups, fill with a placeholder if needed
                # Using -1 for unknown
                soil_df[wrb_col].fillna(-1, inplace=True)
                soil_df = pd.get_dummies(soil_df, columns=[wrb_col], prefix='wrb_class')
                
            else:
                logging.warning(f"Column '{wrb_col}' not found. Skipping one-hot encoding.")
            soil_feature_names = soil_df.columns.tolist()
            X_soil = soil_df.values.astype(np.float32)
            current_merged_df = pd.concat([current_merged_df, soil_df], axis=1)            
        
        X_worldcover, worldcover_feature_names = None, []
        if use_worldcover and worldcover_features_path and worldcover_features_path.exists():
            logging.info(f"Loading and integrating WorldCover features from: {worldcover_features_path}")
            try:
                wc_df = pd.read_csv(worldcover_features_path, dtype={'sample_id': str})
                if wc_df['sample_id'].duplicated().any():
                    wc_df.drop_duplicates(subset=['sample_id'], keep='first', inplace=True)
                    
                # One-Hot Encoding
                logging.info(f"Performing one-hot encoding on 'land_cover_class'...")
                wc_dummies = pd.get_dummies(wc_df['land_cover_class'], prefix='wc_class', dtype=float)
                
                # Re-join the sample_id to the new dummy columns
                wc_encoded_df = pd.concat([wc_df['sample_id'], wc_dummies], axis=1)
                
                # Merge into the main dataframe
                initial_rows = len(current_merged_df)
                current_merged_df = pd.merge(current_merged_df, wc_encoded_df, on='sample_id', how='inner')
                logging.info(f"{len(current_merged_df)} of {initial_rows} records remain after merging WorldCover data.")
                
                # Extract the numpy array and feature names for final assembly
                worldcover_feature_names = wc_dummies.columns.tolist()
                X_worldcover = current_merged_df[worldcover_feature_names].values.astype(np.float32)

            except Exception as e:
                logging.error(f"Failed to load or process WorldCover data: {e}", exc_info=True)
                X_worldcover = None # Ensure it's None on failure
                
        elif use_worldcover:
            logging.warning(f"WorldCover data requested but file not found at: {worldcover_features_path}")
            
        # Finalize Features and Targets
        if current_merged_df.empty:
            logging.error("No data remaining after all merge steps.")
            return None, None, [], all_skipped_locations, patch_dims, None

        if use_satellite and X_patch_flat is not None:
            if 'representation_patch' in current_merged_df.columns:
                patch_stack_final = np.stack(current_merged_df['representation_patch'].values).astype(np.float32)
                X_patch_flat_final = patch_stack_final.reshape(len(current_merged_df), -1)
                if np.isnan(X_patch_flat_final).any(): 
                    X_patch_flat_final = np.nan_to_num(X_patch_flat_final)
                
                # Apply the same dimensionality reduction (already fitted)
                if satellite_dim_reduction != 'none':
                    X_patch_flat, _ = self.apply_dimensionality_reduction(
                        X_patch_flat_final, method=satellite_dim_reduction, 
                        is_training=False
                    )
                else:
                    X_patch_flat = X_patch_flat_final
            else:
                logging.error("Patch data lost in final merge. Features will be missing.")
                X_patch_flat = None 
                patch_feature_names_list = []

        final_X_list, all_feature_names = [], []
        if X_patch_flat is not None and use_satellite:
            final_X_list.append(X_patch_flat)
            all_feature_names.extend(patch_feature_names_list)
        if X_climate is not None and use_climate and len(climate_feature_names) > 0:
            final_X_list.append(X_climate)
            all_feature_names.extend(climate_feature_names)
        if X_soil is not None:
            final_X_list.append(X_soil)
            all_feature_names.extend(soil_feature_names)
        if X_worldcover is not None and use_worldcover:
            final_X_list.append(X_worldcover)
            all_feature_names.extend(worldcover_feature_names)
        if not final_X_list:
            logging.error("No features to form the final dataset.")
            return None, None, [], all_skipped_locations, patch_dims, None
            
        # Add lat-lon as features
        X_coords = current_merged_df[['latitude', 'longitude']].values.astype(np.float32)
        final_X_list.append(X_coords)
        all_feature_names.extend(['latitude', 'longitude'])
        
        X_combined = np.hstack(final_X_list)
        y_combined = current_merged_df['rarefied'].values.astype(np.float32)
        processed_locations = current_merged_df[['sample_id', 'latitude', 'longitude']].to_dict('records')

        logging.info(f"Final COMBINED feature shape (X): {X_combined.shape}")
        if use_satellite and X_patch_flat is not None: 
            logging.info(f"  Satellite features: {X_patch_flat.shape[1]} (after {satellite_dim_reduction})")
        if use_climate and X_climate is not None: 
            logging.info(f"  Climate features: {X_climate.shape[1]}")
        if use_soil and X_soil is not None: 
            logging.info(f"  Soil features: {X_soil.shape[1]}")
        if use_worldcover and X_worldcover is not None: 
            logging.info(f"  Land cover features: {X_worldcover.shape[1]}")

        logging.info(f"  Lat-lon features: {X_coords.shape[1]}")

        logging.info(f"Final target shape (y): {y_combined.shape}")
            
        return X_combined, y_combined, processed_locations, all_skipped_locations, patch_dims, all_feature_names

    def train_and_evaluate(self, X, y, location_info, all_feature_names, random_seed=42, model_name='rf', model_params=None):
        if X is None or y is None or len(X) == 0: return {'error': 'Empty training data'}
        model_params = model_params or {}
        
        # X_train, X_test, y_train, y_test, _, info_test = train_test_split(
            # X, y, location_info, test_size=0.2, random_state=random_seed
        # )
        
        # Three-way data split (train/validation/test)
        # First, split into training+validation (80%) and test (20%)
        X_train_val, X_test, y_train_val, y_test, _, info_test = train_test_split(
            X, y, location_info, test_size=0.2, random_state=random_seed
        )
        
        # Then, split the 80% into training (70% of total) and validation (10% of total)
        # The new test_size is 0.1 / 0.8 = 0.125
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=random_seed
        )

        logging.info(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.")
        
        model = None
        logging.info(f"Training with model: {model_name.upper()}")

        try:
            if model_name == 'lightgbm':
                lgbm_defaults = {'random_state': random_seed, 'n_estimators': 1000, 'learning_rate': 0.05, 'n_jobs': -1}
                model = lgb.LGBMRegressor(**{**lgbm_defaults, **model_params}, device='cuda')
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])

            elif model_name == 'xgboost':
                xgb_defaults = {'random_state': random_seed, 'n_estimators': 1000, 'learning_rate': 0.05, 'n_jobs': -1}
                model = xgb.XGBRegressor(**{**xgb_defaults, **model_params}, device="cuda")
                model.fit(X_train, y_train, verbose=False, eval_set=[(X_val, y_val)], early_stopping_rounds=15)

            else: # Default to RandomForest
                rf_defaults = {'random_state': random_seed, 'n_jobs': -1, 'n_estimators': 100}
                model = RandomForestRegressor(**{**rf_defaults, **model_params})
                model.fit(X_train, y_train)

            test_pred = model.predict(X_test)
            # Standard metrics
            r2 = r2_score(y_test, test_pred)
            mae = mean_absolute_error(y_test, test_pred)
            medae = median_absolute_error(y_test, test_pred)
            me = np.mean(test_pred - y_test)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Calculate R2 on the central 98% of the data to exclude extreme outliers
            error_df = pd.DataFrame({'actual': y_test, 'pred': test_pred})
            error_df['abs_error'] = np.abs(error_df['actual'] - error_df['pred'])
            
            # Find the error value at the 98th percentile
            error_threshold = error_df['abs_error'].quantile(0.98)
            
            # Filter out the top 2% of largest errors
            filtered_df = error_df[error_df['abs_error'] <= error_threshold]
            r2_filtered = r2_score(filtered_df['actual'], filtered_df['pred'])
        
            return {
                'model_type': model_name,
                'random_seed': random_seed,
                'test_stats': {
                    'r2': r2, 
                    'rmse': rmse,
                    'mae': mae,
                    'me': me,
                    'medae': medae,
                    'r2_filtered_98_percent': r2_filtered
                },
                'feature_importance': model.feature_importances_.tolist(),
                'error_analysis_data': {
                    'y_test': y_test.tolist(),
                    'test_pred': test_pred.tolist(),
                    'info_test': info_test
                }
            }
        except Exception as e:
            logging.error(f"Failed during {model_name} training/evaluation: {e}", exc_info=True)
            return {'error': f"{model_name} failed: {e}"}
            
def analyze_and_save_feature_importance(all_run_results, all_feature_names, run_dir):
    logging.info("--- Starting Feature Importance Analysis ---")
    
    successful_runs = [r for r in all_run_results if 'error' not in r and 'feature_importance' in r]
    if not successful_runs:
        logging.warning("No successful runs with feature importance found. Skipping analysis.")
        return

    avg_importances = np.mean([r['feature_importance'] for r in successful_runs], axis=0)
    
    importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': avg_importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

    importance_csv_path = run_dir / "feature_importance_full.csv"
    importance_df.to_csv(importance_csv_path, index=False)
    logging.info(f"Full feature importance data saved to {importance_csv_path}")

    # --- Aggregate importances by category ---
    importance_df['category'] = 'Other'
    importance_df.loc[importance_df['feature'].str.startswith('pca_') | importance_df['feature'].str.startswith('umap_'), 'category'] = 'Satellite'
    importance_df.loc[importance_df['feature'].str.startswith('patch_'), 'category'] = 'Satellite (Raw)'
    importance_df.loc[importance_df['feature'].str.startswith('soil_') | importance_df['feature'].str.startswith('wrb_'), 'category'] = 'Soil'
    importance_df.loc[importance_df['feature'].str.startswith('bio_'), 'category'] = 'Bioclimatic'
    importance_df.loc[importance_df['feature'].isin(['elev', 'slope', 'aspect']), 'category'] = 'Topography'
    importance_df.loc[importance_df['feature'].str.contains('_annual_'), 'category'] = 'Climate'
    importance_df.loc[importance_df['feature'].isin(['latitude', 'longitude']), 'category'] = 'Location'
    importance_df.loc[importance_df['feature'].str.startswith('wc_class_'), 'category'] = 'Land Cover'
    
    category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
    
    # --- Plot 1: Aggregated Category Importance ---
    fig_cat, ax_cat = plt.subplots(figsize=(12, 7))
    sns.barplot(
        x=category_importance.values,
        y=category_importance.index,
        hue=category_importance.index,
        ax=ax_cat,
        palette='viridis',
        legend=False
    )
    ax_cat.set_title('Aggregated Feature Importance by Category')
    ax_cat.set_xlabel('Sum of Importance Scores')
    ax_cat.set_ylabel('Category')
    plt.tight_layout()
    cat_plot_path = run_dir / "feature_importance_by_category.png"
    fig_cat.savefig(cat_plot_path, dpi=600)
    plt.close(fig_cat)
    logging.info(f"Category importance plot saved to {cat_plot_path}")

    # --- Plot 2: Top 30 Individual Features ---
    top_n = 30
    top_features = importance_df.head(top_n)
    
    fig_top, ax_top = plt.subplots(figsize=(10, 12))
    sns.barplot(
        x='importance',
        y='feature',
        hue='feature',
        data=top_features,
        ax=ax_top,
        palette='mako',
        legend=False
    )    
    ax_top.set_title(f'Top {top_n} Most Important Features (Averaged Across Runs)')
    ax_top.set_xlabel('Importance Score')
    ax_top.set_ylabel('Feature Name')
    plt.tight_layout()
    top_plot_path = run_dir / "feature_importance_top_features.png"
    fig_top.savefig(top_plot_path, dpi=600)
    plt.close(fig_top)
    logging.info(f"Top features plot saved to {top_plot_path}")
    
def analyze_and_save_error_analysis(run_result, run_dir):
    logging.info("--- Starting Error Analysis ---")

    data = run_result['error_analysis_data']
    y_test = np.array(data['y_test'])
    test_pred = np.array(data['test_pred'])
    info_test = data['info_test']

    error_df = pd.DataFrame(info_test)
    error_df['actual_rarefied'] = y_test
    error_df['predicted_rarefied'] = test_pred
    error_df['absolute_error'] = np.abs(y_test - test_pred)
    error_df = error_df.sort_values(by='absolute_error', ascending=False)
    
    error_csv_path = run_dir / "error_analysis_details.csv"
    error_df.to_csv(error_csv_path, index=False)
    logging.info(f"Error analysis data saved to {error_csv_path}")

    # --- Plot 1: Predicted vs. Actual ---
    fig_pvsa, ax_pvsa = plt.subplots(figsize=(8, 8))
    ax_pvsa.scatter(y_test, test_pred, alpha=0.5, linewidth=0.5)
    ax_pvsa.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal 1:1 Line')
    ax_pvsa.set_xlabel("Actual Biodiversity")
    ax_pvsa.set_ylabel("Predicted Biodiversity")
    ax_pvsa.set_title("Predicted vs. Actual Values (Test Set)")
    ax_pvsa.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_pvsa.legend()
    ax_pvsa.set_aspect('equal', 'box')
    plt.tight_layout()
    pvsa_plot_path = run_dir / "error_analysis_predicted_vs_actual.png"
    fig_pvsa.savefig(pvsa_plot_path, dpi=600)
    plt.close(fig_pvsa)
    logging.info(f"Predicted vs. actual plot saved to {pvsa_plot_path}")

    # --- Plot 2: Error vs. Actual ---
    fig_evsa, ax_evsa = plt.subplots(figsize=(9, 7))
    ax_evsa.scatter(y_test, error_df['absolute_error'], alpha=0.5, linewidth=0.5)
    ax_evsa.set_xlabel("Actual Biodiversity")
    ax_evsa.set_ylabel("Absolute Error")
    ax_evsa.set_title("Absolute Error vs. Actual Value")
    ax_evsa.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    evsa_plot_path = run_dir / "error_analysis_error_vs_actual.png"
    fig_evsa.savefig(evsa_plot_path, dpi=600)
    plt.close(fig_evsa)
    logging.info(f"Error vs. actual plot saved to {evsa_plot_path}")

def analyze_and_save_spatial_error_maps(run_result, run_dir):
    logging.info("--- Starting Spatial Error Analysis ---")

    # --- 1. Prepare Data ---
    # Reconstruct the error dataframe from the run result
    data = run_result['error_analysis_data']
    if not all(k in data for k in ['y_test', 'test_pred', 'info_test']):
        logging.warning("Missing necessary keys in 'error_analysis_data'. Skipping spatial analysis.")
        return

    # Ensure info_test is a DataFrame, as it might be a list of dicts
    info_df = pd.DataFrame(data['info_test'])
    if 'latitude' not in info_df.columns or 'longitude' not in info_df.columns:
        logging.warning("Latitude or Longitude not found in info_test. Skipping spatial analysis.")
        return
        
    error_df = info_df.copy()
    error_df['actual_rarefied'] = data['y_test']
    error_df['predicted_rarefied'] = data['test_pred']
    error_df['absolute_error'] = abs(error_df['actual_rarefied'] - error_df['predicted_rarefied'])
    error_df['percent_error'] = (error_df['absolute_error'] / error_df['actual_rarefied']) * 100

    # Create the GeoDataFrame
    gdf = geopandas.GeoDataFrame(
        error_df, geometry=geopandas.points_from_xy(error_df.longitude, error_df.latitude)
    )
    gdf.set_crs(epsg=4326, inplace=True)

    # --- 2. Load Basemap ---
    # This URL is hit every time; for frequent runs, consider caching this file locally.
    logging.info("Downloading world basemap...")
    try:
        world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = geopandas.read_file(world_url)
    except Exception as e:
        logging.error(f"Could not download or read the world basemap. Skipping spatial plots. Error: {e}")
        return

    # --- 3. Generate and Save Plots ---
    # Calculate dynamic plot boundaries from data points
    west, south, east, north = gdf.total_bounds
    lon_margin = (east - west) * 0.1
    lat_margin = (north - south) * 0.1
    plot_bounds = {
        'xlim': (west - lon_margin, east + lon_margin),
        'ylim': (south - lat_margin, north + lat_margin)
    }

    # Plot 1: Basic Point Map of Sample Distribution
    logging.info("Generating basic sample distribution map...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(15, 10))
    world.plot(ax=ax1, color='#e0e0e0', edgecolor='black', linewidth=0.5)
    gdf.plot(ax=ax1, marker='o', color='crimson', markersize=25, alpha=0.6)
    ax1.set_title('Geographic Distribution of Data Samples')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_xlim(*plot_bounds['xlim'])
    ax1.set_ylim(*plot_bounds['ylim'])
    map1_path = run_dir / "spatial_distribution_map.png"
    fig1.savefig(map1_path, dpi=600, bbox_inches='tight')
    plt.close(fig1)
    logging.info(f"Distribution map saved to {map1_path}")

    # Plot 2: Bubble Map of Absolute Error
    logging.info("Generating bubble map of prediction errors...")
    # Scale marker size for better visualization
    # We add a small base size and scale the error to avoid zero-sized markers
    min_marker_size = 5
    # Normalize error to 0-1 range and scale it for visibility
    max_error = gdf['absolute_error'].max()
    if max_error > 0:
        scaled_size = (gdf['absolute_error'] / max_error) * 150
    else: # Handle case where all errors are zero
        scaled_size = 0
    marker_sizes = min_marker_size + scaled_size
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 12))
    world.plot(ax=ax2, color='#e0e0e0', edgecolor='black', linewidth=0.5)
    gdf.plot(ax=ax2,
             column='absolute_error',
             cmap='plasma',
             markersize=marker_sizes,
             legend=True,
             legend_kwds={'label': "Absolute Error", 'orientation': "horizontal", 'pad': 0.01},
             alpha=0.75)
    ax2.set_title('Bubble Map of Absolute Error in Predictions')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_xlim(*plot_bounds['xlim'])
    ax2.set_ylim(*plot_bounds['ylim'])
    map2_path = run_dir / "spatial_error_bubble_map.png"
    fig2.savefig(map2_path, dpi=600, bbox_inches='tight')
    plt.close(fig2)
    logging.info(f"Error bubble map saved to {map2_path}")

    # Plot 3: Bubble Map of Percentage Error
    logging.info("Generating bubble map of percentage errors...")
    # Scale marker size for percentage error
    max_percent_error = gdf['percent_error'].max()
    if max_percent_error > 0:
        scaled_size_pct = (gdf['percent_error'] / max_percent_error) * 150
    else:
        scaled_size_pct = 0
    marker_sizes_pct = min_marker_size + scaled_size_pct

    fig3, ax3 = plt.subplots(1, 1, figsize=(15, 12))
    world.plot(ax=ax3, color='#e0e0e0', edgecolor='black', linewidth=0.5)
    gdf.plot(ax=ax3,
             column='percent_error',
             cmap='plasma',
             markersize=marker_sizes_pct,
             legend=True,
             legend_kwds={'label': "Percent Error (%)", 'orientation': "horizontal", 'pad': 0.01},
             alpha=0.75)
    ax3.set_title('Bubble Map of Percent Error in Predictions')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_xlim(*plot_bounds['xlim'])
    ax3.set_ylim(*plot_bounds['ylim'])
    map3_path = run_dir / "spatial_percent_error_bubble_map.png"
    fig3.savefig(map3_path, dpi=600, bbox_inches='tight')
    plt.close(fig3)
    logging.info(f"Percent error bubble map saved to {map3_path}")

def save_per_run_performance(all_run_results: List[Dict[str, Any]], run_dir: Path):
    logging.info("--- Saving Per-Run Performance Metrics ---")
    
    successful_runs = [r for r in all_run_results if 'error' not in r and 'test_stats' in r]
    if not successful_runs:
        logging.warning("No successful runs found to save individual performance metrics.")
        return

    # Create a flat list of dictionaries, where each item represents a run's stats
    per_run_stats_list = []
    for run in successful_runs:
        # Combine the run identifier (seed) with its performance metrics
        stats = {
            'run_seed': run.get('random_seed', 'N/A'),
            **run.get('test_stats', {})  # Unpacks {'r2': x, 'mae': y, ...} into the main dict
        }
        per_run_stats_list.append(stats)
    
    if not per_run_stats_list:
        logging.warning("No performance data was extracted to save.")
        return
        
    # --- Save to CSV file ---
    try:
        df = pd.DataFrame(per_run_stats_list)
        csv_path = run_dir / "per_run_performance_metrics.csv"
        # Order columns for clarity
        cols = ['run_seed'] + [col for col in df.columns if col != 'run_seed']
        df[cols].to_csv(csv_path, index=False)
        logging.info(f"Per-run performance metrics saved to {csv_path}")
    except Exception as e:
        logging.error(f"Failed to save per-run metrics to CSV: {e}", exc_info=True)

    # --- Save to JSON file ---
    try:
        json_path = run_dir / "per_run_performance_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(per_run_stats_list, f, indent=4)
        logging.info(f"Per-run performance metrics saved to {json_path}")
    except Exception as e:
        logging.error(f"Failed to save per-run metrics to JSON: {e}", exc_info=True)
        
def main_evaluation(args):
    
    # --- Configuration from args ---
    MODEL_TO_USE = args.model
    SATELLITE_DIM_REDUCTION = args.dim_reduction
    DIM_REDUCTION_COMPONENTS = args.dim_reduction_components
    BIODIVERSITY_CSV_PATHS = args.biodiversity_csvs
    REPRESENTATIONS_DIR_PATH = Path(args.representations_dir)
    CLIMATE_DATA_PATH = Path(args.climate_data_dir)
    CLIMATE_FEATURES_CACHE_DIR = Path(args.climate_cache_dir)
    SOIL_FEATURES_CACHE_DIR = Path(args.soil_cache_dir)
    RESULTS_BASE_DIR = Path(args.results_dir)
    WORLDCOVER_FEATURES_PATH = Path(args.worldcover_path)
    NUM_RUNS = args.num_runs

    # --- Data Source Flags from args ---
    USE_SATELLITE_DATA = args.use_satellite
    USE_CLIMATE_DATA = args.use_climate
    USE_SOIL_DATA = args.use_soil
    USE_WORLDCOVER_DATA = args.use_worldcover

    # --- Create a more descriptive run directory ---
    RESULTS_BASE_DIR.mkdir(exist_ok=True)
    
    # Build a descriptive name for the run configuration
    run_desc_parts = [MODEL_TO_USE]
    if USE_SATELLITE_DATA:
        run_desc_parts.append(f"sat({SATELLITE_DIM_REDUCTION})")
    if USE_CLIMATE_DATA:
        run_desc_parts.append("clim")
    if USE_SOIL_DATA:
        run_desc_parts.append("soil")
    if USE_WORLDCOVER_DATA:
        run_desc_parts.append("wc")
        
    run_name = "_".join(run_desc_parts)
    run_dir = RESULTS_BASE_DIR / f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(exist_ok=True)
    
    logging.info(f"Starting run: {run_name}. Results will be in {run_dir}")
    
    # For spot testing, basically
    USE_BBOX_FILTER = False # Set to True to enable the filter
    FILTER_MODE = 'include' # 'include' or 'exclude'
    BOUNDING_BOXES = [
        {'name': 'Estonia', 'min_lon': 20.043456, 'max_lon': 28.612792, 'min_lat': 55.799961, 'max_lat': 59.756502},
        #{'name': 'Saxony','min_lon': 10.15,'max_lon': 14.19,'min_lat': 50.05,'max_lat': 51.93},
        #{'name': 'Slovenia','min_lon': 13.248047,'max_lon': 16.631836,'min_lat': 45.321113,'max_lat': 46.890627},
    ]

    # --- Load Data ---
    try:
        df_list = [pd.read_csv(csv_path) for csv_path in BIODIVERSITY_CSV_PATHS]
        biodiversity_df = pd.concat(df_list, ignore_index=True)
        biodiversity_df.dropna(subset=['latitude', 'longitude', 'rarefied', 'sample_id'], inplace=True)
        logging.info(f"Loaded and cleaned {len(biodiversity_df)} total records.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}"); return
    
    # --- Applying bounding box filter ---
    if USE_BBOX_FILTER and BOUNDING_BOXES:
        logging.info(f"Applying bounding box filter (mode: {FILTER_MODE})...")
        initial_count = len(biodiversity_df)
        
        # Create a boolean mask that is False for all rows initially
        combined_mask = pd.Series(False, index=biodiversity_df.index)
        
        # For each box, find the points inside it and update the combined mask
        for box in BOUNDING_BOXES:
            logging.info(f"  ...using box '{box['name']}'")
            box_mask = (biodiversity_df['longitude'] >= box['min_lon']) & \
                       (biodiversity_df['longitude'] <= box['max_lon']) & \
                       (biodiversity_df['latitude']  >= box['min_lat']) & \
                       (biodiversity_df['latitude']  <= box['max_lat'])
            combined_mask |= box_mask # Logical OR: a point is included if it's in ANY box

        if FILTER_MODE == 'include':
            biodiversity_df = biodiversity_df[combined_mask]
            logging.info(f"Filter complete. Kept {len(biodiversity_df)} records inside the specified boxes (removed {initial_count - len(biodiversity_df)}).")
        elif FILTER_MODE == 'exclude':
            biodiversity_df = biodiversity_df[~combined_mask] # Invert mask to keep points OUTSIDE the boxes
            logging.info(f"Filter complete. Removed {initial_count - len(biodiversity_df)} records inside the specified boxes. {len(biodiversity_df)} records remain.")
        else:
            logging.error(f"Invalid FILTER_MODE: '{FILTER_MODE}'. Must be 'include' or 'exclude'. Skipping filter.")
        
        if len(biodiversity_df) == 0:
            logging.error("No records remain after filtering. Aborting."); return

    # --- Apply Biome-Based Filtering ---
    USE_BIOME_FILTER = args.use_biome_filter
    BIOME_IQR_MULTIPLIER = args.biome_iqr_multiplier
    BIOME_CACHE_DIR = Path(args.biome_cache_dir) if args.biome_cache_dir else None

    if USE_BIOME_FILTER:
        logging.info(f"Applying biome-based outlier filtering (IQR multiplier: {BIOME_IQR_MULTIPLIER})...")
        initial_count = len(biodiversity_df)

        # Save removed samples if requested
        removed_samples_path = None
        if args.save_biome_removed:
            removed_samples_path = run_dir / "biome_filtered_samples.csv"

        try:
            biodiversity_df = filter_by_biome(
                biodiversity_df,
                richness_column='rarefied',
                iqr_multiplier=BIOME_IQR_MULTIPLIER,
                remove_unassigned=args.biome_remove_unassigned,
                cache_dir=BIOME_CACHE_DIR,
                save_removed=removed_samples_path
            )
            logging.info(
                f"Biome filtering complete. "
                f"Kept {len(biodiversity_df)}/{initial_count} samples "
                f"({len(biodiversity_df)/initial_count*100:.1f}%)"
            )
        except Exception as e:
            logging.error(f"Biome filtering failed: {e}", exc_info=True)
            logging.warning("Continuing without biome filtering...")

        if len(biodiversity_df) == 0:
            logging.error("No records remain after biome filtering. Aborting."); return

    # --- Prepare Dataset ---    
    evaluator = CombinedPatchClimateEvaluator(
        climate_data_path=CLIMATE_DATA_PATH,
        use_climate_cache=True,
        climate_features_cache_dir=CLIMATE_FEATURES_CACHE_DIR,
        soil_features_cache_dir=SOIL_FEATURES_CACHE_DIR 
    )
    logging.info(f"Preparing combined dataset with {SATELLITE_DIM_REDUCTION} dimensionality reduction...")
    X_combined, y_combined, locations, _, _, f_names = evaluator.prepare_dataset(
        biodiversity_df, 
        representations_dir_path=REPRESENTATIONS_DIR_PATH,
        worldcover_features_path=WORLDCOVER_FEATURES_PATH,
        use_satellite=USE_SATELLITE_DATA, 
        use_climate=USE_CLIMATE_DATA, 
        use_soil=USE_SOIL_DATA,
        use_worldcover=USE_WORLDCOVER_DATA,
        satellite_dim_reduction=SATELLITE_DIM_REDUCTION,
        dim_reduction_components=DIM_REDUCTION_COMPONENTS,
        random_state=42
    )
    if X_combined is None: logging.error("Dataset preparation failed."); return
    logging.info(f"Dataset ready with {X_combined.shape[1]} features.")
    
    # --- Run Evaluations ---
    all_run_results = []
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n--- {MODEL_TO_USE.upper()} Evaluation Run {run_idx}/{NUM_RUNS} ---")
        run_result = evaluator.train_and_evaluate(
            X_combined, y_combined, list(locations), f_names,
            random_seed=run_idx,
            model_name=MODEL_TO_USE,
            model_params={}
        )
        all_run_results.append(run_result)
        if 'error' not in run_result:
            stats = run_result['test_stats']
            logging.info(f"Run {run_idx} - R2: {stats['r2']:.4f}, R2 (filtered): {stats['r2_filtered_98_percent']:.4f}, MAE: {stats['mae']:.4f}, MedAE: {stats['medae']:.4f}")
        else:
            logging.warning(f"Run {run_idx} - Failed: {run_result.get('error', 'Unknown')}")

    # --- Summarize Results ---
    successful_runs = [r for r in all_run_results if 'error' not in r]
    if successful_runs:
        save_per_run_performance(all_run_results, run_dir)

        # Collect all metrics from each run
        r2_scores = [r['test_stats']['r2'] for r in successful_runs]
        mae_scores = [r['test_stats']['mae'] for r in successful_runs]
        medae_scores = [r['test_stats']['medae'] for r in successful_runs]
        r2_filtered_scores = [r['test_stats']['r2_filtered_98_percent'] for r in successful_runs]
        me_scores = [r['test_stats']['me'] for r in successful_runs]
        rmse_scores = [r['test_stats']['rmse'] for r in successful_runs]

        mean_r2 = np.mean(r2_scores)
        summary_stats = {
            'model_type': MODEL_TO_USE,
            'satellite_dim_reduction': SATELLITE_DIM_REDUCTION,
            'num_samples': len(X_combined),
            'num_features': X_combined.shape[1],
            'num_successful_runs': len(successful_runs),
            'r2_mean': float(np.mean(r2_scores)), 'r2_std': float(np.std(r2_scores)),
            'rmse_mean': float(np.mean(rmse_scores)), 'rmse_std': float(np.std(rmse_scores)),
            'mae_mean': float(np.mean(mae_scores)), 'mae_std': float(np.std(mae_scores)),
            'me_mean': float(np.mean(me_scores)), 'me_std': float(np.std(me_scores)),
            'medae_mean': float(np.mean(medae_scores)), 'medae_std': float(np.std(medae_scores)),
            'r2_filtered_98_percent_mean': float(np.mean(r2_filtered_scores)), 'r2_filtered_98_percent_std': float(np.std(r2_filtered_scores))
        }

        print("\n--- Overall Summary ---")
        print(json.dumps(summary_stats, indent=2))
        summary_file = run_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f: json.dump(summary_stats, f, indent=2)
        print(f"\nSummary saved to {summary_file}")
    else:
        print("\nNo successful evaluation runs completed.")
        
    if successful_runs:
        # Find the single run that is most representative of the average performance
        best_run_index = -1
        min_diff = float('inf')
        for i, run in enumerate(successful_runs):
            diff = abs(run['test_stats']['r2'] - mean_r2)
            if diff < min_diff:
                min_diff = diff
                best_run_index = i
        
        most_representative_run = successful_runs[best_run_index]
        logging.info(f"\n--- Analyzing Most Representative Run (Run Index: {best_run_index}) ---")
        logging.info(f"Its R2 score ({most_representative_run['test_stats']['r2']:.4f}) is closest to the mean R2 ({mean_r2:.4f}).")

        # Now, use this single, most representative run for all visual analyses
        analyze_and_save_feature_importance(successful_runs, f_names, run_dir)
        #analyze_and_save_error_analysis(most_representative_run, run_dir)
        #analyze_and_save_spatial_error_maps(most_representative_run, run_dir)
        analyze_and_save_error_analysis(successful_runs[0], run_dir)
        analyze_and_save_spatial_error_maps(successful_runs[0], run_dir)

    else:
        print("\nNo successful runs to analyze.")
        
    print("\nEvaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model eval")

    # --- Model & Run Configs ---
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'lightgbm', 'xgboost'], help='Model to use')
    parser.add_argument('--num_runs', type=int, default=50, help='Number of random seed iterations')
    
    # --- Data Source Flags ---
    parser.add_argument('--use-satellite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-climate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-soil', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-worldcover', action=argparse.BooleanOptionalAction, default=True)

    # --- Dimensionality Reduction ---
    parser.add_argument('--dim_reduction', type=str, default='umap', choices=['none', 'pca', 'umap'], help='Dimensionality reduction for satellite features')
    parser.add_argument('--dim_reduction_components', type=int, default=256, help='Number of components for dimensionality reduction')

    # --- Biome Filtering ---
    parser.add_argument('--use-biome-filter', action='store_true', default=False, help='Enable biome-based outlier filtering')
    parser.add_argument('--biome-iqr-multiplier', type=float, default=5.0, help='IQR multiplier for biome outlier threshold (default: 5.0)')
    parser.add_argument('--biome-cache-dir', type=str, default='/scratch/ray25/ecoregions_cache', help='Directory to cache RESOLVE Ecoregions data')
    parser.add_argument('--biome-remove-unassigned', action='store_true', default=True, help='Remove samples that cannot be assigned to a biome')
    parser.add_argument('--save-biome-removed', action='store_true', default=False, help='Save removed biome outliers to CSV')

    # --- Path Configs ---
    parser.add_argument('--biodiversity_csvs', nargs='+', default=["/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv", "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"])
    parser.add_argument('--representations_dir', type=str, default="/maps-priv/maps/ray25/data/ecm_representations")
    parser.add_argument('--climate_data_dir', type=str, default="/maps-priv/maps/ray25/data/worldclim/data")
    parser.add_argument('--climate_cache_dir', type=str, default="/scratch/ray25/climate_features_cache")
    parser.add_argument('--soil_cache_dir', type=str, default="/scratch/ray25/soil_features_cache")
    parser.add_argument('--worldcover_path', type=str, default="/scratch/ray25/worldcover_features.csv")
    parser.add_argument('--results_dir', type=str, default="patch_climate_representation_results")
    
    args = parser.parse_args()
    main_evaluation(args)
    