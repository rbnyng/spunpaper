"""
Retrain SSL-only model with PCA instead of UMAP.

PCA transform is O(n) vs UMAP's O(n log n) with huge constants.
Expected speedup: ~100x for inference.

Usage:
    python retrain_with_pca.py
"""

import pickle
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from existing training code
from spun_train_patch import CombinedPatchClimateEvaluator
import lightgbm as lgb


class Config:
    TRAINING_BIODIVERSITY_CSVS = [
        "/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv",
        "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"
    ]
    TRAINING_REPRESENTATIONS_DIR = "/maps-priv/maps/ray25/data/ecm_representations"
    MODEL_OUTPUT_DIR = Path("./model")

    # Changed from 'umap' to 'pca'
    SATELLITE_DIM_REDUCTION = 'pca'
    DIM_REDUCTION_COMPONENTS = 256
    USE_BIOME_FILTER = False  # Disable to avoid tempfile import error in spun_train_patch

    # Output paths for PCA model
    EVALUATOR_SAVE_PATH = MODEL_OUTPUT_DIR / "evaluator_ssl_pca.pkl"
    MODEL_SSL_ONLY_SAVE_PATH = MODEL_OUTPUT_DIR / "model_ssl_pca.pkl"


def main():
    config = Config()
    config.MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    logging.info("=" * 60)
    logging.info("Retraining SSL-only model with PCA (instead of UMAP)")
    logging.info("=" * 60)

    # Check if already exists
    if config.MODEL_SSL_ONLY_SAVE_PATH.exists() and config.EVALUATOR_SAVE_PATH.exists():
        logging.info("PCA model already exists. Delete to retrain:")
        logging.info(f"  rm {config.EVALUATOR_SAVE_PATH}")
        logging.info(f"  rm {config.MODEL_SSL_ONLY_SAVE_PATH}")
        return

    # Load training data
    logging.info("\nLoading training data...")
    df_list = [pd.read_csv(csv_path) for csv_path in config.TRAINING_BIODIVERSITY_CSVS]
    biodiversity_df = pd.concat(df_list, ignore_index=True).dropna(
        subset=['latitude', 'longitude', 'rarefied', 'sample_id']
    )
    logging.info(f"  Loaded {len(biodiversity_df)} samples")

    # Biome filter disabled due to missing tempfile import in spun_train_patch
    # if config.USE_BIOME_FILTER:
    #     biodiversity_df = filter_by_biome(biodiversity_df)
    #     logging.info(f"  After biome filter: {len(biodiversity_df)} samples")

    # Create evaluator and prepare dataset
    logging.info(f"\nPreparing dataset with {config.SATELLITE_DIM_REDUCTION.upper()} dimensionality reduction...")
    logging.info(f"  Components: {config.DIM_REDUCTION_COMPONENTS}")

    evaluator_ssl = CombinedPatchClimateEvaluator(
        climate_data_path=None,
        use_climate_cache=False,
        climate_features_cache_dir=None,
        soil_features_cache_dir=None
    )

    X_ssl, y_ssl, _, _, _, _ = evaluator_ssl.prepare_dataset(
        biodiversity_df,
        representations_dir_path=config.TRAINING_REPRESENTATIONS_DIR,
        use_satellite=True,
        use_climate=False,
        use_soil=False,
        use_worldcover=False,
        satellite_dim_reduction=config.SATELLITE_DIM_REDUCTION,
        dim_reduction_components=config.DIM_REDUCTION_COMPONENTS,
        random_state=42
    )

    # Remove lat/lon columns (last 2 columns)
    X_ssl_features_only = X_ssl[:, :-2]
    logging.info(f"  Feature matrix shape: {X_ssl_features_only.shape}")

    # Train LightGBM
    logging.info("\nTraining LightGBM model...")
    model_ssl = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        n_jobs=-1,
        verbose=-1
    )
    model_ssl.fit(X_ssl_features_only, y_ssl)
    logging.info("  Training complete")

    # Save
    logging.info(f"\nSaving evaluator to {config.EVALUATOR_SAVE_PATH}")
    with open(config.EVALUATOR_SAVE_PATH, 'wb') as f:
        pickle.dump(evaluator_ssl, f)

    logging.info(f"Saving model to {config.MODEL_SSL_ONLY_SAVE_PATH}")
    with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'wb') as f:
        pickle.dump(model_ssl, f)

    logging.info("\n" + "=" * 60)
    logging.info("DONE! PCA model saved.")
    logging.info("=" * 60)
    logging.info("\nTo use the PCA model, update uk_national_parks_inference.py:")
    logging.info("  EVALUATOR_SAVE_PATH = MODEL_OUTPUT_DIR / 'evaluator_ssl_pca.pkl'")
    logging.info("  MODEL_SSL_ONLY_SAVE_PATH = MODEL_OUTPUT_DIR / 'model_ssl_pca.pkl'")


if __name__ == "__main__":
    main()
