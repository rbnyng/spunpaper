import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import logging

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_comparative_feature_importance(
    xgb_csv_path: Path,
    lgbm_csv_path: Path,
    rf_csv_path: Path,
    output_dir: Path,
    top_n: int = 30
):

    # --- Load and combine data ---
    try:
        df_xgb = pd.read_csv(xgb_csv_path)
        df_lgbm = pd.read_csv(lgbm_csv_path)
        df_rf = pd.read_csv(rf_csv_path)
    except FileNotFoundError as e:
        logging.error(f"Could not find an input file: {e}. Aborting.")
        return

    df_xgb['model'] = 'XGBoost'
    df_lgbm['model'] = 'LightGBM'
    df_rf['model'] = 'Random Forest'

    # normalize importances before combining,
    df_xgb['importance_norm'] = df_xgb['importance'] / df_xgb['importance'].sum()
    df_lgbm['importance_norm'] = df_lgbm['importance'] / df_lgbm['importance'].sum()
    df_rf['importance_norm'] = df_rf['importance'] / df_rf['importance'].sum()
    
    combined_df = pd.concat([df_xgb, df_lgbm, df_rf], ignore_index=True)

    # --- Plot 1: Top N Individual Features ---
    # Determine the top N features based on the average normalized importance across models
    mean_importance = combined_df.groupby('feature')['importance_norm'].mean().sort_values(ascending=False)
    top_n_features = mean_importance.head(top_n).index
    
    plot_data_top_n = combined_df[combined_df['feature'].isin(top_n_features)].copy()
    
    # Sort the features in the plot by their mean importance
    plot_data_top_n['feature'] = pd.Categorical(plot_data_top_n['feature'], categories=top_n_features, ordered=True)
    plot_data_top_n = plot_data_top_n.sort_values('feature')

    fig_top, ax_top = plt.subplots(figsize=(12, 14))
    sns.barplot(
        data=plot_data_top_n,
        x='importance_norm',
        y='feature',
        hue='model',
        ax=ax_top,
        palette='viridis'
    )
    ax_top.set_title(f'Top {top_n} Features: Importance Face-off')
    ax_top.set_xlabel('Normalized Importance Score')
    ax_top.set_ylabel('Feature')
    ax_top.legend(title='Model')
    plt.tight_layout()
    
    top_plot_path = output_dir / "feature_importance_top_features_comparison.png"
    fig_top.savefig(top_plot_path, dpi=600)
    plt.close(fig_top)
    logging.info(f"Top features comparison plot saved to {top_plot_path}")

    # --- Plot 2: Aggregated Category Importance ---
    df = combined_df.copy()
    df['category'] = 'Other'
    df.loc[df['feature'].str.startswith('pca_') | df['feature'].str.startswith('umap_'), 'category'] = 'Satellite'
    df.loc[df['feature'].str.startswith('patch_'), 'category'] = 'Satellite (Raw)'
    df.loc[df['feature'].str.startswith('soil_') | df['feature'].str.startswith('wrb_'), 'category'] = 'Soil'
    df.loc[df['feature'].str.startswith('bio_'), 'category'] = 'Bioclimatic'
    df.loc[df['feature'].isin(['elev', 'slope', 'aspect']), 'category'] = 'Topography'
    df.loc[df['feature'].str.contains('_annual_'), 'category'] = 'Climate'
    df.loc[df['feature'].isin(['latitude', 'longitude']), 'category'] = 'Location'
    df.loc[df['feature'].str.startswith('wc_class_'), 'category'] = 'Land Cover'

    category_summary = df.groupby(['category', 'model'])['importance_norm'].sum().reset_index()

    # Determine order for y-axis based on total importance per category
    category_order = df.groupby('category')['importance_norm'].sum().sort_values(ascending=False).index

    fig_cat, ax_cat = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=category_summary,
        x='importance_norm',
        y='category',
        hue='model',
        ax=ax_cat,
        order=category_order,
        palette='viridis'
    )
    ax_cat.set_title('Aggregated Feature Importance by Category')
    ax_cat.set_xlabel('Sum of Normalized Importance Scores')
    ax_cat.set_ylabel('Category')
    ax_cat.legend(title='Model')
    plt.tight_layout()
    
    cat_plot_path = output_dir / "feature_importance_by_category_comparison.png"
    fig_cat.savefig(cat_plot_path, dpi=600)
    plt.close(fig_cat)
    logging.info(f"Category importance comparison plot saved to {cat_plot_path}")
    logging.info("--- Analysis Complete ---")


if __name__ == '__main__':
    plot_comparative_feature_importance(
        xgb_csv_path=Path("/maps-priv/maps/ray25/config_src/patch_climate_representation_results/xgboost_sat(umap)_clim_soil_wc_20250723-191808/feature_importance_full.csv"),
        lgbm_csv_path=Path("/maps-priv/maps/ray25/config_src/patch_climate_representation_results/lightgbm_sat(umap)_clim_soil_wc_20250723-184318/feature_importance_full.csv"),
        rf_csv_path=Path("/maps-priv/maps/ray25/config_src/patch_climate_representation_results/rf_sat(umap)_clim_soil_wc_20250723-175243/feature_importance_full.csv"),
        output_dir=Path("/maps-priv/maps/ray25/config_src/"),
        top_n=25
    )