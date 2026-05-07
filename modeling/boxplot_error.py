import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_comparative_error(
    csv_path1: Path,
    csv_path2: Path,
    csv_path3: Path,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_names = ['XGBoost', 'LightGBM', 'Random Forest']

    # --- Load and combine data ---
    try:
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        df3 = pd.read_csv(csv_path3)
    except FileNotFoundError as e:
        logging.error(f"Could not find an input file: {e}. Aborting.")
        return

    df1['model'] = model_names[0]
    df2['model'] = model_names[1]
    df3['model'] = model_names[2]
    
    # Combine the dataframes
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)

    # --- Create the Box Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=combined_df,
        x='model',
        y='absolute_error',
        hue='model',
        ax=ax,
        palette='mako',
        legend=False
    )
    
    ax.set_title('Box Plot of Absolute Error Distribution')
    ax.set_xlabel('Model')
    ax.set_ylabel('Absolute Error')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    error_plot_path = output_dir / "error_distribution_comparison.png"
    fig.savefig(error_plot_path, dpi=600)
    plt.close(fig)
    logging.info(f"Comparative error box plot saved to {error_plot_path}")


if __name__ == '__main__':
    base_dir = Path("/maps-priv/maps/ray25/config_src")
    results_dir = base_dir / "patch_climate_representation_results"

    xgb_csv_path = results_dir / "xgboost_sat(umap)_clim_soil_wc_20250723-191808/error_analysis_details.csv"
    lgbm_csv_path = results_dir / "lightgbm_sat(umap)_clim_soil_wc_20250723-184318/error_analysis_details.csv"
    rf_csv_path = results_dir / "rf_sat(umap)_clim_soil_wc_20250723-175243/error_analysis_details.csv"

    plot_comparative_error(
        csv_path1=xgb_csv_path,
        csv_path2=lgbm_csv_path,
        csv_path3=rf_csv_path,
        output_dir=base_dir
    )
