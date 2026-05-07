import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import umap.umap_ as umap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


VARIABLE_MAP = {
    # Soil Variables
    'soil_phh2o_0_5cm': 'Soil pH (0-5cm)', 'soil_phh2o_5_15cm': 'Soil pH (5-15cm)', 'soil_phh2o_gradient': 'Soil pH (Gradient)',
    'soil_soc_0_5cm': 'Soil Org. Carbon (0-5cm)', 'soil_soc_5_15cm': 'Soil Org. Carbon (5-15cm)', 'soil_soc_gradient': 'Soil Org. Carbon (Gradient)',
    'soil_nitrogen_0_5cm': 'Soil Nitrogen (0-5cm)', 'soil_nitrogen_5_15cm': 'Soil Nitrogen (5-15cm)', 'soil_nitrogen_gradient': 'Soil Nitrogen (Gradient)',
    'soil_clay_0_5cm': 'Soil Clay % (0-5cm)', 'soil_clay_5_15cm': 'Soil Clay % (5-15cm)', 'soil_clay_gradient': 'Soil Clay % (Gradient)',
    'soil_silt_0_5cm': 'Soil Silt % (0-5cm)', 'soil_silt_5_15cm': 'Soil Silt % (5-15cm)', 'soil_silt_gradient': 'Soil Silt % (Gradient)',
    'soil_sand_0_5cm': 'Soil Sand % (0-5cm)', 'soil_sand_5_15cm': 'Soil Sand % (5-15cm)', 'soil_sand_gradient': 'Soil Sand % (Gradient)',
    'soil_cec_0_5cm': 'Soil CEC (0-5cm)', 'soil_cec_5_15cm': 'Soil CEC (5-15cm)', 'soil_cec_gradient': 'Soil CEC (Gradient)',
    'soil_bdod_0_5cm': 'Soil Bulk Density (0-5cm)', 'soil_bdod_5_15cm': 'Soil Bulk Density (5-15cm)', 'soil_bdod_gradient': 'Soil Bulk Density (Gradient)',
    'soil_wv0033_0_5cm': 'Soil Water (0-5cm)', 'soil_wv0033_5_15cm': 'Soil Water (5-15cm)', 'soil_wv0033_gradient': 'Soil Water (Gradient)',
    'soil_wv1500_0_5cm': 'Soil Water (0-5cm)', 'soil_wv1500_5_15cm': 'Soil Water (5-15cm)', 'soil_wv1500_gradient': 'Soil Water (Gradient)',
    'soil_wrb': 'Soil Type (WRB)',
    # Climate Variables
    'prec_annual_mean': 'Precip (Mean)', 'prec_annual_min': 'Precip (Min)', 'prec_annual_max': 'Precip (Max)', 'prec_annual_std': 'Precip (Std Dev)',
    'tavg_annual_mean': 'Avg Temp (Mean)', 'tavg_annual_min': 'Avg Temp (Min)', 'tavg_annual_max': 'Avg Temp (Max)', 'tavg_annual_std': 'Avg Temp (Std Dev)',
    'tmin_annual_mean': 'Min Temp (Mean)', 'tmin_annual_min': 'Min Temp (Min)', 'tmin_annual_max': 'Min Temp (Max)', 'tmin_annual_std': 'Min Temp (Std Dev)',
    'tmax_annual_mean': 'Max Temp (Mean)', 'tmax_annual_min': 'Max Temp (Min)', 'tmax_annual_max': 'Max Temp (Max)', 'tmax_annual_std': 'Max Temp (Std Dev)',
    'srad_annual_mean': 'Solar Rad. (Mean)', 'srad_annual_min': 'Solar Rad. (Min)', 'srad_annual_max': 'Solar Rad. (Max)', 'srad_annual_std': 'Solar Rad. (Std Dev)',
    'vapr_annual_mean': 'Vapor Pr. (Mean)', 'vapr_annual_min': 'Vapor Pr. (Min)', 'vapr_annual_max': 'Vapor Pr. (Max)', 'vapr_annual_std': 'Vapor Pr. (Std Dev)',
    'wind_annual_mean': 'Wind Speed (Mean)', 'wind_annual_min': 'Wind Speed (Min)', 'wind_annual_max': 'Wind Speed (Max)', 'wind_annual_std': 'Wind Speed (Std Dev)',
    'elev': 'Elevation',
    # Bioclimatic Variables
    'bio_1': 'Bio 1 (Ann. Mean Temp)', 'bio_2': 'Bio 2 (Mean Diurnal Range)', 'bio_3': 'Bio 3 (Isothermality)', 'bio_4': 'Bio 4 (Temp Seasonality)',
    'bio_5': 'Bio 5 (Max Temp Warmest Month)', 'bio_6': 'Bio 6 (Min Temp Coldest Month)', 'bio_7': 'Bio 7 (Temp Annual Range)',
    'bio_8': 'Bio 8 (Mean Temp Wettest Qtr)', 'bio_9': 'Bio 9 (Mean Temp Driest Qtr)', 'bio_10': 'Bio 10 (Mean Temp Warmest Qtr)',
    'bio_11': 'Bio 11 (Mean Temp Coldest Qtr)', 'bio_12': 'Bio 12 (Ann. Precipitation)', 'bio_13': 'Bio 13 (Precip Wettest Month)',
    'bio_14': 'Bio 14 (Precip Driest Month)', 'bio_15': 'Bio 15 (Precip Seasonality)', 'bio_16': 'Bio 16 (Precip Wettest Qtr)',
    'bio_17': 'Bio 17 (Precip Driest Qtr)', 'bio_18': 'Bio 18 (Precip Warmest Qtr)', 'bio_19': 'Bio 19 (Precip Coldest Qtr)',
}

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correlations between SSL representations and environmental data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments for data paths ---
    parser.add_argument(
        '--biodiversity_csvs', nargs='+',
        default=["/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv", "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"],
        help="Paths to the biodiversity CSV files containing sample metadata."
    )
    parser.add_argument(
        '--representations_dir', type=str, default="/maps-priv/maps/ray25/data/ecm_representations",
        help="Directory containing the SSL .npy representation files."
    )
    parser.add_argument(
        '--soil_cache_dir', type=str, default="/scratch/ray25/soil_features_cache",
        help="Directory containing cached SoilGrids features."
    )
    parser.add_argument(
        '--climate_cache_dir', type=str, default="/scratch/ray25/climate_features_cache",
        help="Directory containing cached WorldClim features."
    )
    parser.add_argument(
        '--output_dir', type=str, default="ssl_correlation_analysis",
        help="Directory to save analysis results (plots and CSVs)."
    )
    # --- Arguments for analysis control ---
    parser.add_argument(
        '--dim_reduction_method', type=str, default='pca',
        choices=['pca', 'umap', 'both'],
        help="Dimensionality reduction method(s) to use. 'both' runs a comparative analysis."
    )
    parser.add_argument(
        '--n_components', type=int, default=10,
        help="Number of components for dimensionality reduction."
    )
    return parser.parse_args()


def prepare_data(
    biodiversity_csvs: List[str], representations_dir: Path,
    soil_cache_dir: Path, climate_cache_dir: Path
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    try:
        df_list = [pd.read_csv(p) for p in biodiversity_csvs]
        bio_df = pd.concat(df_list, ignore_index=True)
        bio_df.dropna(subset=['sample_id', 'latitude', 'longitude'], inplace=True)
        bio_df['sample_id'] = bio_df['sample_id'].astype(str)
        logging.info(f"Loaded {len(bio_df)} records from biodiversity CSVs.")
    except Exception as e:
        logging.error(f"Failed to load biodiversity data: {e}", exc_info=True)
        raise

    logging.info(f"Loading SSL representations from: {representations_dir}")
    ssl_features = {}
    for npy_path in tqdm(list(representations_dir.glob('*.npy')), desc="Loading SSL features"):
        try:
            sample_id = npy_path.stem.split('_')[1]
            representation = np.load(npy_path).flatten()
            ssl_features[sample_id] = representation
        except Exception:
            logging.warning(f"Could not load or parse {npy_path.name}", exc_info=False)

    if not ssl_features:
        logging.error("No SSL representations were loaded. Aborting.")
        raise ValueError("SSL representation directory is empty or invalid.")

    ssl_df = pd.DataFrame.from_dict(ssl_features, orient='index')
    ssl_df.columns = [f'ssl_feat_{i}' for i in range(ssl_df.shape[1])]
    ssl_df.index.name = 'sample_id'
    ssl_feature_names = ssl_df.columns.tolist()
    logging.info(f"Loaded {len(ssl_df)} SSL representations with {len(ssl_feature_names)} features each.")

    def load_cached_features(cache_dir: Path, desc: str) -> pd.DataFrame:
        logging.info(f"Loading {desc} features from: {cache_dir}")
        features = {}
        for npy_path in tqdm(list(cache_dir.glob('*.npy')), desc=f"Loading {desc} features"):
            try:
                sample_id = npy_path.stem
                feature_dict = np.load(npy_path, allow_pickle=True).item()
                features[sample_id] = feature_dict
            except Exception:
                logging.warning(f"Could not load {npy_path.name}", exc_info=False)
        return pd.DataFrame.from_dict(features, orient='index').reset_index().rename(columns={'index': 'sample_id'})

    soil_df = load_cached_features(soil_cache_dir, "soil")
    climate_df = load_cached_features(climate_cache_dir, "climate")
    soil_feature_names = [col for col in soil_df.columns if col != 'sample_id']
    climate_feature_names = [col for col in climate_df.columns if col != 'sample_id']

    logging.info("Merging all data sources...")
    merged_df = pd.merge(bio_df, ssl_df, on='sample_id', how='inner')
    merged_df = pd.merge(merged_df, soil_df, on='sample_id', how='left')
    merged_df = pd.merge(merged_df, climate_df, on='sample_id', how='left')

    merged_df.dropna(inplace=True)
    return merged_df, ssl_feature_names, soil_feature_names, climate_feature_names


def reduce_and_correlate(
    df: pd.DataFrame, ssl_cols: List[str], env_cols: List[str],
    method: str, n_components: int, output_dir: Path
) -> pd.DataFrame:
    logging.info(f"--- Running Reduction & Correlation ({method.upper()}) ---")

    X_ssl = df[ssl_cols].values
    X_ssl_scaled = StandardScaler().fit_transform(X_ssl)

    logging.info(f"Applying {method.upper()}...")
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_ssl_scaled)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
        X_reduced = reducer.fit_transform(X_ssl_scaled)
    else:
        raise ValueError(f"Invalid reduction method specified: {method}")

    reduced_cols = [f'{method.upper()}_{i+1}' for i in range(n_components)]
    reduced_df = pd.DataFrame(X_reduced, columns=reduced_cols, index=df.index)

    analysis_df = pd.concat([reduced_df, df[env_cols]], axis=1)
    corr_matrix = analysis_df.corr(method='pearson')
    cross_corr = corr_matrix.loc[env_cols, reduced_cols]

    corr_csv_path = output_dir / f'{method}_cross_correlation_matrix.csv'
    cross_corr.to_csv(corr_csv_path)
    logging.info(f"Cross-correlation matrix saved to: {corr_csv_path}")

    logging.info(f"Generating scatter plots for top {method.upper()} correlated pairs...")
    top_n = 9
    if not cross_corr.empty:
        unstacked_corr = cross_corr.stack().abs().sort_values(ascending=False)
        top_pairs = unstacked_corr.head(min(top_n, len(unstacked_corr))).index.tolist()

        fig_s, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig_s.suptitle(f'Top {len(top_pairs)} Correlated Pairs: {method.upper()} vs. Environmental Features', fontsize=20)
        for i, (env_feat, ssl_comp) in enumerate(top_pairs):
            ax = axes.flatten()[i]
            sns.regplot(
                data=analysis_df, x=ssl_comp, y=env_feat, ax=ax,
                scatter_kws={'alpha': 0.3, 's': 15}, line_kws={'color': 'crimson'}
            )
            corr_val = cross_corr.loc[env_feat, ssl_comp]
            ax.set_title(f'Correlation = {corr_val:.3f}', fontsize=12)
            ax.set_xlabel(ssl_comp, fontsize=10)
            ax.set_ylabel(env_feat, fontsize=10)
        for j in range(i + 1, 9):
            axes.flatten()[j].set_visible(False) # Hide unused subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        scatter_path = output_dir / f'{method}_top_correlations_scatter.png'
        fig_s.savefig(scatter_path, dpi=300)
        plt.close(fig_s)
        logging.info(f"Scatter plots saved to: {scatter_path}")

    return cross_corr


def run_single_method_analysis(
    df: pd.DataFrame, ssl_cols: List[str], env_cols: List[str],
    method: str, n_components: int, output_dir: Path
):
    cross_corr = reduce_and_correlate(df, ssl_cols, env_cols, method, n_components, output_dir)

    logging.info(f"Generating correlation heatmap for {method.upper()}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    vmax = abs(cross_corr).max().max()
    vmax = round(vmax + 0.05, 1) if not np.isnan(vmax) else 1.0

    fig, ax = plt.subplots(figsize=(18, max(12, len(env_cols) // 3)))
    fig.suptitle(f'Correlation: {method.upper()} Components vs. Environmental Variables', fontsize=18)
    sns.heatmap(
        cross_corr, cmap='vlag', vmin=-vmax, vmax=vmax,
        annot=False, linewidths=.5, ax=ax, cbar=True
    )
    ax.set_title('Pearson Correlation', fontsize=16)
    ax.tick_params(axis='x', rotation=45)

    current_labels = [item.get_text() for item in ax.get_yticklabels()]
    descriptive_labels = [VARIABLE_MAP.get(label, label) for label in current_labels]
    ax.set_yticklabels(descriptive_labels, rotation=0, ha='right', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    heatmap_path = output_dir / f'{method}_correlation_heatmap.png'
    fig.savefig(heatmap_path, dpi=600)
    plt.close(fig)
    logging.info(f"Heatmap saved to: {heatmap_path}")


def run_comparative_analysis(
    df: pd.DataFrame, ssl_cols: List[str], env_cols: List[str],
    n_components: int, output_dir: Path
):
    pca_corr = reduce_and_correlate(df, ssl_cols, env_cols, 'pca', n_components, output_dir)
    umap_corr = reduce_and_correlate(df, ssl_cols, env_cols, 'umap', n_components, output_dir)

    logging.info("Generating side-by-side PCA vs. UMAP correlation heatmap...")
    plt.style.use('seaborn-v0_8-whitegrid')

    vmax = max(abs(pca_corr.min().min()), abs(pca_corr.max().max()),
               abs(umap_corr.min().min()), abs(umap_corr.max().max()))
    vmax = round(vmax + 0.05, 1) if not np.isnan(vmax) else 1.0

    fig, (ax1, ax2, cbar_ax) = plt.subplots(
        1, 3, figsize=(18, max(12, len(env_cols) // 3)),
        gridspec_kw={'width_ratios': [10, 10, 1]}
    )
    fig.suptitle('Correlation Between Reduced SSL Features and Environmental Variables', fontsize=20)

    sns.heatmap(
        pca_corr, cmap='vlag', vmin=-vmax, vmax=vmax,
        annot=False, linewidths=.5, ax=ax1, cbar=False
    )
    ax1.set_title('PCA Components', fontsize=16)
    ax1.tick_params(axis='x', rotation=45)

    sns.heatmap(
        umap_corr, cmap='vlag', vmin=-vmax, vmax=vmax,
        annot=False, linewidths=.5, ax=ax2, cbar_ax=cbar_ax
    )
    ax2.set_title('UMAP Components', fontsize=16)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.tick_params(axis='x', rotation=45)

    current_labels = [item.get_text() for item in ax1.get_yticklabels()]
    descriptive_labels = [VARIABLE_MAP.get(label, label) for label in current_labels]
    ax1.set_yticklabels(descriptive_labels, rotation=0, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    comparison_path = output_dir / 'pca_vs_umap_comparison_heatmap.png'
    fig.savefig(comparison_path, dpi=600)
    plt.close(fig)
    logging.info(f"Comparison heatmap saved to: {comparison_path}")


def main():
    args = get_args()
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        merged_df, ssl_cols, soil_cols, climate_cols = prepare_data(
            args.biodiversity_csvs, Path(args.representations_dir),
            Path(args.soil_cache_dir), Path(args.climate_cache_dir)
        )
        environmental_cols = sorted(soil_cols + climate_cols)

        if args.dim_reduction_method in ['pca', 'umap']:
            logging.info(f"--- Running Single Analysis: {args.dim_reduction_method.upper()} ---")
            run_single_method_analysis(
                merged_df, ssl_cols, environmental_cols,
                args.dim_reduction_method, args.n_components, output_dir
            )
        elif args.dim_reduction_method == 'both':
            logging.info("--- Running Comparative Analysis: PCA vs. UMAP ---")
            run_comparative_analysis(
                merged_df, ssl_cols, environmental_cols,
                args.n_components, output_dir
            )

    except Exception as e:
        logging.error("An error occurred during the script execution.", exc_info=True)
    finally:
        end_time = time.time()
        logging.info(f"--- Script finished in {end_time - start_time:.2f} seconds ---")


if __name__ == '__main__':
    main()