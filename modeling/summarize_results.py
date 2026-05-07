import pandas as pd
import json
from pathlib import Path

def summarize_all_runs(results_base_dir="/maps-priv/maps/ray25/config_src/patch_climate_representation_results"):
    """
    Aggregates evaluation_summary.json files from all subdirectories
    into a single CSV file for easy comparison.
    """
    base_path = Path(results_base_dir)
    if not base_path.exists():
        print(f"Error: Base results directory '{base_path}' not found.")
        return

    # Find all summary files
    summary_files = list(base_path.rglob("evaluation_summary.json"))
    
    if not summary_files:
        print(f"No 'evaluation_summary.json' files found in '{base_path}'.")
        return

    print(f"Found {len(summary_files)} summary files to aggregate.")

    all_results = []
    for f in summary_files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                # The run configuration is captured by the parent directory's name
                run_config_name = f.parent.name
                data['run_configuration'] = run_config_name
                all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not read or parse {f}. Error: {e}")

    if not all_results:
        print("No valid results could be loaded.")
        return

    # Convert to a DataFrame for easy analysis and saving
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    cols = ['run_configuration', 'model_type', 'satellite_dim_reduction', 
            'r2_mean', 'r2_std', 'r2_filtered_98_percent_mean', 'r2_filtered_98_percent_std',
            'mae_mean', 'mae_std', 'medae_mean', 'medae_std', 'me_mean', 'me_std',
            'rmse_mean', 'rmse_std', 'num_samples', 'num_features', 'num_successful_runs']
    
    # Ensure all desired columns exist, adding any missing ones
    for col in cols:
        if col not in results_df.columns:
            results_df[col] = None
            
    results_df = results_df[cols]

    # Sort by the main performance metric
    results_df = results_df.sort_values(by='r2_mean', ascending=False)
    
    # Save to a master CSV file
    output_path = base_path / "master_ablation_summary.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n--- Aggregation Complete ---")
    print(f"Master summary saved to: {output_path}")
    print("\nTop 5 Configurations by Mean R2:")
    print(results_df.head(5).to_string(index=False))


if __name__ == "__main__":
    summarize_all_runs()