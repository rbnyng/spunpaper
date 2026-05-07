import itertools
import subprocess
from pathlib import Path

def run_ablation_studies():
    # --- Define the Parameter Space ---
    MODELS = ['rf', 'lightgbm', 'xgboost']
    # 1. Limit dimensionality reduction to UMAP only
    DIM_REDUCTION_METHODS = ['umap']
    
    DATA_FLAGS = {
        '--use-satellite': True,
        '--use-climate': True,
        '--use-soil': True,
        '--use-worldcover': True
    }
    
    data_source_keys = list(DATA_FLAGS.keys())
    # Test each source alone
    single_sources = [(key,) for key in data_source_keys]
    # Test all sources together
    all_sources_together = tuple(data_source_keys)
    # Final list of experiments
    all_source_combos = single_sources + [all_sources_together]
    print(all_source_combos)
    # --- Build and Run Commands ---
    base_cmd = [
        'python', 'src/spun_train_patch.py',
        '--num_runs', '50',
    ]
    
    run_configs = []

    for model in MODELS:
        for source_combo in all_source_combos:
            
            has_satellite = '--use-satellite' in source_combo
            
            if has_satellite:
                # If satellite data is used, run with UMAP
                for dim_red_method in DIM_REDUCTION_METHODS:
                    run_configs.append({
                        'model': model,
                        'sources': source_combo,
                        'dim_reduction': dim_red_method
                    })
            else:
                # If no satellite data, dim_reduction is 'none'
                run_configs.append({
                    'model': model,
                    'sources': source_combo,
                    'dim_reduction': 'none'
                })

    print(f"--- Preparing to execute {len(run_configs)} unique ablation runs. ---")

    for i, config in enumerate(run_configs):
        print(f"\n--- RUN {i+1}/{len(run_configs)} ---")
        
        cmd = base_cmd.copy()
        cmd.extend(['--model', config['model']])
        cmd.extend(['--dim_reduction', config['dim_reduction']])
        
        # Add data source flags
        for flag in data_source_keys:
            if flag in config['sources']:
                cmd.append(flag)
            else:
                cmd.append(f"--no-{flag.strip('--')}")

        print(f"Executing: {' '.join(cmd)}")

        try:
            source_desc = "_".join(s.split('-')[-1] for s in config['sources'])
            log_path = Path(f"ablation_log_{config['model']}_{source_desc}_{config['dim_reduction']}.txt")
            with open(log_path, 'w') as log_file:
                subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            print(f"--- Run {i+1} completed successfully. Log: {log_path} ---")

        except subprocess.CalledProcessError as e:
            print(f"Run {i+1} Failed")
            print(f"Check the log file '{log_path}' for error details.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_ablation_studies()