import argparse
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import paths

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--run-dir", default=None,
                help="Run directory containing per_run_performance_metrics.csv. "
                     "Default: newest lightgbm_sat(umap)_clim_soil_wc_* under RESULTS_DIR.")
_p.add_argument("--out-dir", default=".")
_args = _p.parse_args()

if _args.run_dir:
    run_dir = Path(_args.run_dir)
else:
    base = Path(paths.RESULTS_DIR)
    cands = sorted(base.glob("lightgbm_sat(umap)_clim_soil_wc_*")) or \
        sorted(base.glob("lightgbm_*"))
    if not cands:
        sys.exit(f"No lightgbm run directory found under {base}. "
                 f"Run src/run_ablations.py first, or pass --run-dir.")
    run_dir = cands[-1]

csv_data = run_dir / "per_run_performance_metrics.csv"
if not csv_data.is_file():
    sys.exit(f"{csv_data} not found.")
print(f"reading {csv_data}")
data = pd.read_csv(csv_data)
out_dir = Path(_args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 10))

sns.kdeplot(data=data['r2'], ax=ax, fill=True, color='skyblue', label='Standard r2', alpha=0.5)
sns.kdeplot(data=data['r2_filtered_98_percent'], ax=ax, fill=True, color='salmon', label='Filtered r2 (98%)', alpha=0.5)

mean_r2 = data['r2'].mean()
mean_r2_filtered = data['r2_filtered_98_percent'].mean()

ax.axvline(mean_r2, color='skyblue', linestyle='--', linewidth=2, label=f'Mean r2: {mean_r2:.3f}')
ax.axvline(mean_r2_filtered, color='salmon', linestyle='--', linewidth=2, label=f'Mean Filtered r2: {mean_r2_filtered:.3f}')
ax.set_xlim(0.4, 0.7)

ax.set_title('Comparison of R-squared Distributions', fontsize=16)
ax.set_xlabel('R-squared Value')
ax.set_ylabel('Density')

plt.tight_layout()

map1_path = out_dir / "kde_plot.png"
fig.savefig(map1_path, dpi=600, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(15, 10))

sns.histplot(data=data, x='r2', ax=ax, color='skyblue', label='Standard r2', alpha=0.5, stat='density', bins=10)
sns.histplot(data=data, x='r2_filtered_98_percent', ax=ax, color='salmon', label='Filtered r2 (98%)', alpha=0.5, stat='density', bins=10)

mean_r2 = data['r2'].mean()
mean_r2_filtered = data['r2_filtered_98_percent'].mean()

ax.axvline(mean_r2, color='blue', linestyle='--', linewidth=2, label=f'Mean r2: {mean_r2:.3f}')
ax.axvline(mean_r2_filtered, color='darkred', linestyle='--', linewidth=2, label=f'Mean Filtered r2: {mean_r2_filtered:.3f}')

ax.set_title('Histogram Comparison of R-squared Distributions', fontsize=16)
ax.set_xlabel('R-squared Value')
ax.set_ylabel('Density')
ax.set_xlim(0.45, 0.7)

plt.tight_layout()
map2_path = out_dir / "hist_plot.png"
fig.savefig(map2_path, dpi=600, bbox_inches='tight')
plt.close(fig)
