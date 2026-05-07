import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

csv_data = "/maps-priv/maps/ray25/config_src/patch_climate_representation_results/lightgbm_umap_runs_20250701-202108/per_run_performance_metrics.csv"
data = pd.read_csv(csv_data)

fig, ax = plt.subplots(figsize=(15, 10))

sns.kdeplot(data=data['r2'], ax=ax, fill=True, color='skyblue', label='Standard r2', alpha=0.5)
sns.kdeplot(data=data['r2_filtered_98_percent'], ax=ax, fill=True, color='salmon', label='Filtered r2 (98%)', alpha=0.5)

mean_r2 = data['r2'].mean()
mean_r2_filtered = data['r2_filtered_98_percent'].mean()

ax.axvline(mean_r2, color='skyblue', linestyle='--', linewidth=2, label=f'Mean r2: {mean_r2:.3f}')
ax.axvline(mean_r2_filtered, color='salmon', linestyle='--', linewidth=2, label=f'Mean Filtered r2: {mean_r2_filtered:.3f}')
ax.set_xlim(0.4, 0.7)
#ax.set_ylim(0, 80)

ax.set_title('Comparison of R-squared Distributions', fontsize=16)
ax.set_xlabel('R-squared Value')
ax.set_ylabel('Density')

# ax.legend()

plt.tight_layout()

map1_path = "kde_plot.png"
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

#ax.legend()

plt.tight_layout()
map2_path = "hist_plot.png"
fig.savefig(map2_path, dpi=600, bbox_inches='tight')
plt.close(fig)
