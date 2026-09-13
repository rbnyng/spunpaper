"""Geographic distribution of prediction error.

Reads one run's per-sample error table. With no --run-dir it picks the most
recent satellite-only LightGBM run under RESULTS_DIR, which is the configuration
the error map in the manuscript is drawn from.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import geopandas
import matplotlib.pyplot as plt

import paths

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--run-dir", default=None,
                help="Run directory containing error_analysis_details.csv. "
                     "Default: newest lightgbm_sat(umap)_* under RESULTS_DIR.")
_p.add_argument("--out-dir", default=".")
_args = _p.parse_args()

if _args.run_dir:
    run_dir = Path(_args.run_dir)
else:
    base = Path(paths.RESULTS_DIR)
    cands = sorted(base.glob("lightgbm_sat(umap)_2*")) or sorted(base.glob("lightgbm_*"))
    if not cands:
        sys.exit(f"No lightgbm run directory found under {base}. "
                 f"Run src/run_ablations.py first, or pass --run-dir.")
    run_dir = cands[-1]

csv_path = run_dir / "error_analysis_details.csv"
if not csv_path.is_file():
    sys.exit(f"{csv_path} not found.")
print(f"reading {csv_path}")
df = pd.read_csv(csv_path)
out_dir = Path(_args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

df['percent_error'] = (df['absolute_error'] / df['actual_rarefied']) * 100

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))
gdf.set_crs(epsg=4326, inplace=True)
# Alternative basemap: swap in the 110m Natural Earth file below for a much
# smaller download when fine coastline detail is not needed.
#world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world_url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"

world = geopandas.read_file(world_url)

print("Displaying basic plot...")
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

gdf.plot(ax=ax, marker='o', color='crimson', markersize=25, alpha=0.7)

ax.set_title('Location of Data Samples')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(-20, 180)
ax.set_ylim(0, 80)
plt.savefig(out_dir / 'basic_point_map.png', dpi=600, bbox_inches='tight')
plt.close(fig)

print("Displaying bubble map...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

gdf.plot(ax=ax,
         column='absolute_error',
         cmap='plasma',
         markersize=gdf['absolute_error'] / 2,
         legend=True,
         legend_kwds={'label': "Absolute Error", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Bubble Map of Absolute Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(-20, 180)
ax.set_ylim(0, 80)
plt.savefig(out_dir / 'bubble_map_error.png', dpi=600, bbox_inches='tight')
plt.close(fig)

print("Displaying bubble map (% error)...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

gdf.plot(ax=ax,
         column='percent_error',
         cmap='plasma',
         markersize=gdf['percent_error'] / 5,
         legend=True,
         legend_kwds={'label': "Percent Error (%)", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Bubble Map of Percent Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(-20, 180)
ax.set_ylim(0, 80)
plt.savefig(out_dir / 'bubble_map_percent_error.png', dpi=600, bbox_inches='tight')
plt.close(fig)
