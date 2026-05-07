import pandas as pd
import geopandas
from shapely.geometry import Point
import matplotlib.pyplot as plt


# Load your data from the CSV file
df = pd.read_csv('/maps-priv/maps/ray25/config_src/patch_climate_representation_results_1/lightgbm_sat(umap)_20250827-201410/error_analysis_details.csv')

# Calculate percentage error
df['percent_error'] = (df['absolute_error'] / df['actual_rarefied']) * 100

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))
gdf.set_crs(epsg=4326, inplace=True)
#world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world_url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"

world = geopandas.read_file(world_url)
# --- Visualization 1: Basic Point Map ---

print("Displaying basic plot...")
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# Plot the world map as a light gray background
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

# Plot your data points on top
gdf.plot(ax=ax, marker='o', color='crimson', markersize=25, alpha=0.7)

ax.set_title('Location of Data Samples')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# Focus the map on the general area of your data points for clarity
ax.set_xlim(-20, 180)
ax.set_ylim(0, 80)
plt.savefig('basic_point_map.png', dpi=600, bbox_inches='tight')
plt.close(fig)
# --- Visualization 2: Bubble Map ---
# A more compelling visualization where point size and color encode the 'absolute_error'.
# This immediately shows where the model's error is largest.

print("Displaying bubble map...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

# Plot the data, using the 'absolute_error' column to determine both color and size.
gdf.plot(ax=ax,
         column='absolute_error',    # Data column for color
         cmap='plasma',              # Colormap (see matplotlib for more options)
         markersize=gdf['absolute_error'] / 2, # Scale marker size by error value
         legend=True,
         legend_kwds={'label': "Absolute Error", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Bubble Map of Absolute Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(-20, 180)
ax.set_ylim(0, 80)
plt.savefig('bubble_map_error.png', dpi=600, bbox_inches='tight')
plt.close(fig)

# Bubble map with percentage error
print("Displaying bubble map (% error)...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

# Plot the data, using the 'percent_error' column to determine both color and size.
gdf.plot(ax=ax,
         column='percent_error',    # Data column for color
         cmap='plasma',              # Colormap (see matplotlib for more options)
         markersize=gdf['percent_error'] / 5, # Scale marker size by error value
         legend=True,
         legend_kwds={'label': "Percent Error (%)", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Bubble Map of Percent Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(-20, 180)
ax.set_ylim(0, 80)
plt.savefig('bubble_map_percent_error.png', dpi=600, bbox_inches='tight')
plt.close(fig)


print("Displaying bubble map...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

# Plot the data, using the 'absolute_error' column to determine both color and size.
gdf.plot(ax=ax,
         column='absolute_error',    # Data column for color
         cmap='plasma',              # Colormap (see matplotlib for more options)
         markersize=gdf['absolute_error'] / 2, # Scale marker size by error value
         legend=True,
         legend_kwds={'label': "Absolute Error", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Estonia Map of Absolute Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(20, 30)
ax.set_ylim(55, 60)
plt.savefig('bubble_map_error_estonia.png', dpi=600, bbox_inches='tight')
plt.close(fig)

print("Displaying bubble map (Estonia % error)...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

# Plot the data, using the 'percent_error' column to determine both color and size.
gdf.plot(ax=ax,
         column='percent_error',    # Data column for color
         cmap='plasma',              # Colormap (see matplotlib for more options)
         markersize=gdf['percent_error'] / 5, # Scale marker size by error value
         legend=True,
         legend_kwds={'label': "Percent Error (%)", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Estonia Map of Percent Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(20, 30)
ax.set_ylim(55, 60)
plt.savefig('bubble_map_percent_error_estonia.png', dpi=600, bbox_inches='tight')
plt.close(fig)

print("Displaying bubble map...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

# Plot the data, using the 'absolute_error' column to determine both color and size.
gdf.plot(ax=ax,
         column='absolute_error',    # Data column for color
         cmap='plasma',              # Colormap (see matplotlib for more options)
         markersize=gdf['absolute_error'] / 2, # Scale marker size by error value
         legend=True,
         legend_kwds={'label': "Absolute Error", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Sweden Map of Absolute Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(10, 25)
ax.set_ylim(57, 70)
plt.savefig('bubble_map_error_sweden.png', dpi=600, bbox_inches='tight')
plt.close(fig)

print("Displaying bubble map (Sweden % error)...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
world.plot(ax=ax, color='#e0e0e0', edgecolor='black')

# Plot the data, using the 'percent_error' column to determine both color and size.
gdf.plot(ax=ax,
         column='percent_error',    # Data column for color
         cmap='plasma',              # Colormap (see matplotlib for more options)
         markersize=gdf['percent_error'] / 5, # Scale marker size by error value
         legend=True,
         legend_kwds={'label': "Percent Error (%)", 'orientation': "horizontal", 'pad': 0.01})

ax.set_title('Sweden Map of Percent Error in Predictions')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(10, 25)
ax.set_ylim(57, 70)
plt.savefig('bubble_map_percent_error_sweden.png', dpi=600, bbox_inches='tight')
plt.close(fig)