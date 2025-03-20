'''
Sampling Stations.py
A grid of (256*30) m is created over the state, and then random pairs of row,column are 
generated to extract the sampling points for the patches.
The total number of sampling stations for all states together is 2000.
Number of samples per state per year is calculated based on the area of the state.
There are 5 years in total (2019-2023)
'''
import os
import pandas as pd
import ee
from util import *
from dotenv import load_dotenv
import geopandas as gpd
import ee
import ee
import os
from shapely.geometry import Point
import random

# Authorize the token for GEE
ee.Authenticate()
ee.Initialize(project='gee-sur14')
authenticate_gee_flag = False
download_flag = True
move_to_local_flag = True

# To match the size of CDL, other dataset should be resampled to 30m
export_resolution = 30
patch_size_pixels = 256
patch_size_meters = export_resolution * patch_size_pixels

# state_code for TIGER dataset -  Illinois 17; Nebraska 31; Indiana 18; Iowa 19; Minnesota 27
state_code = "31" 
output_folder = "E:/Thesis_DataPrep/Nebraska/"  
num_samples = 92
num_iterations = 5

states = ee.FeatureCollection("TIGER/2018/States")
state_geometry = states.filter(ee.Filter.eq("STATEFP", state_code)).geometry()

# Create grid - using projection coordinate system 32616 for calculating patch distance
grid = ee.FeatureCollection(state_geometry.coveringGrid(
    ee.Projection("EPSG:32615").atScale(patch_size_meters),
    patch_size_meters,
))

grid_list = grid.toList(grid.size())
total_grid_cells = grid.size().getInfo()

# Each sample - treated independently
# Initialize global ID to have unique IDs across all iterations
global_id = 1

# Rabdom sampling of grids
for iteration in range(1, num_iterations + 1):
    sample_indices = random.sample(range(total_grid_cells), num_samples)

    # Extract centroid of sampled patch and save in WGS84
    centroids = []
    for feature in sample_indices:
        grid_feature = ee.Feature(grid_list.get(feature))
        projected_centroid = grid_feature.geometry().centroid(1).transform("EPSG:32616")
        geographic_centroid = projected_centroid.transform("EPSG:4326").getInfo()
        coordinates = geographic_centroid["coordinates"]

        # Calculate row and column from the grid coordinates
        row, col = divmod(feature, total_grid_cells // patch_size_pixels)

        # Consistent ID for each grid cell based on row and column
        grid_id = f"NE_{row}_{col}"  

        centroids.append({
            "id": grid_id,  
            "lon": coordinates[0],
            "lat": coordinates[1],
        })
        global_id += 1

    # Convert into a GeoDataFrame
    centroids_df = pd.DataFrame(centroids)
    gdf = gpd.GeoDataFrame(
        centroids_df,
        geometry=[Point(x, y) for x, y in zip(centroids_df["lon"], centroids_df["lat"])],
        crs="EPSG:4326"
    )

    # Save output in CSV format
    year = 2019 + iteration - 1
    output_csv = os.path.join(output_folder, f"NE_sampling_{year}.csv")
    gdf.to_csv(output_csv, index=False)
    print(f"Iteration {iteration}: Saved to {output_csv}")