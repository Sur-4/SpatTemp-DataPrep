# import os
# import pandas as pd
# output_folder = "E:/Thesis_DataPrep/Illinois/"
# csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]

# id_rows_dict = {}
# for csv_file in csv_files:
#     file_path = os.path.join(output_folder, csv_file)
#     df = pd.read_csv(file_path)
#     for idx, row in df.iterrows():
#         grid_id = row['id']  
#         if grid_id in id_rows_dict:
#             id_rows_dict[grid_id].append((csv_file, idx))  
#         else:
#             id_rows_dict[grid_id] = [(csv_file, idx)]  

# for grid_id, rows in id_rows_dict.items():
#     print(f"ID: {grid_id}")
#     for file_name, row_index in rows:
#         print(f"  File: {file_name}, Row: {row_index}")

import ee
import geemap

# Authenticate and initialize GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Define a point in Illinois (Example: near Champaign, IL)
illinois_point = ee.Geometry.Point([-88.2434, 40.1164])  # Replace with your coordinates

# Define the year (2019)
year = 2019
start_date = f"{year}-01-01"
end_date = f"{year}-12-31"

# Load CDL image collection for 2019
collection = (ee.ImageCollection('USDA/NASS/CDL')
              .filterBounds(illinois_point)
              .filterDate(start_date, end_date)
              .select('cropland')
              )

# Get the image for 2019
image = collection.mosaic()  # Merges tiles into a single image
# Get the number of images in the collection
image_count = collection.size().getInfo()

# Print the number of images
print(f"Number of images in the collection: {image_count}")
# # Define export parameters
# file_name = "CDL_Illinois_2019"
# download_path = f"{file_name}.tif"

# # Set scale (30m) and region (small buffer around point)
# region = illinois_point.buffer(5000).bounds()  # 5 km buffer around point

# # Download the image using geemap
# geemap.ee_export_image(image, filename=download_path, scale=30, region=region, file_per_band=False)

# print(f"Download complete: {download_path}")
