import os
import pandas as pd
import ee
from util import *
from dotenv import load_dotenv
import geopandas as gpd
import ee
import ee
import os
load_dotenv()

ee.Authenticate()
ee.Initialize(project='gee-sur14')
authenticate_gee_flag = False
download_flag = True
move_to_local_flag = True

export_resolution = 30
patch_size_pixels = 64
patch_size_meters = export_resolution * patch_size_pixels

model_input_width = 5 # number of images before the event
sentinel_2_revisit_time = 5
model_input_width_days = (model_input_width + 2) * sentinel_2_revisit_time
input_date = "2019-07-31"

load_dotenv()

num_stations = 93

dataset_folder = os.path.join('datasets', f'illinois_{num_stations}_stations')
os.makedirs(dataset_folder, exist_ok=True)

pep725_data_folder = r"data"
os.makedirs(pep725_data_folder, exist_ok=True)

project_name_on_gee = os.getenv('GEE_PROJECT_NAME')

# Data folder name
export_folder_name = f'illinois_eo_data'

# The path to Google Drives folder
google_drive_path = os.getenv('GOOGLE_DRIVE_PATH')
eodata_download_folder_path_on_google_cloud = os.path.join(google_drive_path, export_folder_name)
# print(f"Download Folder Path on Google Drive: {eodata_download_folder_path_on_google_cloud}")

# Operating system's download folder
os_download_folder_path = os.getenv('OS_DOWNLOAD_FOLDER_PATH')

eodata_download_folder_path_local = os.path.join(os_download_folder_path, export_folder_name)
eodata_download_folder_path_local = os.path.normpath(eodata_download_folder_path_local)

os.makedirs(eodata_download_folder_path_local, exist_ok=True)


dataset = 'E:/Thesis_DataPrep/Illinois/IL_sampling_2019.csv'
df = pd.read_csv(dataset, sep=',', low_memory=False) # Because columns have mixed data types, pandas suggested to use low_memory=False 
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
) 

station_df = df.groupby('id').id.count().sort_values(ascending=False)
station_list = station_df.index.to_list()


station_list = station_list[:num_stations]
fdf = gdf[gdf.id.isin(station_list)]


if authenticate_gee_flag:
    ee.Authenticate(force=True)
    ee.Initialize(project=project_name_on_gee)

# print(f"Google Drive path: {eodata_download_folder_path_on_google_cloud}")
# print(f"Local OS path: {eodata_download_folder_path_local}")
if download_flag:
    download_sentinel2_modis_from_gee(
        fdf,
        export_folder_name,
        export_resolution,
        model_input_width,
        patch_size_meters,
        model_input_width_days,
        move_to_local_flag,
        eodata_download_folder_path_on_google_cloud,
        eodata_download_folder_path_local,
        input_date = input_date
    )