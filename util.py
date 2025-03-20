import os
from tqdm.notebook import tqdm  # a progress bar
import ee
from datetime import datetime
import pyproj
from shapely.geometry import box
from pyproj import Transformer, CRS
from datetime import datetime, timedelta
from tqdm.auto import tqdm  # Importing the auto version which is notebook-friendly
import time
import shutil
import os
import time
import glob
from pathlib import Path
import re
from datetime import datetime
from tqdm.notebook import tqdm
import rasterio
from rasterio.enums import Resampling
import numpy as np
from scipy.interpolate import griddata
import os
import numpy as np
import rasterio
import torch
from tqdm import tqdm
from datetime import datetime


def resample_geotiff_in_place(file_path, new_width=64, new_height=64):
    """
    Resample a GeoTIFF file to a new dimension (new_width x new_height)
    and overwrite the original file while preserving its georeferencing information.

    Parameters:
    - file_path: str, path to the GeoTIFF file to be resampled and overwritten
    - new_width: int, the target width in pixels
    - new_height: int, the target height in pixels
    """
    with rasterio.open(file_path, "r") as src:
        if src.width != new_width or src.height != new_height:
            # Read the metadata of the source file
            src_meta = src.meta.copy()

            # Calculate the new transform and update metadata
            new_transform = src.transform * src.transform.scale(
                (src.width / new_width), (src.height / new_height)
            )

            # Read the data and resample it
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.bilinear,
            )

            # Update the metadata to reflect the new dimensions and transform
            src_meta.update(
                {"height": new_height, "width": new_width, "transform": new_transform}
            )

            # Write the resampled data to the file, overwriting the original
            with rasterio.open(file_path, "w", **src_meta) as dst:
                dst.write(data)


def cancel_running_tasks():
    # Fetch the list of all tasks.
    tasks = ee.data.getTaskList()
    any_cancellation = False
    for task in tasks:
        # Check if the task is in RUNNING state.
        if task["state"] == "RUNNING":
            print(f"Canceling task {task['id']} - {task['description']}")
            # Cancel the task based on its ID.
            ee.data.cancelTask(task["id"])
            any_cancellation = True
    return any_cancellation


def monitor_and_cancel_running_tasks(timeout):
    # Initialize the Earth Engine module.
    ee.Initialize()

    start_time = time.time()
    last_cancel_time = start_time  # Initialize with the start time
    next_check = 2
    while True:
        current_time = time.time()
        if current_time - last_cancel_time > timeout:
            print(f"No new cancel for {timeout} seconds. Stopping.")
            break
        if cancel_running_tasks():
            last_cancel_time = time.time()
        print(f"Wait for {next_check} seconds")
        time.sleep(next_check)


def move_downloaded_files(source_dir_prefix, destination_dir):

    os.makedirs(destination_dir, exist_ok=True)
    print(f"Moving files from {source_dir_prefix}(s) to {destination_dir}")

    # Use glob to find all directories that start with source_dir_prefix
    for source_dir in glob.glob(f"{source_dir_prefix}*"):

        if not os.path.isdir(source_dir):
            continue  # Skip if not a directory

        for filename in os.listdir(source_dir):
            if filename.endswith(".tif"):  # Assuming GeoTIFF format; adjust as needed
                source_path = os.path.join(source_dir, filename)
                destination_path = os.path.join(destination_dir, filename)
                shutil.move(source_path, destination_path)
                # delete the file in the source folder if it's still there
                file_path = Path(source_path)
                file_path.unlink(missing_ok=True)  # Python 3.8+
                print(".", end="", flush=True)
    print()
    print("File moved to the destination folder.")


def get_previous_date_range(input_date_str, days_before=30):
    """
    Returns two dates: one 1 day before and the other N days before the input date.
    Args:
    - input_date_str (str): The input date in 'YYYY-MM-DD' format.
    - days_before (int): The number of days before the input date to calculate. Default is 30.

    Returns:
    - Tuple of str: (date_one_day_before, date_n_days_before) in 'YYYY-MM-DD' format.
    """
    # Convert the input string to a datetime object
    input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
    # Calculate dates
    one_day_before = input_date - timedelta(days=1)
    n_days_before = input_date - timedelta(days=days_before)

    # Convert dates back to strings
    one_day_before_str = one_day_before.strftime("%Y-%m-%d")
    n_days_before_str = n_days_before.strftime("%Y-%m-%d")

    return (one_day_before_str, n_days_before_str)


def get_bounding_box(point, buffer_dist=600):
    # Define the WGS84 and UTM projection, UTM will be determined dynamically for accuracy
    wgs84 = pyproj.CRS("EPSG:4326")

    # Determine the UTM zone dynamically for the given point
    utm_zone = int((point.x + 180) / 6) + 1
    utm_crs = CRS(f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs")

    # Initialize transformers
    transformer_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    # Transform point to UTM
    point_utm_x, point_utm_y = transformer_to_utm.transform(point.x, point.y)

    # Create bounding box in UTM
    buffer_distance = 500  # Half of 1000 meters to create the box around the point
    bbox_utm = box(
        point_utm_x - buffer_distance,
        point_utm_y - buffer_distance,
        point_utm_x + buffer_distance,
        point_utm_y + buffer_distance,
    )

    # Get corners of the bounding box in UTM, then transform back to WGS84
    bottom_left_x, bottom_left_y = transformer_to_wgs84.transform(
        bbox_utm.bounds[0], bbox_utm.bounds[1]
    )
    top_right_x, top_right_y = transformer_to_wgs84.transform(
        bbox_utm.bounds[2], bbox_utm.bounds[3]
    )

    return (bottom_left_y, bottom_left_x, top_right_y, top_right_x)


def wait_for_tasks_to_complete(tasks, max_wait_time=3600):
    """
    Waits for all tasks in the list to complete or fail, up to a maximum wait time.

    Parameters:
    - tasks: List of ee.batch.Task objects
    - max_wait_time: Maximum time to wait in seconds
    """
    start_time = time.time()
    while True:
        all_finished = all(
            task.status()["state"] in ["COMPLETED", "FAILED", "CANCELLED"]
            for task in tasks
        )
        if all_finished:
            print("All tasks in the batch have completed.")
            break
        if (time.time() - start_time) > max_wait_time:
            print("Reached maximum wait time. Some tasks may still be running.")
            break
        time.sleep(30)  # Check every 30 seconds


def execute_tasks_in_batches(tasks_list, batch_size=20):
    """
    Executes tasks in batches, waiting for each batch to complete before starting the next.

    Parameters:
    - tasks_list: List of all tasks to be executed
    - batch_size: Number of tasks to execute in each batch
    """
    print(f"Waiting for the download tasks to complete")
    for i in range(0, len(tasks_list), batch_size):
        batch = tasks_list[i : i + batch_size]
        print(
            f"Download batch {i//batch_size + 1}/{(len(tasks_list) - 1)//batch_size + 1}"
        )
        for task in batch:
            task.start()
            print(".", end="", flush=True)
        print()
        wait_for_tasks_to_complete(batch)


def extract_data_from_sentinel2_datatake_identifier(datatake_identifier):
    # Regular expression to match the string format
    pattern = re.compile(
        r"^(?P<satelliteCode>[A-Z0-9]+)_(?P<date>\d{8})T(?P<time>\d{6})_(?P<identifier>\d+)_(?P<version>N\d+\.\d+)$"
    )
    match = pattern.match(datatake_identifier)

    if match:
        # Extracting data using named groups
        data = match.groupdict()

        # Formatting date and time for easier readability
        data["date"] = f"{data['date'][:4]}-{data['date'][4:6]}-{data['date'][6:]}"
        data["time"] = f"{data['time'][:2]}:{data['time'][2:4]}:{data['time'][4:]}"

        return data["date"]
    else:
        return None


def get_resampled_image(image, sentinel2_bands, export_resolution):
    """
    Resample Sentinel-2 bands to the same resolution and combine them.
    Assumes sentinel2_bands is a list of band names to be selected and resampled.
    """
    # Bands with their original resolutions
    bands_10m = ["B2", "B3", "B4", "B8"]
    bands_20m = ["B5", "B6", "B7", "B8A", "B11", "B12"]

    # Select and resample 20m bands to 10m
    resampled_20m = (
        image.select(bands_20m)
        .resample("bilinear")
        .reproject(crs=image.select(bands_10m[0]).projection(), scale=export_resolution)
    )

    # Select 10m bands without resampling
    bands_10m_image = image.select(bands_10m)

    # Combine all selected and resampled bands
    combined_image = bands_10m_image.addBands(resampled_20m)

    return combined_image


def load_files_list(download_folder_path_on_google_cloud, download_folder_path_local):
    unique_names = set()  # Using a set to avoid duplicates

    # Iterate over all files in the given folder
    if os.path.isdir(download_folder_path_on_google_cloud):
        for file_name in os.listdir(download_folder_path_on_google_cloud):
            # Check if the file is a TIFF file and extract the base name
            if (
                file_name.endswith("_sentinel2.tif")
                or file_name.endswith("_sentinel1.tif")
                or file_name.endswith("_modis.tif")
                or file_name.endswith("_cdl.tif")
            ):
                base_name = file_name.rsplit(".", 1)[
                    0
                ]  # Split on the last underscore and take the first part
                unique_names.add(base_name)  # Add the name to the set

    if os.path.isdir(download_folder_path_local):
        for file_name in os.listdir(download_folder_path_local):
            # Check if the file is a TIFF file and extract the base name
            if (
                file_name.endswith("_sentinel2.tif")
                or file_name.endswith("_sentinel1.tif")
                or file_name.endswith("_modis.tif")
                or file_name.endswith("_cdl.tif")
            ):
                base_name = file_name.rsplit(".", 1)[
                    0
                ]  # Split on the last underscore and take the first part
                unique_names.add(base_name)  # Add the name to the set

    return list(unique_names)


def get_modis_lst_download_task(
    date, data_bbox, export_resolution, file_prefix, export_folder_name_on_gee
):
    # Define the MODIS LST dataset ID.
    dataset_id = "MODIS/006/MOD11A1"

    # Create the date filter.
    start_date = ee.Date(date)
    end_date = start_date.advance(1, "day")

    # Load the dataset and filter.
    lst_dataset = (
        ee.ImageCollection(dataset_id)
        .filterDate(start_date, end_date)
        .filterBounds(data_bbox)
    )

    # Assuming filtering leaves one image, select it.
    lst_image = ee.Image(
        lst_dataset.first()
    )  # This ensures we're working with a single image

    # Convert 'LST_Day_1km' and 'LST_Night_1km' from Kelvin to Celsius.
    day_temp = (
        lst_image.select("LST_Day_1km").multiply(0.02).subtract(273.15).toFloat()
    )  # Kelvin to Celsius
    night_temp = (
        lst_image.select("LST_Night_1km").multiply(0.02).subtract(273.15).toFloat()
    )  # Kelvin to Celsius

    # Create a new image from the day and night temperature bands.
    lst_combined = ee.Image.cat(
        [day_temp.rename("LST_Day_Celsius"), night_temp.rename("LST_Night_Celsius")]
    )

    # Resample and reproject the image
    # Note: This step assumes the sentinel2_bands variable from previous context; adjust as necessary.
    lst_resampled_reprojected = lst_combined.resample("bilinear").reproject(
        crs=ee.Projection("EPSG:4326").atScale(export_resolution)
    )

    # Define export parameters with the resampled and reprojected image.
    export_params = {
        "image": lst_resampled_reprojected,
        "description": f"MODIS LST {date}",
        "scale": export_resolution,
        "region": data_bbox.coordinates().getInfo(),
        "fileFormat": "GeoTIFF",
        "folder": export_folder_name_on_gee,
        "crs": "EPSG:4326",
        "fileNamePrefix": file_prefix,
        "maxPixels": 1e9,
    }

    # Start the export to Google Drive.
    task = ee.batch.Export.image.toDrive(**export_params)
    return task


def get_ncep_ncar_download_task(
    date, data_bbox, export_resolution, file_prefix, export_folder_name_on_gee
):
    # Define the NCEP/NCAR Reanalysis dataset ID.
    dataset_id = (
        "NOAA/CFSV2/FOR6H"  # Example: change to a specific NCEP/NCAR dataset as needed
    )

    # Create the date filter.
    start_date = ee.Date(date)
    end_date = start_date.advance(1, "day")

    # Load the dataset and filter.
    ncep_dataset = (
        ee.ImageCollection(dataset_id)
        .filterDate(start_date, end_date)
        .filterBounds(data_bbox)
    )

    # Assuming filtering leaves one image, select it.
    ncep_image = ee.Image(
        ncep_dataset.first()
    )  # This ensures we're working with a single image

    ncep_selected_bands = ncep_image.select(["Temperature_height_above_ground"])

    # Resample and reproject the image
    ncep_resampled_reprojected = ncep_selected_bands.resample("bilinear").reproject(
        crs=ee.Projection("EPSG:4326").atScale(export_resolution)
    )

    # Define export parameters with the resampled and reprojected image.
    export_params = {
        "image": ncep_resampled_reprojected,
        "description": f"NCEP NCAR Reanalysis {date}",
        "scale": export_resolution,
        "region": data_bbox.coordinates().getInfo(),
        "fileFormat": "GeoTIFF",
        "folder": export_folder_name_on_gee,
        "crs": "EPSG:4326",
        "fileNamePrefix": file_prefix,
        "maxPixels": 1e9,
    }

    # Start the export to Google Drive.
    task = ee.batch.Export.image.toDrive(**export_params)
    return task


def get_usda_crop_type_cover_download_task(
    data_bbox, file_prefix, export_folder_name_on_gee
):
    # Define the year (2019)
    year = 2019
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    region = data_bbox  

    collection = (ee.ImageCollection('USDA/NASS/CDL')
                  .filterBounds(region) 
                  .filterDate(start_date, end_date)  
                  .select('cropland')  
                  )
    clc_dataset = collection.mosaic().clip(region)  

    # Define the crop classes for Corn (1) and Soybeans (5)
    crop_classes = [1, 5]  

    # Create a mask for the crops 
    crop_mask = ee.Image(0)
    for crop_class in crop_classes:
        crop_mask = crop_mask.Or(clc_dataset.eq(crop_class))

    # Mask dataset using crop mask (Corn and Soybeans)
    clc_dataset_filtered = clc_dataset.updateMask(crop_mask)

    # Define export parameters with the image.
    export_params = {
        "image": clc_dataset_filtered,
        "description": f"USDA_Crop_Cover_2019",
        "scale": 30,
        "region": region,
        "fileFormat": "GeoTIFF",
        "folder": export_folder_name_on_gee,
        "crs": "EPSG:4326",
        "fileNamePrefix": file_prefix,
        "maxPixels": 1e13,
    }

    # Start the export to Google Drive.
    task = ee.batch.Export.image.toDrive(**export_params)
    # task.start()
    # print(f"Export task started: {task.status()}")
    return task


def get_sentinel1_grd_download_task(
    start_date,
    end_date,
    data_bbox,
    export_resolution,
    file_prefix,
    export_folder_name_on_gee,
):
    # Define the Sentinel-1 GRD dataset ID.
    dataset_id = "COPERNICUS/S1_GRD"

    # Load the dataset and filter by date and bounds.
    s1_dataset = (
        ee.ImageCollection(dataset_id)
        .filterDate(start_date, end_date)
        .filterBounds(data_bbox)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    # Assuming filtering leaves one image, select it.
    s1_image = ee.Image(s1_dataset.first())

    # Select the VV and VH bands.
    s1_vv_vh = s1_image.select(["VV", "VH"])

    # Resample and reproject the image
    s1_resampled_reprojected = s1_vv_vh.resample("bilinear").reproject(
        crs=ee.Projection("EPSG:4326").atScale(export_resolution)
    )

    # Define export parameters with the resampled and reprojected image.
    export_params = {
        "image": s1_resampled_reprojected,
        "description": f"Sentinel-1 GRD VV and VH {start_date} to {end_date}",
        "scale": export_resolution,
        "region": data_bbox.coordinates().getInfo(),
        "fileFormat": "GeoTIFF",
        "folder": export_folder_name_on_gee,
        "crs": "EPSG:4326",
        "fileNamePrefix": file_prefix,
        "maxPixels": 1e9,
    }

    # Start the export to Google Drive.
    task = ee.batch.Export.image.toDrive(**export_params)
    return task


def download_sentinel2_modis_from_gee(
    fdf,
    export_folder_name_on_gee,
    export_resolution,
    model_input_width,
    patch_size_meters,
    model_input_width_days,
    move_to_local_folder=False,
    download_folder_path_on_google_cloud=None,
    download_folder_path_local=None,
    download_if_not_exit=True,
    input_date = None
):
    ee.Initialize()

    files_list = []
    if download_if_not_exit:
        files_list = load_files_list(
            download_folder_path_on_google_cloud, download_folder_path_local
        )

    for index, row in tqdm(fdf.iterrows(), total=fdf.shape[0]):

        retry = True
        while retry:
            try:

                retry = False
                tasks_list = []
                point = row.geometry
                # date = row.date
                station_id = row['id']
                # record_id = row['id']
                date = input_date

                print(
                    f"Station {station_id} on {date} - download started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                (min_lat, min_lon, max_lat, max_lon) = get_bounding_box(
                    point, patch_size_meters / 2
                )
                (end_date, start_date) = get_previous_date_range(
                    date, model_input_width_days
                )

                data_bbox = ee.Geometry.Polygon(
                    [
                        [min_lon, min_lat],  # Bottom left
                        [min_lon, max_lat],  # Top left
                        [max_lon, max_lat],  # Top right
                        [max_lon, min_lat],  # Bottom right
                        [min_lon, min_lat],  # Closing the polygon
                    ],
                    proj="epsg:4326",
                    geodesic=False,
                )

                sentinel2_bands = [
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "B7",
                    "B8",
                    "B8A",
                    "B11",
                    "B12",
                ]  # Define your band selection here

                s2 = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(data_bbox)
                    .filterDate(start_date, end_date)
                    .map(
                        lambda img: get_resampled_image(
                            img, sentinel2_bands, export_resolution
                        )
                    )
                )

                image_list = s2.toList(s2.size())

                num_images = s2.size().getInfo()

                if num_images > model_input_width:
                    for i in range(num_images):
                        img = ee.Image(image_list.get(i))
                        img_info = img.getInfo()
                        img_id = img_info["properties"]["GRANULE_ID"]
                        datatake_identifier = img_info["properties"][
                            "DATATAKE_IDENTIFIER"
                        ]
                        img_date = extract_data_from_sentinel2_datatake_identifier(
                            datatake_identifier
                        )
                        file_prefix = (
                            f"sid_{station_id}_event_{date}_imgdate_{img_date}"
                        )

                        sentinel2_prefix = f"{file_prefix}_sentinel2"
                        if sentinel2_prefix not in files_list:
                            task_sentinel = ee.batch.Export.image.toDrive(
                                image=img,
                                description=f"Sentinel 2 - {date}",
                                folder=export_folder_name_on_gee,
                                fileNamePrefix=sentinel2_prefix,
                                scale=export_resolution,
                                region=data_bbox.coordinates().getInfo(),
                                fileFormat="GeoTIFF",
                                crs="EPSG:4326",
                                maxPixels=1e9,
                            )
                            tasks_list.append(task_sentinel)
                            files_list.append(sentinel2_prefix)

                        sentinel1_prefix = f"{file_prefix}_sentinel1"
                        if sentinel1_prefix not in files_list:
                            task_sentinel1 = get_sentinel1_grd_download_task(
                                start_date,
                                end_date,
                                data_bbox,
                                export_resolution,
                                sentinel1_prefix,
                                export_folder_name_on_gee,
                            )
                            tasks_list.append(task_sentinel1)
                            files_list.append(sentinel1_prefix)

                        modis_prefix = f"{file_prefix}_modis"
                        if modis_prefix not in files_list:
                            task_modis = get_modis_lst_download_task(
                                img_date,
                                data_bbox,
                                export_resolution,
                                modis_prefix,
                                export_folder_name_on_gee,
                            )
                            tasks_list.append(task_modis)
                            files_list.append(modis_prefix)

                        ncep_prefix = f"{file_prefix}_ncep_ncar"
                        if ncep_prefix not in files_list:
                            task_ncep = get_ncep_ncar_download_task(
                                img_date,
                                data_bbox,
                                export_resolution,
                                ncep_prefix,
                                export_folder_name_on_gee,
                            )
                            tasks_list.append(task_ncep)
                            files_list.append(ncep_prefix)

                    cdl_prefix = f"sid_{station_id}_cdl"
                    if cdl_prefix not in files_list:
                        task_cdl = get_usda_crop_type_cover_download_task(
                            data_bbox, cdl_prefix, export_folder_name_on_gee
                        )
                        tasks_list.append(task_cdl)
                        files_list.append(cdl_prefix)

                if len(tasks_list) > 0:
                    execute_tasks_in_batches(tasks_list)

                    if move_to_local_folder:
                        if (
                            download_folder_path_on_google_cloud is None
                            or download_folder_path_local is None
                        ):
                            raise Exception(
                                "download_folder_path_local and/or download_folder_path_local are not specified."
                            )

                        move_downloaded_files(
                            download_folder_path_on_google_cloud,
                            download_folder_path_local,
                        )
                print(
                    f"Station {station_id} on {date} - download ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                retry = False
            except Exception as e:
                print(f"Encountered an error for station {station_id} on {date}: {e}")
                print("Waiting for 5 minutes before retrying...")
                time.sleep(300)
                print("Retrying...")


def interpolate_nan_for_band(data, band_num):
    """
    Interpolates or extrapolates NaN values for the specified band across time steps.

    Args:
    data: The input tensor of shape (time_steps, width, height, bands).
    band_num: The band number where NaN values need to be interpolated/extrapolated.

    Returns:
    data: The input tensor with NaN values in the specified band interpolated or extrapolated.
    """
    # Interpolate NaNs for the specified band across all time steps
    for t in range(
        data.shape[0]
    ):  # Iterate over time steps (e.g., 6 time steps in your case)
        band_data = data[t, :, :, band_num]  # Get the specific band at time step t

        # Check if the entire band is NaN
        if np.isnan(band_data).all():
            # If the entire band is NaN, fill it with a default value, e.g., 0.0
            data[t, :, :, band_num] = np.nan_to_num(band_data, nan=0.0)
        else:
            # Create a mask for NaN values
            nan_mask = np.isnan(band_data)
            if np.any(nan_mask):
                # Get the coordinates of non-NaN and NaN values
                non_nan_coords = np.array(np.where(~nan_mask)).T
                nan_coords = np.array(np.where(nan_mask)).T

                # Get the values of the non-NaN points
                non_nan_values = band_data[~nan_mask]

                # Use griddata to interpolate/extrapolate NaN values
                band_data[nan_mask] = griddata(
                    points=non_nan_coords,
                    values=non_nan_values,
                    xi=nan_coords,
                    method="nearest",
                )
                # Replace the original NaN band with interpolated values
                data[t, :, :, band_num] = band_data
    return data


# def load_and_stack_geotiffs_in_folder_for_keras(
#     fdf,
#     src_folder_path,
#     dest_file_path,
#     model_input_width,
#     cdl_landcover_class_code,
#     acceptable_landcover_coverage,
# ):
#     data = {}
#     fdf["valid"] = False
#     for index, row in tqdm(fdf.iterrows(), total=fdf.shape[0]):
#         date = row.date
#         station_id = row.s_id
#         record_id = row.record_id
#         arr = load_and_stack_geotiffs_for_keras(
#             station_id,
#             date,
#             src_folder_path,
#             model_input_width,
#             cdl_landcover_class_code,
#             acceptable_landcover_coverage,
#         )
#         if arr is not None:
#             data[record_id] = arr
#             fdf.loc[fdf.s_id == station_id, "valid"] = True

#     np.savez_compressed(dest_file_path, **data)

#     fdf = fdf[fdf["valid"] == True]

#     fdf = fdf.drop(columns=["valid"])

#     return fdf


# def load_and_stack_geotiffs_for_keras(
#     station_id,
#     date,
#     folder_path,
#     num_slices,
#     cdl_landcover_class_code,
#     acceptable_landcover_coverage,
# ):
#     # Initialize a list to hold the 3D arrays for each file
#     arrays = []
#     prefix = f"sid_{station_id}_event_{date}_"
#     prefix = prefix.lower()
#     cdl_file_path = os.path.join(folder_path, f"sid_{station_id}_cdl.tif")

#     sentinel2_files = {}
#     sentinel1_files = {}
#     modis_files = {}
#     for f in os.listdir(folder_path):
#         if f.lower().startswith(prefix):
#             # Extract the datetime string from the filename
#             datetime_str = f.split("_")[-2].split(".")[
#                 0
#             ]  # Get the second last part before '.tif'

#             sensor_str = f.split("_")[-1].split(".")[
#                 0
#             ]  # Get the last part before '.tif'

#             if sensor_str == "sentinel2":
#                 sentinel2_files[datetime_str] = f
#             elif sensor_str == "sentinel1":
#                 sentinel1_files[datetime_str] = f
#             elif sensor_str == "modis":
#                 modis_files[datetime_str] = f
#             else:
#                 pass

#     # Sort the files based on their date
#     sentinel2_files = dict(sorted(sentinel2_files.items(), reverse=False))
#     sentinel1_files = dict(sorted(sentinel1_files.items(), reverse=False))
#     modis_files = dict(sorted(modis_files.items(), reverse=False))

#     file_keys = list(sentinel2_files.keys())

#     if len(file_keys) >= num_slices:
#         with rasterio.open(cdl_file_path) as cdl_src:
#             cdl_array = cdl_src.read(1)  # Assuming the cdl data is single-band
#             mask = cdl_array == cdl_landcover_class_code

#             # Create a binary raster (1 where true, 0 where false)
#             binary_raster = mask.astype(np.uint8)

#             # Check the sequence
#             valid_sequence = True
#             file_keys_dates = [datetime.strptime(key, "%Y-%m-%d") for key in file_keys]
#             for i in range(len(file_keys) - num_slices, len(file_keys) - 1):
#                 valid_sequence = (
#                     valid_sequence
#                     and (file_keys_dates[i] - file_keys_dates[i - 1]).days <= 10
#                 )

#             if not valid_sequence:
#                 return None

#             # Check landcover coverage
#             total_pixels = mask.size
#             covered_pixels = np.sum(mask)
#             covered_percentage = (covered_pixels / total_pixels) * 100
#             if covered_percentage < acceptable_landcover_coverage:
#                 return None

#             for i in range(len(file_keys) - num_slices, len(file_keys)):
#                 file_key = file_keys[i]
#                 sentinel2_file_name = sentinel2_files[file_key]
#                 sentinel1_file_name = sentinel1_files[file_key]
#                 modis_file_name = modis_files[file_key]
#                 sentinel2_file_path = os.path.join(folder_path, sentinel2_file_name)
#                 sentinel1_file_path = os.path.join(folder_path, sentinel1_file_name)
#                 modis_file_path = os.path.join(folder_path, modis_file_name)

#                 with rasterio.open(sentinel2_file_path) as sentinel2_src:
#                     # Read all bands from the GeoTIFF file
#                     sentinel2_array = sentinel2_src.read()
#                     sentinel2_array = np.moveaxis(
#                         sentinel2_array, 0, -1
#                     )  # Move channels to last dimension

#                     with rasterio.open(modis_file_path) as modis_src:
#                         modis_array = modis_src.read()
#                         modis_array = np.moveaxis(
#                             modis_array, 0, -1
#                         )  # Move channels to last dimension

#                         with rasterio.open(sentinel1_file_path) as sentinel1_src:
#                             sentinel1_array = sentinel1_src.read()
#                             sentinel1_array = np.moveaxis(
#                                 sentinel1_array, 0, -1
#                             )  # Move channels to last dimension

#                             # Concatenate along the last axis (channels)
#                             combined_array = np.concatenate(
#                                 (sentinel2_array, sentinel1_array, modis_array), axis=-1
#                             )

#                             # Apply the mask
#                             # combined_array[~mask] = np.nan

#                             # Add the binary raster as the first band
#                             binary_raster_expanded = np.expand_dims(
#                                 binary_raster, axis=-1
#                             )  # Expand dims to match shape
#                             combined_array_with_binary = np.concatenate(
#                                 (binary_raster_expanded, combined_array), axis=-1
#                             )

#                             arrays.append(combined_array_with_binary)

#             # Stack arrays to form a single 4D array (time_steps, height, width, channels)
#             stacked_array = np.stack(arrays)

#             if stacked_array.size > 0:
#                 stacked_array = interpolate_nan_for_band(stacked_array, 13)
#                 stacked_array = interpolate_nan_for_band(stacked_array, 14)
#                 return stacked_array

#     return None

DEFAULT_SENSORS = ["sentinel1", "sentinel2", "modis", "ncep_ncar"]


def load_and_stack_geotiffs_in_folder_for_pytorch(
    fdf,
    src_folder_path,
    dest_file_path,
    model_input_width,
    cdl_landcover_class_code,
    acceptable_landcover_coverage,
    included_sensors=None,
):
    included_sensors = included_sensors or DEFAULT_SENSORS
    data = {}
    fdf["valid"] = False
    for index, row in tqdm(fdf.iterrows(), total=fdf.shape[0]):
        date = row.date
        station_id = row.s_id
        record_id = row.record_id
        arr = load_and_stack_geotiffs_for_pytorch(
            station_id,
            date,
            src_folder_path,
            model_input_width,
            cdl_landcover_class_code,
            acceptable_landcover_coverage,
            included_sensors,
        )
        if arr is not None:
            data[record_id] = arr
            fdf.loc[fdf.s_id == station_id, "valid"] = True

    torch.save(data, dest_file_path)

    fdf = fdf[fdf["valid"] == True]
    fdf = fdf.drop(columns=["valid"])

    return fdf


def load_and_stack_geotiffs_for_pytorch(
    station_id,
    date,
    folder_path,
    num_slices,
    cdl_landcover_class_code,
    acceptable_landcover_coverage,
    included_sensors=None,
):
    included_sensors = included_sensors or DEFAULT_SENSORS
    arrays = []
    prefix = f"sid_{station_id}_event_{date}_".lower()
    cdl_file_path = os.path.join(folder_path, f"sid_{station_id}_cdl.tif")

    sensor_files = {sensor: {} for sensor in included_sensors}

    for f in os.listdir(folder_path):
        if f.lower().startswith(prefix):
            datetime_str = f.split("_")[-2].split(".")[0]
            sensor_str = f.split("_")[-1].split(".")[0]

            if sensor_str in included_sensors:
                sensor_files[sensor_str][datetime_str] = f

    for sensor in included_sensors:
        sensor_files[sensor] = dict(sorted(sensor_files[sensor].items(), reverse=False))

    file_keys = list(sensor_files[included_sensors[0]].keys())

    if len(file_keys) >= num_slices:
        with rasterio.open(cdl_file_path) as cdl_src:
            cdl_array = cdl_src.read(1)
            mask = cdl_array == cdl_landcover_class_code
            binary_raster = mask.astype(np.uint8)

            valid_sequence = True
            file_keys_dates = [datetime.strptime(key, "%Y-%m-%d") for key in file_keys]
            for i in range(len(file_keys) - num_slices, len(file_keys) - 1):
                valid_sequence = (
                    valid_sequence
                    and (file_keys_dates[i] - file_keys_dates[i - 1]).days <= 10
                )

            if not valid_sequence:
                return None

            total_pixels = mask.size
            covered_pixels = np.sum(mask)
            covered_percentage = (covered_pixels / total_pixels) * 100
            if covered_percentage < acceptable_landcover_coverage:
                return None

            for i in range(len(file_keys) - num_slices, len(file_keys)):
                file_key = file_keys[i]
                combined_arrays = []

                for sensor in included_sensors:
                    file_name = sensor_files[sensor].get(file_key)
                    if file_name:
                        file_path = os.path.join(folder_path, file_name)
                        with rasterio.open(file_path) as src:
                            sensor_array = src.read()
                            combined_arrays.append(sensor_array)

                combined_array = np.concatenate(combined_arrays, axis=0)
                binary_raster_expanded = np.expand_dims(binary_raster, axis=0)
                combined_array_with_binary = np.concatenate(
                    (binary_raster_expanded, combined_array), axis=0
                )

                arrays.append(combined_array_with_binary)

            stacked_array = np.stack(arrays)

            if stacked_array.size > 0:
                stacked_array = torch.tensor(stacked_array, dtype=torch.float32)
                return stacked_array

    return None


def load_and_stack_geotiffs_in_folder_for_keras(
    fdf,
    src_folder_path,
    dest_file_path,
    model_input_width,
    cdl_landcover_class_code,
    acceptable_landcover_coverage,
    included_sensors=None,
):
    included_sensors = included_sensors or DEFAULT_SENSORS
    data = {}
    fdf["valid"] = False
    for index, row in tqdm(fdf.iterrows(), total=fdf.shape[0]):
        date = row.date
        station_id = row.s_id
        record_id = row.record_id
        arr = load_and_stack_geotiffs_for_keras(
            station_id,
            date,
            src_folder_path,
            model_input_width,
            cdl_landcover_class_code,
            acceptable_landcover_coverage,
            included_sensors,
        )
        if arr is not None:
            data[record_id] = arr
            fdf.loc[fdf.s_id == station_id, "valid"] = True

    np.savez_compressed(dest_file_path, **data)

    fdf = fdf[fdf["valid"] == True]
    fdf = fdf.drop(columns=["valid"])

    return fdf


def load_and_stack_geotiffs_for_keras(
    station_id,
    date,
    folder_path,
    num_slices,
    cdl_landcover_class_code,
    acceptable_landcover_coverage,
    included_sensors=None,
):
    included_sensors = included_sensors or DEFAULT_SENSORS
    arrays = []
    prefix = f"sid_{station_id}_event_{date}_".lower()
    cdl_file_path = os.path.join(folder_path, f"sid_{station_id}_cdl.tif")

    sensor_files = {sensor: {} for sensor in included_sensors}

    for f in os.listdir(folder_path):
        if f.lower().startswith(prefix):
            datetime_str = f.split("_")[-2].split(".")[0]
            sensor_str = f.split("_")[-1].split(".")[0]

            if sensor_str in included_sensors:
                sensor_files[sensor_str][datetime_str] = f

    for sensor in included_sensors:
        sensor_files[sensor] = dict(sorted(sensor_files[sensor].items(), reverse=False))

    file_keys = list(sensor_files[included_sensors[0]].keys())

    if len(file_keys) >= num_slices:
        with rasterio.open(cdl_file_path) as cdl_src:
            cdl_array = cdl_src.read(1)
            mask = cdl_array == cdl_landcover_class_code
            binary_raster = mask.astype(np.uint8)

            valid_sequence = True
            file_keys_dates = [datetime.strptime(key, "%Y-%m-%d") for key in file_keys]
            for i in range(len(file_keys) - num_slices, len(file_keys) - 1):
                valid_sequence = (
                    valid_sequence
                    and (file_keys_dates[i] - file_keys_dates[i - 1]).days <= 10
                )

            if not valid_sequence:
                return None

            total_pixels = mask.size
            covered_pixels = np.sum(mask)
            covered_percentage = (covered_pixels / total_pixels) * 100
            if covered_percentage < acceptable_landcover_coverage:
                return None

            for i in range(len(file_keys) - num_slices, len(file_keys)):
                file_key = file_keys[i]
                combined_arrays = []

                for sensor in included_sensors:
                    file_name = sensor_files[sensor].get(file_key)
                    if file_name:
                        file_path = os.path.join(folder_path, file_name)
                        with rasterio.open(file_path) as src:
                            sensor_array = src.read()
                            sensor_array = np.moveaxis(
                                sensor_array, 0, -1
                            )  # Move channels to last dimension
                            combined_arrays.append(sensor_array)

                combined_array = np.concatenate(combined_arrays, axis=-1)
                binary_raster_expanded = np.expand_dims(binary_raster, axis=-1)
                combined_array_with_binary = np.concatenate(
                    (binary_raster_expanded, combined_array), axis=-1
                )

                arrays.append(combined_array_with_binary)

            stacked_array = np.stack(arrays)

            if stacked_array.size > 0:
                #     stacked_array = interpolate_nan_for_band(stacked_array, 13)
                #     stacked_array = interpolate_nan_for_band(stacked_array, 14)
                return stacked_array

    return None
