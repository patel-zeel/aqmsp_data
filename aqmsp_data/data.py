import os
import string
from os.path import join, exists
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
from aqmsp.debug_utils import verbose_print
from aqmsp_data.preprocessing import preprocess_raw_cpcb

import botocore
import boto3
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from time import time

from typing import Sequence, Union
from itertools import product

import boto3
from multiprocessing import Pool

from psutil import cpu_count


class Data:
    def __init__(self):
        pass


class StationData(Data):
    def __init__(self):
        super().__init__()

    def get_stations(self, data_dir: str) -> xr.Dataset:
        """Get all station names and their latitude-longitude information from the data.

        Args:
            data_dir (str): Path to the directory containing the data.
        """
        path = join(data_dir, "stations.nc")
        with xr.open_dataset(path, cache=True) as ds:
            return ds

    def get_data(self, data_dir: str) -> xr.Dataset:
        """Get data from stations in xarray format.

        Args:
            data_dir (str): Path to the directory containing the data.
        """
        data_path = join(data_dir, "data.nc")
        with xr.open_dataset(data_path, cache=True) as ds:
            return ds


class CPCBData(StationData):
    def __init__(self):
        super().__init__()

    def create_station_ds(self, save_dir: str, temp_dir: str = None, cache=False, n_jobs: int = 1) -> Union[str, None]:
        """Create a dataset containing latitude-longitude information by reading historical data from OpenAQ. The latitude-longitude information is not available in the raw data files from CPCB.

        Args:
            save_dir (str): Directory to save the dataset.
            temp_dir (str, optional): Directory to read the temporary downloaded files from. Defaults to None.
            cache (bool, optional): Whether to cache the temporary downloaded files. Defaults to False. If `load_temp_dir` is not None, this argument is overriden to True.
            n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1.

        Returns:
            None: If `cache` is False and `load_temp_dir` is not provided, all temporary files are deleted.
            str: Path to the directory containing the downloaded files if `cache` is True and/or `load_temp_dir` is provided.
        """

        # if n_jobs is -1 then use all available cores
        if n_jobs == -1:
            n_jobs = cpu_count()

        # If temp_dir is not None, preserve the downloaded files
        if temp_dir is not None:
            cache = True

        def get_one_file(prefix):
            s3 = boto3.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
            object = s3.list_objects_v2(Bucket=source_bucket, Prefix=prefix)
            all_files = [file["Key"] for file in object.get("Contents", [])]
            file = np.random.choice(all_files)

            # Download the file
            file_name = file.split("/")[-1]
            download_path = os.path.join(load_temp_dir, file_name)
            s3.download_file(source_bucket, file, download_path)

        s3 = boto3.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))

        source_bucket = "openaq-data-archive"

        # Get all locations in India provided by CPCB
        prefix = "records/csv.gz/provider=caaqm/country=in/"
        object = s3.list_objects_v2(Bucket=source_bucket, Prefix=prefix, Delimiter="/")

        all_location_prefixes = [common_prefix["Prefix"] for common_prefix in object.get("CommonPrefixes", [])]
        verbose_print(f"Found {len(all_location_prefixes)} locations", decorate=True)

        if temp_dir is None:
            # For each location, get one file
            folder_name = "aqmsp_" + "".join(np.random.choice(list(string.ascii_lowercase), 10))
            load_temp_dir = os.path.join("/tmp", folder_name)
            os.makedirs(load_temp_dir, exist_ok=True)
            verbose_print(f"Using temporary directory {load_temp_dir}")

            init = time()
            verbose_print(f"Downloading temporary files", decorate=True)
            with tqdm_joblib(tqdm(total=len(all_location_prefixes))) as pbar:
                Parallel(n_jobs=n_jobs)(delayed(get_one_file)(prefix) for prefix in all_location_prefixes)
            verbose_print(f"Downloaded all files in {time() - init} seconds", decorate=True)
        else:
            load_temp_dir = temp_dir

        check_files = glob(os.path.join(load_temp_dir, "*.csv.gz"))
        assert len(check_files) == len(
            all_location_prefixes
        ), f"Downloaded {len(check_files)} files, expected {len(all_location_prefixes)}"

        # Find the latitude-longitude information
        def get_lat_lon(file):
            df = pd.read_csv(file)
            # format: 2020-09-29T03:00:00+05:30
            df["time"] = pd.to_time(df.time, format="%Y-%m-%dT%H:%M:%S%z")

            station = int(df.station.iloc[0])
            location = df.location.iloc[0]
            latitude = float(df.lat.iloc[0])
            longitude = float(df.lon.iloc[0])
            time = df.time.values[0].astype("time64[ns]")

            # remove location-id from the location name. Example: "Sector - 62, Noida - IMD-5616" -> "Sector - 62, Noida - IMD"
            station_name = location[::-1].split("-", 1)[1][::-1]

            return {
                "station": station_name,
                "latitude": latitude,
                "longitude": longitude,
                "station": station,
                "time": time,
            }

        init = time()
        lat_lon_dict = Parallel(n_jobs=n_jobs)(delayed(get_lat_lon)(file) for file in check_files)
        verbose_print(
            f"Found latitude-longitude information for {len(lat_lon_dict)} locations in {time() - init} seconds"
        )

        # create a dataframe
        df = pd.DataFrame(lat_lon_dict).set_index("station")

        # some stations are duplicate because their latitude-longitude got updated over time. Keep the latest one based on station.
        df.sort_values(by="station", inplace=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        verbose_print(df.head())
        ds = df.to_xarray()

        # assign latitude and longitude as coordinates
        ds = ds.set_coords(["latitude", "longitude"])

        # save the dataframe
        ds.to_netcdf(os.path.join(save_dir, "stations.nc"))
        verbose_print(f"Saved the dataset to {save_dir}")

        # delete all temporary files
        if cache is False and temp_dir is None:
            for file in check_files:
                os.remove(file)
            return None
        return load_temp_dir

    def preprocess_and_save(self, raw_data_dir: str, save_dir: str, station_ds_path: str, n_jobs: int = 1) -> None:
        """Preprocess the raw data and save it in a netcdf file.

        Args:
            raw_data_dir (str): Path to the directory containing the raw data files.
            save_dir (str): Path to the directory to save the preprocessed data.
            station_ds_path (str): Path to the dataset containing station information. Use `aqmsp_data.data.CPCBData().create_station_ds` method to get this dataset.
            n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1.
        """

        # if n_jobs is -1 then use all available cores
        if n_jobs == -1:
            n_jobs = cpu_count()

        all_files = glob(join(raw_data_dir, "site_*.xlsx"))

        # parallelize
        if n_jobs > 1:
            ds_list = Parallel(n_jobs=n_jobs)(delayed(preprocess_raw_cpcb)(file, station_ds_path) for file in all_files)
        else:
            ds_list = [preprocess_raw_cpcb(file, station_ds_path) for file in tqdm(all_files)]

        # filter out None values
        ds_list = [ds for ds in ds_list if ds is not None]
        verbose_print(f"Found {len(ds_list)} valid files")

        ds = xr.merge(ds_list)

        # sort by time
        ds = ds.sortby("time")

        # assert consistent time
        assert (ds.time.diff("time") == ds.time.diff("time")[0]).all(), "Time is not consistent across stations"

        # save
        ds.to_netcdf(join(save_dir, "data.nc"))
        verbose_print(f"Saved the dataset as `data.nc` to {save_dir}")


# def load_camx(
#     years: Union[Sequence, int],
#     months: Union[Sequence, int] = None,
#     days: Union[Sequence, int] = None,
#     variables: Union[Sequence, str] = None,
#     lag: int = 0,
# ):
#     root = get_repo_root(__file__)
#     path = join(root, C.CAMX_PATH, "20*.nc")
#     ds = load_data(path, years, months, days, variables)
#     return ds.sel(lag=lag).drop_vars("lag")


# def load_data(path, years, months, days, variables):
## The following code is commented out because it is slow
# files = glob(path)
# data_files = [xr.open_dataset(file) for file in files]
# ds = xr.concat(data_files, dim="time")
# ds = xr.open_mfdataset(path)

# ds = ds.sel(time=ds.time.dt.year.isin(years))
# if months is not None:
#     ds = ds.sel(time=ds.time.dt.month.isin(months))
# if days is not None:
#     ds = ds.sel(time=ds.time.dt.day.isin(days))
# if variables is not None:
#     if isinstance(variables, str):
#         variables = [variables]
#     ds = ds[variables]
# return ds


# def load_shapefile(name):
#     root = get_repo_root(__file__)
#     path = join(root, C.SHAPEFILES_PATH, name, "*.shp")
#     file = glob(path)[0]
#     return gpd.read_file(file)


# if __name__ == "__main__":
#     set_verbose(True)

#     if exists("/tmp/2023-09-07/data.nc"):
#         verbose_print("Found existing data. Deleting it", decorate=True)
#         os.remove("/tmp/2023-09-07/data.nc")

#     # declare the data object
#     data = CPCBData()

#     # Download it from Zenodo

#     print("Preprocessing data")
#     init = time()
#     data.preprocess_and_save(
#         raw_data_dir="/tmp/2023-09-07",
#         save_dir="/tmp/2023-09-07",
#         n_jobs=42,
#         station_ds_path="/home/patel_zeel/aqmsp/lab/stations.nc",
#     )
#     print(f"Preprocessing took {time()-init:.2f} seconds")
#     ds = xr.open_dataset("/tmp/2023-09-07/data.nc")
#     print(ds)
#     ds = load_cpcb(2022, 1, 1, ["PM2.5", "PM10"])
#     print(ds)
#     ds = load_camx(2022, 1, 1, ["P25", "P10"])
#     print(ds)


# def load_station_ds():
#     path = join(root, C.STATIONS_PATH, "stations.nc")
#     ds = xr.open_dataset(path, cache=True)
#     return ds

# def load_cpcb(
#     years: Union[Sequence, int],
#     months: Union[Sequence, int] = None,
#     days: Union[Sequence, int] = None,
#     variables: Union[Sequence, str] = None,
#     stations: Union[Sequence, str] = None,
# ):
#     # convert to list if not already
#     (years, months, days, variables, stations) = map(
#         lambda x: [x] if isinstance(x, (int, str)) else x,
#         (years, months, days, variables, stations),
#     )

#     # check inputs and set defaults
#     assert years is not None, "years must be specified"

#     root = get_repo_root(__file__)
#     path = join(root, C.CPCB_PATH)

#     # read ds
#     read_paths = [join(path, f"{year}.nc") for year in years]
#     for read_path, year in zip(read_paths, years):
#         assert exists(
#             read_path
#         ), f"File {read_path} does not exist.\nPlease remove the year '{year}' from the list of years"

#     if len(ds.time) == 0:
#         raise ValueError("No data found for the specified time range")

#     if months is not None:
#         ds = ds.sel(time=ds.time.dt.month.isin(months))
#     if days is not None:
#         ds = ds.sel(time=ds.time.dt.day.isin(days))
#     if variables is not None:
#         ds = ds[variables]
#     if stations is not None:
#         ds = ds.sel(station=stations)

#     # sort by time
#     return ds.load()
