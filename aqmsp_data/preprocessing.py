import os
import calendar
import numpy as np
import pandas as pd
import xarray as xr
from aqmsp.path_utils import get_repo_root
import aqmsp_data.constants as C
from aqmsp.debug_utils import set_verbose, verbose_print
from time import time
from typing import Union


def preprocess_raw_cpcb(path: str, stations_ds_path: str) -> Union[xr.Dataset, None]:
    """Preprocess raw CPCB data and return an xr dataset.

    Args:
        path (str): Path to the raw CPCB data file.
        stations_ds_path (xr.Dataset): path to the dataset containing station information. Run `aqmsp_data.data.CPCBData().create_station_ds()` to get this dataset.

    Raises:
        an: _description_

    Returns:
        Union[xr.Dataset, None]: _description_
    """

    file_name = os.path.basename(path)
    assert file_name.endswith(".xlsx"), f"File '{path}' must be an excel file downloaded from the CPCB website"
    assert file_name.startswith("site_"), f"File '{path}' must start with 'site_'"

    verbose_print(f"Reading station dataset from {stations_ds_path}")
    with xr.open_dataset(stations_ds_path) as station_ds:
        station_ds = station_ds.load()

    df = pd.read_excel(path, header=None)
    verbose_print(f"Loaded {path} with shape {df.shape}")

    # get the station name
    station = df.iloc[6, 1]
    if station not in station_ds.station.values:
        verbose_print(f"Station '{station}' not found in station dataset. Skipping.")
        return None

    # extract the data
    df = df.iloc[16:, :]  # remove the first 16 rows
    df.columns = df.iloc[0, :].values  # set the column names
    df = df.iloc[1:, :]  # remove the column names row
    # verbose_print(f"columns: {df.columns}")
    for col in df.columns:
        allowed_cols = list(C.ALL_ATTRS.keys()) + C.EXCEPTION_VARS + ["From Date", "To Date"]
        assert col in allowed_cols, f"Variable {col} not found in ALL_ATTRS"

    # convert to datetime
    df["From Date"] = pd.to_datetime(df["From Date"], format="%d-%m-%Y %H:%M")
    df["To Date"] = pd.to_datetime(df["To Date"], format="%d-%m-%Y %H:%M")
    df_len = len(df)

    # remove inconsistent dates
    df = df[df["From Date"] == df["To Date"] - pd.Timedelta(minutes=60)]
    verbose_print(f"Removed {df_len - len(df)} inconsistent date rows")

    # add time dimension
    df["time"] = df["From Date"] + pd.Timedelta(minutes=30)
    df = df.drop(columns=["From Date", "To Date"])

    # assert that time is continuous
    unique_diffs = (df.time - df.time.shift(1)).unique()
    assert len(unique_diffs) == 2, "Time is not continuous"
    assert unique_diffs[0] is pd.NaT, "This assert should never trigger, raise an issue"
    assert unique_diffs[1] == pd.Timedelta(hours=1), "Time is not hourly"
    verbose_print("Start date", df.time.min())
    verbose_print("End date", df.time.max())

    # add latitude, longitude and station
    station_ds = station_ds.sel(station=station)
    df["station"] = station_ds.station.values.item()

    # set the index
    df = df.set_index(["time", "station"])

    # convert data to numeric
    df = df.apply(pd.to_numeric)

    # convert to xr dataset
    ds = df.to_xarray()

    # assign latitudes and longitudes
    ds = ds.assign_coords(
        latitude=("station", [station_ds.latitude.values.item()]),
        longitude=("station", [station_ds.longitude.values.item()]),
    )

    # assert that the data is valid
    for var in ds.data_vars:
        if var in C.EXCEPTION_VARS:
            verbose_print(f"Skipping variable {var}")
            continue
        lower_limit = C.ALL_ATTRS[var]["range"][0]
        upper_limit = C.ALL_ATTRS[var]["range"][1]

        ds[var].attrs = C.ALL_ATTRS[var]
        min_val = ds[var].min().values.squeeze().item()
        max_val = ds[var].max().values.squeeze().item()

        if not np.isnan(min_val):
            assert min_val >= lower_limit, f"Variable {var} has value {min_val} less than {lower_limit}"
        if not np.isnan(max_val):
            assert max_val <= upper_limit, f"Variable {var} has value {max_val} greater than {upper_limit}"

    return ds


def test_preprocess_raw_cpcb():
    set_verbose(True)
    path = "/home/patel_zeel/aqmsp/aqmsp_data/data/cpcb/another_raw/site_539320230905184650.xlsx"
    ds = preprocess_raw_cpcb(path)
    verbose_print(ds)
    assert isinstance(ds, xr.Dataset)
    return ds


def split_and_save_cpcb(ds: xr.Dataset):
    """
    Provide a single station file preprocessed by `preprocess_raw_cpcb` function.
    """
    if ds is None:
        verbose_print("ds is None. Skipping.")
        return None
    assert isinstance(ds, xr.Dataset), "ds must be an xr dataset"
    root = get_repo_root(__file__)
    station = ds.station.values.item()
    station = station.replace(" ", "_")
    path = os.path.join(root, C.CPCB_PATH)

    years = np.unique(ds.time.dt.year)
    init_time = time()
    for year in years:
        year_ds = ds.sel(time=ds.time.dt.year == year)
        year_path = os.path.join(path, str(year), station)
        months = np.unique(year_ds.time.dt.month)
        for month in months:
            month_ds = year_ds.sel(time=year_ds.time.dt.month == month)

            n_days = calendar.monthrange(year, month)[1]
            all_days = np.arange(1, n_days + 1)

            str_month = str(month).zfill(2)
            month_path = os.path.join(year_path, str_month)
            os.makedirs(month_path, exist_ok=True)

            present_days = np.unique(month_ds.time.dt.day.values)
            diff = np.setdiff1d(all_days, present_days)
            if len(diff) > 0:
                verbose_print(f"Missing {len(diff)} days in month {month} of year {year}. Skipping.")
                continue
            else:
                pass

            for var in month_ds.data_vars:
                if var not in C.EXCEPTION_VARS:
                    save_path = os.path.join(month_path, f"{var}.nc")
                    month_ds[[var]].to_netcdf(save_path)
                else:
                    # verbose_print(f"Skipping variable {var}")
                    pass

    verbose_print(f"Saved CPCB data in {time() - init_time} seconds")


def test_split_and_save_cpcb():
    ds = test_preprocess_raw_cpcb()
    split_and_save_cpcb(ds)


def combine_and_save_cpcb():
    root = get_repo_root(__file__)
    path = os.path.join(root, C.CPCB_PATH)

    # detect directories with name as year
    years = os.listdir(path)
    years = [year for year in years if year.isdigit() and len(year) == 4 and int(year) >= 2000 and int(year) <= 2100]
    years = sorted(years)
    verbose_print(f"Found {len(years)} years: {years}")

    all_files_to_read = []
    for year in years:
        year_path = os.path.join(path, year)
        stations = os.listdir(year_path)
        assert_correct_station(stations)
        for station in stations:
            station_path = os.path.join(year_path, station)
            months = os.listdir(station_path)
            assert_correct_month(months)
            for month in months:
                months_path = os.path.join(station_path, month)
                variables = os.listdir(months_path)
                assert_correct_variable(
                    map(
                        lambda x: x.replace(".nc", "", 1) if x.endswith(".nc") else x,
                        variables,
                    )
                )
                for variable in variables:
                    variable_path = os.path.join(months_path, variable)
                    all_files_to_read.append(f"{variable_path}")

    # concatenate all data
    init = time()
    verbose_print(f"Reading {len(all_files_to_read)} files")
    ds = xr.open_mfdataset(all_files_to_read).load()
    verbose_print(f"Loaded {len(all_files_to_read)} files in {time() - init} seconds")

    # sort by time
    ds = ds.sortby("time")

    # save the dataset year-wise
    for year in years:
        year_ds = ds.sel(time=ds.time.dt.year == int(year))
        save_path = os.path.join(root, C.CPCB_PATH, f"{year}.nc")
        verbose_print(f"Saving {year} data in {save_path}")
        # check if dataset is not empty
        if year_ds.dims["time"] == 0:
            verbose_print(f"Skipping {year} because it is empty")
            continue
        year_ds.to_netcdf(save_path)
        verbose_print(f"Saved {year} data in {save_path}")


def assert_correct_station(stations):
    station_ds = load_station_ds()
    all_stations = [name.replace(" ", "_") for name in station_ds.station.values]
    stations = [station.replace(" ", "_") for station in stations]
    for station in stations:
        assert station in all_stations, f"Station {station} not found in station dataset"
    return True


def assert_correct_month(months):
    for month in months:
        assert month in C.MONTHS, f"Month {month} not found in C.MONTHS"


def assert_correct_variable(variables):
    for var in variables:
        assert var in C.ALL_ATTRS.keys(), f"Variable {var} not found in C.ALL_ATTRS"


def preprocess_raw_camx_output():
    pass

def preprocess_camx_met(camx_met_file: str, verbose: bool = False) -> xr.Dataset:
    assert camx_met_file.endswith(".nc"), f"File '{camx_met_file}' must be a netcdf file"
    assert camx_met_file.startswith("camxmet2d.delhi.2023"), f"File '{camx_met_file}' must start with 'camxmet2d.delhi.2023'"
    camx_met = xr.open_dataset(camx_met_file)
    xorig = camx_met.XORIG
    yorig = camx_met.YORIG
    longitude = xorig + (camx_met.COL.values - 1) * 0.01
    latitude = yorig + (camx_met.ROW.values - 1) * 0.01
    camx_met = camx_met.rename({'ROW': 'latitude', 'COL': 'longitude','TSEP':'time'})
    camx_met['latitude'], camx_met['longitude'] = latitude, longitude
    temp_datasets = []
    for lag in range(4):
        data_temp = camx_met.sel(TSTEP=slice(lag * 24, 24 * (lag + 1)))
        date_str = camx_met_file.split('.')[-3]
        timesteps = data_temp.dims['time']
        start_time = pd.Timestamp(date_str + ' 00:00:00')
        datetime_index = pd.date_range(start=start_time, periods=timesteps, freq='H')
        data_temp['time'] = datetime_index
        tstep_array = np.array(data_temp['time'].values, dtype='datetime64[ns]')
        tstep_array += np.timedelta64(5, 'h') + np.timedelta64(30, 'm')
        data_temp['time'] = ('time', tstep_array)
        temp_datasets.append(data_temp)
        if verbose:
            print(f"Preprocessed CAMx meteorological data for lag {lag}:")
            print(data_temp)
    reshaped_camx = xr.concat(temp_datasets, pd.Index(range(4), name='lag')).sel(LAY=0, VAR=0)
    return reshaped_camx

if __name__ == "__main__":
    set_verbose(True)
    test_preprocess_raw_cpcb()
    test_split_and_save_cpcb()
    combine_and_save_cpcb()
