from os.path import join
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
from aqmsp.path_utils import get_repo_root
import aqmsp_data.constants as C
from aqmsp_data.data import load_station_ds


def get_closest_data_from_camx(
    camx: xr.Dataset, k: int, agg: str = "mean", stations: list = None
):
    """Get aggregated data from k closest grid cells. The value of k must be a full square number.

    Args:
        camx: CAMx dataset
        caaqm: CAAQM dataset
        k: Number of closest grid cells to consider
        agg: Aggregation function to apply to the k closest grid cells
    """

    # TODO: add support for other aggregation functions

    assert np.sqrt(k) ** 2 == k, f"k must be a full square number (got {k})"
    closest_n = int(np.sqrt(k))  # number of closest grid cells to consider

    station_ds = load_station_ds()
    if stations is None:
        stations = station_ds.station.values
    else:
        assert (
            len(set(stations) - set(station_ds.station.values)) == 0
        ), "Some stations are not present in the CPCB dataset"

    camx_list = []
    for station in tqdm(stations):
        lat = station_ds.sel(station=station).latitude.values.item()
        lon = station_ds.sel(station=station).longitude.values.item()
        lat_diff = np.abs(camx.latitude.values - lat)
        # print(lat_diff, type(lat_diff))
        lon_diff = np.abs(camx.longitude.values - lon)
        lat_idx = np.atleast_1d(
            np.argpartition(lat_diff, closest_n)[:closest_n]
        )  # argpartition is faster than argsort
        lon_idx = np.atleast_1d(
            np.argpartition(lon_diff, closest_n)[:closest_n]
        )  # argpartition is faster than argsort
        # print(lat_idx, lon_idx, closest_n)
        sorted_lat_idx = lat_idx[np.argsort(lat_diff[lat_idx])]
        sorted_lon_idx = lon_idx[np.argsort(lon_diff[lon_idx])]
        closest_lats = camx.latitude.values[sorted_lat_idx]
        closest_lons = camx.longitude.values[sorted_lon_idx]
        closest_ds = camx.sel(latitude=closest_lats, longitude=closest_lons)

        if agg == "mean":
            closest_ds = closest_ds.mean(dim=["latitude", "longitude"])
        else:
            raise NotImplementedError(f"Aggregation function {agg} not implemented")
        # closest_ds = closest_ds.assign_coords(station=station)
        closest_df = closest_ds.to_dataframe()
        closest_df["station"] = station
        closest_ds = closest_df.reset_index().set_index(["station", "time"]).to_xarray()
        closest_ds = closest_ds.assign_coords(latitude=(["station"], [lat]))
        closest_ds = closest_ds.assign_coords(longitude=(["station"], [lon]))
        camx_list.append(closest_ds)
    return xr.concat(camx_list, dim="station")
