import os
import xarray as xr
from aqmsp.path_utils import get_repo_root

MONTHS = list(map(lambda x: str(x).zfill(2), range(1, 13)))

# set the attributes
ALL_ATTRS = {
    "PM2.5": {"unit": "ug/m3", "long_name": "PM2.5", "range": (0, 1100)},
    "PM10": {"unit": "ug/m3", "long_name": "PM10", "range": (0, 1100)},
    "AT": {
        "unit": "degree C",
        "long_name": "Atmospheric Temperature",
        "range": (0, 60),
    },
    "BP": {
        "unit": "mmHg",
        "long_name": "Barometric Pressure",
        "range": (700, 1200),
    },
    "RH": {"unit": "%", "long_name": "Relative Humidity", "range": (0, 100)},
    "RF": {"unit": "mm", "long_name": "Rainfall", "range": (0, 30)},
    # "TOT-RF": {"unit": "mm", "long_name": "Total Rainfall", "range": (0, 50)}, # Ignore for now
    "SR": {"unit": "W/mt2", "long_name": "Solar Radiation", "range": (0, 2000)},
    # "WS": {"unit": "m/s", "long_name": "Wind Speed", "range": (0, 10)},
    # "WD": {"unit": "degree", "long_name": "Wind Direction"},
}

EXCEPTION_VARS = ["TOT-RF"]

STATIONS = {}
