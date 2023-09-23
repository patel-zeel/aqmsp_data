import os
import string
import numpy as np
from dotenv import load_dotenv
from aqmsp.debug_utils import set_verbose, verbose_print
from aqmsp_data.data import CPCBData
from aqmsp_data.download import download_raw_data_from_kaggle
from psutil import cpu_count


def test_cpcb_end_to_end_pipeline():
    set_verbose(True)
    n_jobs = cpu_count() // 2

    # load the environment variables
    load_dotenv()

    # define paths
    random_folder = "aqmsp_" + "".join(np.random.choice(list(string.ascii_lowercase), 10))
    download_dir = os.path.join("/tmp", random_folder)
    os.makedirs(download_dir, exist_ok=True)
    verbose_print(f"Download directory: {download_dir}", decorate=True)

    # download data
    os.makedirs(download_dir, exist_ok=True)
    download_raw_data_from_kaggle(dataset_name="zeelpatel19310068/cpcb-2023-dataset", save_dir=download_dir)

    # create data object
    cpcb_data = CPCBData()

    verbose_print("Downloading station metadata", decorate=True)
    cpcb_data.create_station_ds(
        save_dir=download_dir,
        cache=False,
        temp_dir=None,
        n_jobs=n_jobs,
    )

    # preprocess raw data
    verbose_print("Preprocessing raw data", decorate=True)
    raw_data_dir = os.path.join(download_dir)
    station_ds_path = os.path.join(download_dir, "stations.nc")
    cpcb_data.preprocess_and_save(
        raw_data_dir=raw_data_dir, save_dir=download_dir, station_ds_path=station_ds_path, n_jobs=n_jobs
    )

    # load preprocessed data
    verbose_print("Loading preprocessed data", decorate=True)
    ds = cpcb_data.get_data(data_dir=download_dir)
    verbose_print(ds, decorate=True)
    verbose_print(ds.info(), decorate=True)
