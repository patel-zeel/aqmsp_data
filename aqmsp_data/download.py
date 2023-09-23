from os.path import join, exists
from time import time
import kaggle

# import requests
# from requests_oauthlib import OAuth2Session

from aqmsp.debug_utils import verbose_print


def download_raw_data_from_kaggle(
    dataset_name: str,
    save_dir: str,
):
    """Download raw data from Kaggle.

    Args:
        dataset_name (str): Name of the dataset.
        save_dir (str): Path to the directory where the data will be downloaded.
    """
    kaggle.api.authenticate()
    init = time()
    kaggle.api.dataset_download_files(
        dataset=dataset_name,
        path=save_dir,
        unzip=True,
    )
    verbose_print(f"Download: downloaded data in {(time() - init)/60} minutes", decorate=True)


# def download_raw_data_from_zenodo(
#     access_token: str,
#     record_id: str,
#     save_dir: str,
# ):
#     """Download raw data from Zenodo.

#     Args:
#         access_token (str): Access token for the Zenodo API.
#         record_id (str): Record ID for the dataset.
#         save_dir (str): Path to the directory where the data will be downloaded.
#     """
#     zenodo_api_url = "https://zenodo.org/api/records/"
#     session = OAuth2Session(token={"access_token": access_token})
#     response = session.get(f"{zenodo_api_url}{record_id}")

#     init = time()
#     # get data download link
#     if response.status_code == 200:
#         data = response.json()
#         link = data["files"][0]["links"]["self"]
#     else:
#         raise ValueError(f"Error: Unable to retrieve record. Status code: {response.status_code}")

#     verbose_print(f"Download: got initial response")

#     # download data
#     response = session.get(link)
#     if response.status_code == 200:
#         with open(join(save_dir, "data.zip"), "wb") as f:
#             f.write(response.content)
#     else:
#         raise ValueError(f"Error: Unable to download data. Status code: {response.status_code}")

#     verbose_print(f"Download: downloaded data in {(time() - init)/60} minutes", decorate=True)
