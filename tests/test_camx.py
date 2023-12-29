from aqmsp_data import preprocessing
import xarray as xr

def test_preprocess_raw_camx():
    path = "tests/camxmet2d.delhi.20231018.96hours.nc"
    ds = preprocessing.preprocess_camx_met(path,verbose=True)
    print(ds)
    assert isinstance(ds, xr.Dataset)
    return ds

if __name__ == "__main__":
    test_preprocess_raw_camx()