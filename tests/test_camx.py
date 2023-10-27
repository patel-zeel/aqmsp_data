from aqmsp_data.preprocessing import preprocess_camx_met
import xarray as xr
from aqmsp.debug_utils import set_verbose, verbose_print

def test_preprocess_raw_cpcb():
    path = "/home/utkarsh.mittal/camxmet2d.delhi.2023/camxmet2d.delhi.20230717.96hours.nc"
    ds = preprocess_camx_met(path,verbose=True)
    print(ds)
    assert isinstance(ds, xr.Dataset)
    return ds