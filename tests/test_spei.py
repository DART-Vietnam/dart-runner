import os
from unittest import mock

import xarray as xr
import numpy as np
import numpy.testing as npt

from main import add_spi_spei_bc


def finite_max(data: xr.DataArray) -> float:
    finite_max = data.where(np.isfinite(data)).max()
    return float(finite_max.values)


def is_inf(x) -> bool:
    return bool(np.isinf(x.values))


@mock.patch.dict(os.environ, {"DART_PIPELINE_DATA_HOME": "data"})
def test_spi_spei():
    ds = xr.open_dataset(
        "data/sources/VNM/ecmwf/VNM-2025-06-24-ecmwf.forecast.corrected.nc",
        decode_timedelta=True,
    )
    sds = add_spi_spei_bc(ds)

    # check maximums
    npt.assert_approx_equal(finite_max(sds.spei_bc), 8.2095365524292)
    npt.assert_approx_equal(finite_max(sds.spi_bc), 8.014016151428223)

    # check minimums
    npt.assert_almost_equal(float(sds.spei_bc.min().values), -0.6938957)
    npt.assert_almost_equal(float(sds.spi_bc.min().values), -0.6930721402168274)

    # Max SPEI/SPI is infinity for points that fall outside the reference REMOCLIC
    # dataset which covers only Vietnam. This includes points in neighbouring
    # countries and points in the ocean
    assert is_inf(sds.spei_bc.max())
    assert is_inf(sds.spi_bc.max())
