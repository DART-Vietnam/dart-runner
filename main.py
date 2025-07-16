# dart-runner
#
# Orchestrates DART-Pipeline runs with forecast downloads and model predictions

import os
import re
import sys
import types
import shutil
import docker
import logging
import argparse
import warnings
import datetime
from pathlib import Path
from typing import overload, Callable

import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import norm
import xclim.indicators.atmos
from geoglue.util import sha256, set_lonlat_attrs
from geoglue.cds import ReanalysisSingleLevels
from geoglue.region import gadm
from dart_pipeline.paths import get_path
import dart_pipeline.metrics as metrics
import dart_pipeline.metrics.era5.list_metrics as list_metrics
from dart_pipeline.metrics.era5.util import gamma_func, norminv
from dart_pipeline.metrics.ecmwf import get_forecast_open_data, process_forecast
from dart_bias_correct.precipitation import bias_correct_precipitation
from dart_bias_correct.forecast import bias_correct_forecast_from_paths


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO
)

MIN_DISK_SPACE_GB = 40
STEPS = {
    "dl": "Fetching forecast",
    "bc": "Performing forecast bias correction",
    "spei": "Computing SPI and SPEI",
    "zs": "Processing forecast",
    "m": "Running model",
}
BASE_PATH = Path.home() / ".local/share/dart-pipeline"
ISO3 = "VNM"
ADMIN = 2
DEFAULT_FORECAST_DATE = "2025-05-28"
OBSERVATION_REFERENCE = "T2m_r_tp_Vietnam_ERA5.nc"
PREDICTION_CSV_SUFFIX = "dengue-predictions.csv"
REGION = gadm(ISO3, ADMIN)


class MismatchError(Exception):
    pass


class AxisMismatchError(MismatchError):
    pass


def parse_checksums(s: str) -> dict[str, str]:
    out = {}
    for line in s.split("\n"):
        if line.strip() == "":
            continue
        checksum, *parts = line.split()
        out[" ".join(parts)] = checksum
    return out


# output of sha256sum (Linux) or shasum -a 256 (macOS)
CHECKSUMS = parse_checksums("""
282ad3246e79c28c68e469e74388fe68b6e3496673a2e807178151f462adb729  T2m_r_tp_Vietnam_ERA5.nc
fd40c75c3dcf8bb25e0c32e5689cef941927964a132feab3686ae0d0d9454a7a  eefh_testv2_test_githubv1_3.nc
c1ec01cee2efda44772b14c0e3b6f455de883601d0c4481257e7654cbaf7ef96  vngp_regrid_era_full.nc
""")

MODEL_IMAGE = "dart-model-container:0.4.0"
ALIASES = {
    "observation": "T2m_r_tp_Vietnam_ERA5.nc",
    "historical-forecast": "eefh_testv2_test_githubv1_3.nc",
    # REMOCLIC. (2016). VnGP - Vietnam Gridded Precipitation dataset
    # (0.10°×0.10°) [Data set]. Data Integration and Analysis System (DIAS).
    # https://doi.org/10.20783/DIAS.270
    "reference-precipitation": "vngp_regrid_era_full.nc",
}
MODEL_DATA_HOME = Path.home() / ".local/share/dart-pipeline/sources/VNM/model"
MODEL_EXPECTED_OUTPUT = MODEL_DATA_HOME / "forecast_snapshot_ref.csv"

if not MODEL_DATA_HOME.exists():
    warnings.warn("Model data home does not exist, cannot run actual model")

DOCKER_VOLUMES = {
    "local_renv_cache": {
        "bind": "/modelling/renv/.cache",
        "mode": "rw",
    },
    os.path.expanduser("~/.local/share/dart-pipeline/sources/VNM/model"): {
        "bind": "/modelling/data",
        "mode": "rw",
    },
}

M1 = types.SimpleNamespace(
    WORKDIR="/modelling",
    RENV_CACHE_CONTAINER="/modelling/renv/.cache",
    INPUT_INC_FILE="data/fake_incidence_full.csv",
    R_OUTFILE_STAMP="snapshot_ref",
    FORECAST_OUTFILE="forecast_snapshot_ref",
    FCST_HORIZON=2,
    PARALLEL_WORKERS=-1,
)

model_command = f"""
mkdir -p {M1.RENV_CACHE_CONTAINER} && \
Rscript -e 'renv::paths$cache();renv::restore(prompt=FALSE)' && \
Rscript 01_data_prepper.R \
    {M1.INPUT_INC_FILE} \
    data/weather/VNM-2-2001-2019-era5.nc \
    {M1.R_OUTFILE_STAMP} \
    data/gadm/gid2_lookup_df.rds && \
Rscript 99_mlr3_modelling.R \
    data/weekly_inc_weather_{M1.R_OUTFILE_STAMP}.rds \
    data/tuned_lrners_tunedRF_fh-1-12_wa.rds \
    data/updated_agaci_obj_lists_tunedRF_fh-1-12_wa.qs \
    {M1.FORECAST_OUTFILE} \
    --horizon {M1.FCST_HORIZON} \
    --future_plan multicore \
    --future_workers {M1.PARALLEL_WORKERS}
"""


def run_actual_1():
    client = docker.from_env()
    container = client.containers.run(
        image=MODEL_IMAGE,
        command=["/bin/bash", "-c", model_command],
        working_dir="/modelling",
        environment={"RENV_PATHS_CACHE": "/modelling/renv/.cache"},
        volumes=DOCKER_VOLUMES,
        detach=True,
        remove=True,
    )

    for line in container.logs(stream=True):
        sys.stdout.buffer.write(line)
        sys.stdout.flush()


def generate_dummy_data(districts, start_date=None, weeks=4, seed=42):
    """
    Generates dummy forecast data with confidence intervals for given districts.

    Parameters:
        districts (list): List of GID_2 district codes.
        start_date (str or datetime, optional): Start date (default: today).
        weeks (int): Number of weeks to generate (default: 4).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Dummy data matching the specified schema.
    """
    np.random.seed(seed)
    confidence_levels = [0.5, 0.75, 0.9, 0.95, 0.99]
    z_scores = {ci: norm.ppf(1 - (1 - ci) / 2) for ci in confidence_levels}
    start_date = (
        datetime.datetime.today() if start_date is None else pd.to_datetime(start_date)
    )
    dates = [start_date + datetime.timedelta(weeks=i) for i in range(weeks)]

    rows = []
    for district in districts:
        for date in dates:
            base_pred = np.random.uniform(1, 30)  # Central prediction
            std_dev = base_pred * np.random.uniform(0.05, 0.15)  # Simulated uncertainty
            for ci in confidence_levels:
                z = z_scores[ci]
                margin = z * std_dev
                pred_lower = max(0, base_pred - margin)  # type: ignore
                pred_upper = base_pred + margin
                rows.append(
                    {
                        ".pred": base_pred,
                        "intervals": ci,
                        ".pred_lower": pred_lower,
                        ".pred_upper": pred_upper,
                        "date": date.date(),
                        "district": district,
                    }
                )
    return pd.DataFrame(rows)


def get_file(alias: str, root: Path | None = None) -> Path:
    root = root or Path.cwd()
    if alias not in ALIASES:
        raise ValueError(f"Could not find {alias=} in ALIASES")
    expected_checksum, file = CHECKSUMS[ALIASES[alias]], ALIASES[alias]
    path = root / file
    if not path.exists():
        raise FileNotFoundError(f"Could not find {file=} under {root=}")
    actual_checksum = sha256(path)
    if actual_checksum == expected_checksum:
        return path
    raise ValueError(f"""Checksum verification failed for {path}:
    Expected: {expected_checksum}
      Actual: {actual_checksum}""")


def msg(bold: str, *args):
    print(f"\033[1m{bold}\033[0m", *args)


def fail(bold: str, *args):
    print(f"\033[31;1m{bold}\033[0m", *args)


def get_free_gigabytes():
    _, _, free = shutil.disk_usage("/")
    return free / (1024**3)  # GB


def parse_args():
    today = datetime.datetime.today().date().isoformat()
    parser = argparse.ArgumentParser(
        description="Orchestrates DART-Pipeline runs with forecast downloads and model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Cache steps (default=dl)
  * dl: Forecast downloaded files, e.g. VNM-2025-05-28-ecmwf.forecast.nc
  * bc: Bias corrected files, e.g. VNM-2025-05-28-ecmwf.forecast.corrected.nc
  * spei: Corrected files with SPI and SPEI added, e.g. VNM-2025-05-28-ecmwf.forecast.corrected_spei.nc
  * zs: Zonal statistics files, e.g. VNM-2-2025-05-28-ecmwf.forecast.nc

   Use --cache=s1,s2 to use cached files in s1 and s2, for example, set--cache=dl,bc
   to use cached files for the download and zonal statistics steps.
   Set --cache=all to enable maximum caching
   """,
    )
    parser.add_argument("--date", help="Forecast date to run for", default=today)
    parser.add_argument(
        "--model",
        help="Name of the model to use",
        choices=["dummy", "actual-1"],
        default="dummy",
    )
    parser.add_argument(
        "--cache",
        help="Use cached files for the following (comma separated) steps if available",
        default="all",
    )

    return parser.parse_args()


@overload
def run(
    cache: list[str],
    step: str,
    expected_paths: Path,
    func: Callable[..., Path],
    *args,
    **kwargs,
) -> Path: ...


@overload
def run(
    cache: list[str],
    step: str,
    expected_paths: list[Path],
    func: Callable[..., list[Path]],
    *args,
    **kwargs,
) -> list[Path]: ...


def run(
    cache: list[str],
    step: str,
    expected_paths: list[Path] | Path,
    func: Callable[..., list[Path] | Path],
    *args,
    **kwargs,
) -> Path | list[Path]:
    paths = [expected_paths] if isinstance(expected_paths, Path) else expected_paths
    if step in cache and all(p.exists() for p in paths):
        msg(f"... {STEPS[step]}:", *paths)
        return expected_paths
    msg(f"==> {STEPS[step]}")
    return func(*args, **kwargs)


def get_gamma_window(gamma_params: xr.Dataset) -> int:
    re_matches = re.match(r".*window=(\d+)", gamma_params.attrs["DART_history"])
    if re_matches is None:
        raise ValueError("No window option found in gamma parameters file")
    return int(re_matches.groups()[0])


def add_spi_spei_bc(cf: xr.Dataset) -> xr.Dataset:
    """Module to add SPI and SPEI to forecast"""
    cur_forecast = cf.median(dim="number")
    sources = get_path("sources", ISO3, "ecmwf")
    forecast_date: datetime.date = pd.to_datetime(cf.time.min().values).date()
    prev_week_forecast_date: datetime.date = forecast_date - datetime.timedelta(days=7)
    prev_forecast = xr.open_dataset(
        sources / f"{ISO3}-{prev_week_forecast_date}-ecmwf.forecast.corrected.nc",
        decode_timedelta=True,
    )
    # select only the first week of previous week's forecast
    prev_forecast = prev_forecast.isel(time=slice(0, 1)).median(dim="number")
    if "pevt" not in cur_forecast:
        raise KeyError(
            "pevt not found in current forecast, required for spei_bc calculation"
        )
    if "pevt" not in prev_forecast:
        raise KeyError(
            "pevt not found in previous forecast, required for spei_bc calculation"
        )

    # limit forecasts to pevt and tp_bc which is all that is needed for SPI and SPEI calc
    cur_forecast = cur_forecast[["tp_bc", "pevt"]]
    prev_forecast = prev_forecast[["tp_bc", "pevt"]]

    # get window
    spi_gamma_params = metrics.get_gamma_params(ISO3, "spi_corrected")
    spei_gamma_params = metrics.get_gamma_params(ISO3, "spei_corrected")
    spi_w = get_gamma_window(spi_gamma_params)
    spei_w = get_gamma_window(spei_gamma_params)
    if spi_w != spei_w:
        raise MismatchError(f"Differing windows {spi_w=} and {spei_w=}")
    window_weeks = spi_w
    logging.info("To fetch ERA5 data, using gamma parameters window=%d", window_weeks)

    # Concatenating observational data
    # We concat 4 weeks of observation data for a window of 6 weeks, with previous
    # week's bias-corrected forecast standing in for observations as ERA5 data
    # is released with a log of 5 days. Schematic diagram shown below depicting

    # ERA5 SPI/SPEI calculation (for a 2-week forecast starting 2025-03-03)
    # ----------------------------------------------------------------------------------------------
    # ▼ ERA5 (Past 4 Weeks)
    # [#######]                                                   Week -4 (2025-01-27 to 2025-02-02)
    #         [#######]                                           Week -3 (2025-02-03 to 2025-02-09)
    #                 [#######]                                   Week -2 (2025-02-10 to 2025-02-16)
    #                         [#######]                           Week -1 (2025-02-17 to 2025-02-23)
    # ▶ previous week's forecast*       [-------]                 Week  0 (2025-02-24 to 2025-03-02)
    # ▶ 2-week forecast*                        [====>>>]         Week  1 (2025-03-03 to 2025-03-09)
    #   * bias-corrected                                [====>>>] Week  2 (2025-03-10 to 2025-03-16)

    era5_start = prev_week_forecast_date - datetime.timedelta(
        weeks=window_weeks - 2
    )  # 6-2 = 4 weeks
    era5_path = get_path("sources", ISO3, "era5")
    pool = ReanalysisSingleLevels(
        REGION, list_metrics.VARIABLES, era5_path
    ).get_dataset_pool()
    era5_end = prev_week_forecast_date - datetime.timedelta(days=1)
    if era5_end.month == 12:
        if era5_end.day == 31:
            _era5 = pool[era5_end.year].sel(valid_time=slice(era5_start, era5_end))
        else:
            raise ValueError(
                "Currently SPI and SPEI calculations are only supported for forecasts after January 8"
            )
    else:
        _era5 = pool.get_current_year(era5_start, era5_end)
    era5 = xr.merge([_era5.instant, _era5.accum]).rename({"valid_time": "time"})
    era5 = era5[["t2m", "tp"]]

    # era5: get tp_bc, using dart-bias-correct
    tp_ref = xr.open_dataset(get_file("reference-precipitation"))
    era_tp = xr.open_dataset(get_file("observation"))
    era5["tp_bc"] = bias_correct_precipitation(tp_ref, era_tp, era5[["tp"]]).tp
    # era5: calculate pevt using t2m, mx2t24, mn2t24
    t2m = era5.t2m
    temp_daily = xr.Dataset(
        {
            "t2m": t2m.resample(time="1D").mean(),
            "mx2t24": t2m.resample(time="1D").max(),
            "mn2t24": t2m.resample(time="1D").min(),
        }
    )
    # TODO: See why this is required
    set_lonlat_attrs(temp_daily)
    tp_bc = era5.tp_bc.resample(valid_time="D").sum().rename({"valid_time": "time"})

    # do daily resample of entire dataset
    pevt: xr.DataArray = xclim.indicators.atmos.potential_evapotranspiration(
        tasmin=temp_daily.mn2t24, tasmax=temp_daily.mx2t24, tas=temp_daily.t2m
    ).rename("pevt")  # type: ignore
    era5d = xr.merge([pevt, tp_bc]).rename({"latitude": "lat", "longitude": "lon"})

    # era5: do weekly resample and get balance (wb)
    era5w = (
        era5d.resample(time="7D", closed="left", label="left").sum().astype("float32")
    )  # noqa: F841

    # drop step dimension as that is not present in ERA5
    prev_forecast = prev_forecast.drop_vars("step")
    cur_forecast = cur_forecast.drop_vars("step")
    # concatenate datasets -- include pevt and tp_bc
    ds = xr.concat([era5w, prev_forecast, cur_forecast], dim="time")
    exp_time_len = window_weeks + len(cf.time) - 1
    if len(ds.time) != window_weeks + len(cf.time) - 1:
        raise MismatchError(
            f"Concatenation of ERA5, previous week forecast and current forecast did not result in expected length={exp_time_len}"
        )

    # calculate wb_bc = tp_bc - pevt
    ds["wb_bc"] = ds.tp_bc - ds.pevt
    ds = ds.drop_vars("pevt")

    # perform weekly rolling mean with center=False and window=window --> wb_bc
    ds_ma = (
        ds.rolling(time=window_weeks, center=False).mean(dim="time").dropna(dim="time")
    )
    # rename lat/lon to latitude, longitude
    # gamma parameters are estimated using ERA5 data that uses latitude, longitude names
    ds_ma = ds_ma.rename({"lat": "latitude", "lon": "longitude"})

    # standardised precipitation index (bias corrected) calculation
    gamma_spi = xr.apply_ufunc(
        gamma_func, ds_ma.tp_bc, spi_gamma_params.alpha, spi_gamma_params.beta
    )
    spi_bc: xr.DataArray = xr.apply_ufunc(norminv, gamma_spi).astype("float32")

    # standardised precipitation-evaporation index (bias corrected) calculation
    gamma_spei = xr.apply_ufunc(
        gamma_func, ds_ma.wb_bc, spei_gamma_params.alpha, spei_gamma_params.beta
    )
    spei_bc = xr.apply_ufunc(norminv, gamma_spei).astype("float32")

    # check lengths
    if (spi_bc.time != cf.time).any():
        raise AxisMismatchError(
            f"Temporal axis mismatch between spi_bc {spi_bc.time} and forecast {cf.time}"
        )
    if (spei_bc.time != cf.time).any():
        raise AxisMismatchError(
            f"Temporal axis mismatch between spi_bc {spei_bc.time} and forecast {cf.time}"
        )
    # rename coords back to lat/lon to assign
    spi_bc = spi_bc.rename({"latitude": "lat", "longitude": "lon"})
    spei_bc = spei_bc.rename({"latitude": "lat", "longitude": "lon"})
    return cf.assign(spi_bc=spi_bc, spei_bc=spei_bc)


def validate_output(p: Path | str) -> pd.DataFrame:
    # TODO: add validation here
    return pd.read_csv(p)


def main():
    free_gb = get_free_gigabytes()
    if free_gb < MIN_DISK_SPACE_GB:
        fail(
            "!!! Not enough disk space:",
            f"{MIN_DISK_SPACE_GB} GB required, found {free_gb:.1f} free",
        )
        sys.exit(1)
    sources = get_path("sources", ISO3, "ecmwf")
    output_root = get_path("output", ISO3, "ecmwf")
    dengue_predictions_folder = get_path("output", ISO3, "dengue")
    args = parse_args()
    cache = list(STEPS.keys()) if args.cache == "all" else args.cache.split(",")

    date = args.date
    dengue_dummy_pred = dengue_predictions_folder / f"{ISO3}-{ADMIN}-{date}-example.csv"
    dengue_reload_flag_file = dengue_predictions_folder / ".reload"

    forecast_instant = sources / f"{ISO3}-{date}-ecmwf.forecast.instant.nc"
    forecast_accum = sources / f"{ISO3}-{date}-ecmwf.forecast.accum.nc"
    forecast_corrected = sources / f"{ISO3}-{date}-ecmwf.forecast.corrected.nc"
    forecast_corrected_indices = (  # noqa: F841
        sources / f"{ISO3}-{date}-ecmwf.forecast.corrected_spei.nc"
    )
    forecast_zonal_stats = output_root / f"{ISO3}-{ADMIN}-{date}-ecmwf.forecast.nc"

    run(
        cache,
        "dl",
        [forecast_instant, forecast_accum],
        get_forecast_open_data,
        ISO3,
        date,
    )
    run(
        cache,
        "bc",
        forecast_corrected,
        bias_correct_forecast_from_paths,
        get_file("observation"),
        get_file("historical-forecast"),
        f"{ISO3}-{date}",
        REGION.bbox.int(),
    )
    output = run(
        cache, "zs", [forecast_zonal_stats], process_forecast, f"{ISO3}-{ADMIN}", date
    )
    msg("... Forecast with zonal aggregation:", output[0])
    msg("==> Running model:", args.model)
    match args.model:
        case "dummy":
            geom = REGION.read()
            df = generate_dummy_data(geom.GID_2)
            df["truth"] = 0
            df.to_csv(dengue_dummy_pred, index=False)
            msg("... Model output:", dengue_dummy_pred)
            dengue_reload_flag_file.write_text(date)

        case "actual-1":
            if not MODEL_DATA_HOME.exists():
                fail("[!] Cannot run model without model data folder", MODEL_DATA_HOME)
                sys.exit(1)
            run_actual_1()
            if not MODEL_EXPECTED_OUTPUT.exists():
                fail("[!] Model did not produce output")
            df = validate_output(MODEL_EXPECTED_OUTPUT)
            if df is None:
                fail("[!] Model failed validation")
            else:
                min_date = df.date.min()
                df.to_csv(
                    dengue_predictions_folder
                    / f"{ISO3}-{ADMIN}-{min_date}-forecast.csv",
                    index=False,
                )
                dengue_reload_flag_file.write_text(min_date)


if __name__ == "__main__":
    main()
