# dart-runner
#
# Orchestrates DART-Pipeline runs with forecast downloads and model predictions

import os
import sys
import shutil
import docker
import logging
import argparse
from pathlib import Path
from typing import overload
from typing import Callable

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from geoglue.util import sha256
from geoglue.region import gadm
from dart_pipeline.metrics.ecmwf import get_forecast_open_data, process_forecast
from dart_pipeline.paths import get_path
from dart_bias_correct.forecast import bias_correct_forecast_from_paths

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO
)

MIN_DISK_SPACE_GB = 40
STEPS = {
    "dl": "Fetching forecast",
    "bc": "Performing forecast bias correction",
    "zs": "Processing forecast",
    "m": "Running model",
}
BASE_PATH = Path.home() / ".local/share/dart-pipeline"
ISO3 = "VNM"
ADMIN = 2
DEFAULT_FORECAST_DATE = "2025-05-28"
OBSERVATION_REFERENCE = "T2m_r_tp_Vietnam_ERA5.nc"
PREDICTION_CSV_SUFFIX = "dengue-predictions.csv"


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
""")

ALIASES = {
    "observation": "T2m_r_tp_Vietnam_ERA5.nc",
    "historical-forecast": "eefh_testv2_test_githubv1_3.nc",
}


volumes = {
    "local_renv_cache": {
        "bind": "/modelling/renv/.cache",
        "mode": "rw",
    },
    os.path.expanduser("~/.local/share/dart-pipeline/sources/VNM/model"): {
        "bind": "/modelling/data",
        "mode": "rw",
    },
}

model_command = """
mkdir -p /modelling/renv/.cache && \
Rscript -e 'renv::paths$cache();renv::restore(prompt=FALSE)' && \
Rscript 01_data_prepper.R \
    data/fake_full_incidence.csv \
    data/weather/VNM-2-2001-2019-era5.nc \
    docker_container_test_1-12wa \
    data/gadm/gid2_lookup_df.rds && \
Rscript 99_mlr3_modelling.R \
    data/weekly_inc_weather_docker_container_test_1-12wa.rds \
    data/tuned_lrners_tunedRF_fh-1-12_wa.rds \
    docker_container_test_1-12wa \
    --future_plan multicore \
    --future_workers 7
"""


def run_actual_1():
    client = docker.from_env()
    container = client.containers.run(
        image="dart-model-container:0.3.2",
        command=["/bin/bash", "-c", model_command],
        working_dir="/modelling",
        environment={"RENV_PATHS_CACHE": "/modelling/renv/.cache"},
        volumes=volumes,
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
    start_date = datetime.today() if start_date is None else pd.to_datetime(start_date)
    dates = [start_date + timedelta(weeks=i) for i in range(weeks)]

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
    today = datetime.today().date().isoformat()
    parser = argparse.ArgumentParser(
        description="Orchestrates DART-Pipeline runs with forecast downloads and model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Cache steps (default=dl)
  * dl: Forecast downloaded files, e.g. VNM-2025-05-28-ecmwf.forecast.nc
  * bc: Bias corrected files, e.g. VNM-2025-05-28-ecmwf.forecast.corrected.nc
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
    region = gadm(ISO3, ADMIN)
    dengue_dummy_pred = dengue_predictions_folder / f"{ISO3}-{ADMIN}-{date}-example.csv"
    dengue_reload_flag_file = dengue_predictions_folder / ".reload"

    forecast_instant = sources / f"{ISO3}-{date}-ecmwf.forecast.instant.nc"
    forecast_accum = sources / f"{ISO3}-{date}-ecmwf.forecast.accum.nc"
    forecast_corrected = sources / f"{ISO3}-{date}-ecmwf.forecast.corrected.nc"
    forecast_zonal_stats = output_root / f"{ISO3}-{ADMIN}-{date}-ecmwf.forecast.nc"

    run(
        cache,
        "dl",
        [forecast_instant, forecast_accum],
        get_forecast_open_data,
        ISO3,
        date,
    )
    corrected_file: Path = run(
        cache,
        "bc",
        forecast_corrected,
        bias_correct_forecast_from_paths,
        get_file("observation"),
        get_file("historical-forecast"),
        f"{ISO3}-{date}",
        region.bbox.int(),
    )
    xr.open_dataset(
        corrected_file, decode_timedelta=True
    )  # try opening to catch errors
    output = run(
        cache, "zs", [forecast_zonal_stats], process_forecast, f"{ISO3}-{ADMIN}", date
    )
    msg("... Forecast with zonal aggregation:", output[0])
    msg("==> Running model:", args.model)
    match args.model:
        case "dummy":
            geom = region.read()
            df = generate_dummy_data(geom.GID_2)
            df["truth"] = 0
            df.to_csv(dengue_dummy_pred, index=False)
            msg("... Model output:", dengue_dummy_pred)
            dengue_reload_flag_file.write_text(date)

        case "actual-1":
            run_actual_1()
            # TODO: check expected output, else fail()


if __name__ == "__main__":
    main()
