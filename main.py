# dart-runner
#
# Orchestrates DART-Pipeline runs with forecast downloads and model predictions

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

CHECKSUMS: dict[str, tuple[str, str]] = {
    "observation": (
        "282ad3246e79c28c68e469e74388fe68b6e3496673a2e807178151f462adb729",
        "T2m_r_tp_Vietnam_ERA5.nc",
    ),
    "historical-forecast": (
        "fd40c75c3dcf8bb25e0c32e5689cef941927964a132feab3686ae0d0d9454a7a",
        "eefh_testv2_test_githubv1_3.nc",
    ),
}


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
                pred_lower = max(0, base_pred - margin)
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
    if alias not in CHECKSUMS:
        raise ValueError(f"Could not find {alias=} in CHECKSUMS")
    expected_checksum, file = CHECKSUMS[alias]
    path = root / file
    if not path.exists():
        raise FileNotFoundError(f"Could not find {file=} under {root=}")
    actual_checksum = sha256(path)
    if actual_checksum == expected_checksum:
        return path
    raise ValueError(f"""Checksum verification failed for {path}:
    Expected: {expected_checksum}
      Actual: {actual_checksum}""")


def run_in_docker(image, command, volumes={}, remove=True):
    client = docker.from_env()
    container = client.containers.run(
        image=image,
        command=command,
        volumes=volumes,
        remove=remove,
        detach=False,
    )
    exitcode = 0
    return exitcode, container.decode()


def msg(bold: str, *args):
    print(f"\033[1m{bold}\033[0m", *args)


def fail(bold: str, *args):
    print(f"\033[31;1m{bold}\033[0m", *args)


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
    parser.add_argument("--model", help="Name of the model (docker image) to use")
    parser.add_argument(
        "--cache",
        help="Use cached files for the following (comma separated) steps if available",
        default="dl",
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
    msg("==> Running dummy model")
    geom = region.read()
    df = generate_dummy_data(geom.GID_2)
    df["truth"] = 0
    df.to_csv(dengue_dummy_pred, index=False)
    msg("... Model output:", dengue_dummy_pred)
    dengue_reload_flag_file.write_text(date)
    # run_in_docker(docker_image, "Rscript /app/run.R")
    # expected_output = get_path(
    #     "output", ISO3, "model", "{ISO3}-{ADMIN}-{date}-dengue-predictions.csv"
    # )
    # if not expected_output.exists():
    #     fail("Model failed", "expected output at:", expected_output)


if __name__ == "__main__":
    main()
