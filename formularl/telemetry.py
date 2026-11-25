"""Shared FastF1 acquisition for the Belgian qualifying investigations."""

import sys
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.core import Lap, Laps
from fastf1.exceptions import DataNotLoadedError, F1AuthError, NoLapDataError

from formularl.geometry import FloatArray

EVENT = "2024 Belgian Grand Prix qualifying"
SEASON = 2024
GRAND_PRIX = "Belgian Grand Prix"
SESSION_CODE = "Q"
DRIVERS = ("VER", "LEC", "HAM")
COORDINATES = ["X", "Y", "Z"]
SPA_LENGTH_M = 7_004.0
SPEED_FLAG_KMH = 400.0
CAR_PADDING_SAMPLES = 4


def load_laps(cache_dir: Path, *, offline: bool = False) -> Laps | None:
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache_dir))
    if offline:
        fastf1.Cache.offline_mode(enabled=True)

    try:
        session = fastf1.get_session(SEASON, GRAND_PRIX, SESSION_CODE)
        session.load(laps=True, telemetry=True, weather=False, messages=True)
        laps = session.laps
    except (
        DataNotLoadedError,
        F1AuthError,
        NoLapDataError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Could not load lap timing for {EVENT}: {error}", file=sys.stderr)
        return None
    if laps.empty:
        print(f"No lap timing available for {EVENT}.", file=sys.stderr)
        return None
    if "Deleted" not in laps.columns:
        print(
            "Lap deletion status is unavailable; race control messages are required.",
            file=sys.stderr,
        )
        return None
    return laps


def select_fastest_valid_lap(laps: Laps, driver: str) -> Lap | None:
    """Choose the shortest recorded lap time that passes the validity filters."""
    valid = laps[
        laps["Driver"].eq(driver)
        & laps["Deleted"].eq(False)
        & laps["IsAccurate"].eq(True)
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["LapTime"].notna()
    ]
    if valid.empty:
        return None
    return valid.sort_values("LapTime", kind="stable").iloc[0]


def read_raw_position(lap: Lap) -> pd.DataFrame | None:
    """Read FastF1's native position rows even if a required channel is absent."""
    driver = lap["Driver"]
    try:
        raw = lap.get_pos_data(pad=0, interpolate_edges=False)
    except (DataNotLoadedError, KeyError) as error:
        print(f"{driver}: position telemetry unavailable ({error}).", file=sys.stderr)
        return None
    if raw.empty:
        print(f"{driver}: position telemetry is empty (0 samples).", file=sys.stderr)
        return None
    return raw


def read_position(lap: Lap) -> pd.DataFrame | None:
    """Keep FastF1's original samples and build a separate metric table."""
    raw = read_raw_position(lap)
    if raw is None:
        return None

    missing_columns = {*COORDINATES, "SessionTime"}.difference(raw.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        print(
            f"{lap['Driver']}: position telemetry is missing columns {names}.",
            file=sys.stderr,
        )
        return None
    return normalize_position(raw, lap["LapStartTime"])


def normalize_position(raw: pd.DataFrame, lap_start: pd.Timedelta) -> pd.DataFrame:
    """Derive the metric position table without mutating the source."""
    timestamps = session_timestamps(raw)
    coordinates = pd.DataFrame(raw.reindex(columns=COORDINATES))
    position = coordinates.apply(pd.to_numeric, errors="coerce").astype(float) / 10.0
    position["SessionTime"] = timestamps
    position["ElapsedSeconds"] = (timestamps - lap_start).dt.total_seconds()
    position["IntervalMillis"] = timestamps.diff().dt.total_seconds() * 1_000
    return position


def read_car(lap: Lap, *, require_channels: bool = True) -> pd.DataFrame | None:
    """Read original car rows, with padding for boundary interpolation and lags."""
    try:
        raw = lap.get_car_data(
            pad=CAR_PADDING_SAMPLES, pad_side="both", interpolate_edges=False
        )
    except (DataNotLoadedError, KeyError) as error:
        print(f"  Car telemetry unavailable ({error}).", file=sys.stderr)
        return None
    if raw.empty:
        print("  Car telemetry is empty.", file=sys.stderr)
        return None
    missing = {"SessionTime", "Speed"}.difference(raw.columns)
    if require_channels and missing:
        print(f"  Car telemetry is missing columns {sorted(missing)}.", file=sys.stderr)
        return None
    return raw


def time_seconds(samples: pd.DataFrame) -> FloatArray:
    return samples["SessionTime"].dt.total_seconds().to_numpy()


def session_timestamps(raw: pd.DataFrame) -> pd.Series:
    if "SessionTime" not in raw:
        return pd.Series(pd.NaT, index=raw.index, dtype="timedelta64[ns]")
    return pd.to_timedelta(raw["SessionTime"], errors="coerce")
