"""Native telemetry tables and quality annotations for the Spa dataset splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from formularl.geometry import (
    GAP_THRESHOLDS_M,
    SHORT_CHORD_M,
    ArcLengths,
    FloatArray,
    arc_lengths,
)
from formularl.synchronization import (
    KPH_PER_MPS,
    RESIDUAL_THRESHOLD_M,
    compare_intervals,
    persistent_residual,
    recovery_residuals,
)
from formularl.telemetry import (
    COORDINATES,
    DRIVERS,
    SPEED_FLAG_KMH,
    normalize_position,
    session_timestamps,
)

SPLITS = {"reconstruction": DRIVERS, "validation": ("NOR", "PIA", "RUS")}
REFERENCE_DRIVER = "VER"
RECURRENCE_ARC_M = 2_182.050412
POSITION_REQUIRED = (*COORDINATES, "SessionTime")
CAR_REQUIRED = ("Speed", "SessionTime")
CAR_OPTIONAL = ("Throttle", "Brake", "RPM", "nGear", "DRS")


@dataclass(frozen=True)
class QualityThresholds:
    short_horizontal_m: float = SHORT_CHORD_M
    large_gap_m: float = GAP_THRESHOLDS_M[0]
    severe_gap_m: float = GAP_THRESHOLDS_M[1]
    speed_kmh: float = SPEED_FLAG_KMH
    persistent_residual_m: float = RESIDUAL_THRESHOLD_M
    recurrence_radius_m: float = 25.0


DEFAULT_THRESHOLDS = QualityThresholds()


def dataset_paths(output: Path, split: str, driver: str) -> tuple[Path, Path]:
    """Resolve each assigned driver to a unique pair of native CSV paths."""
    if driver not in SPLITS[split]:
        raise ValueError(f"{driver} is not assigned to {split}.")
    lap_dir = output / split / driver
    return lap_dir / "position.csv", lap_dir / "car.csv"


def clock_columns(
    raw: pd.DataFrame,
    timestamps: pd.Series,
    lap_start: pd.Timedelta,
    lap_end: pd.Timedelta,
) -> pd.DataFrame:
    clock = pd.DataFrame(
        raw.reindex(columns=["Date", "SessionTime", "Time"]).astype("string")
    ).set_axis(["raw_date", "raw_session_time", "raw_slice_time"], axis=1)
    clock["session_time_s"] = timestamps.dt.total_seconds()
    clock["lap_time_s"] = (timestamps - lap_start).dt.total_seconds()
    clock["within_recorded_lap"] = pd.array(
        (timestamps >= lap_start) & (timestamps <= lap_end), dtype="boolean"
    )
    valid_time = np.isfinite(clock["session_time_s"])
    clock.loc[
        ~valid_time | pd.isna(lap_start) | pd.isna(lap_end), "within_recorded_lap"
    ] = pd.NA
    clock["flag_invalid_session_time"] = ~valid_time
    clock["flag_duplicate_timestamp"] = valid_time & timestamps.duplicated(keep=False)
    elapsed = clock["session_time_s"].diff()
    clock["flag_nonpositive_timestamp_step"] = np.isfinite(elapsed) & (elapsed <= 0)
    return clock


def position_table(
    raw: pd.DataFrame, lap_start: pd.Timedelta, lap_end: pd.Timedelta
) -> tuple[pd.DataFrame, ArcLengths]:
    raw = raw.reset_index(drop=True)
    metric = normalize_position(raw, lap_start)
    table = clock_columns(raw, metric["SessionTime"], lap_start, lap_end)
    raw_coordinates = {
        "X": "raw_x_dm",
        "Y": "raw_y_dm",
        "Z": "raw_z_dm",
        "Status": "raw_status",
        "Source": "raw_source",
    }
    table[list(raw_coordinates.values())] = pd.DataFrame(
        raw.reindex(columns=raw_coordinates)
    ).rename(columns=raw_coordinates)
    table[["x_m", "y_m", "z_m"]] = metric[COORDINATES]
    xyz = metric[COORDINATES].to_numpy()
    arc = arc_lengths(xyz)
    table["arc_horizontal_m"] = arc.cumulative_horizontal
    table["arc_3d_m"] = arc.cumulative_3d
    table["flag_invalid_coordinates"] = ~np.isfinite(xyz).all(axis=1)
    return table, arc


def car_table(
    raw: pd.DataFrame,
    lap_start: pd.Timedelta,
    lap_end: pd.Timedelta,
    thresholds: QualityThresholds,
) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    table = clock_columns(raw, session_timestamps(raw), lap_start, lap_end)
    channels = {
        "Speed": "raw_speed_kmh",
        "Throttle": "throttle_pct",
        "Brake": "brake",
        "RPM": "rpm",
        "nGear": "gear",
        "DRS": "drs",
        "Source": "raw_source",
    }
    table[list(channels.values())] = pd.DataFrame(raw.reindex(columns=channels)).rename(
        columns=channels
    )
    raw_kmh = pd.to_numeric(table["raw_speed_kmh"], errors="coerce").astype(float)
    table["speed_mps"] = raw_kmh / KPH_PER_MPS
    table["flag_invalid_speed"] = ~np.isfinite(raw_kmh) | (raw_kmh < 0)
    table["flag_reported_speed_high"] = np.isfinite(raw_kmh) & (
        raw_kmh > thresholds.speed_kmh
    )
    measured_optional = [name for name in CAR_OPTIONAL if name in raw]
    numeric_optional = (
        pd.DataFrame(raw[measured_optional])
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )
    table["flag_invalid_optional_channel"] = ~np.isfinite(numeric_optional).all(axis=1)
    return table


def annotate_position(
    position: pd.DataFrame,
    arc: ArcLengths,
    car: pd.DataFrame,
    reference_arc_m: FloatArray,
    lap_start_s: float,
    thresholds: QualityThresholds,
) -> None:
    """Attach interval quality to its ending sample, preserving native rows."""
    time = position["session_time_s"].to_numpy()
    car_time = car["session_time_s"].to_numpy()
    speed = car["speed_mps"].to_numpy()
    intervals = compare_intervals(time, arc, car_time, speed, lap_start_s)
    duration = intervals["duration_s"]
    start, end = time[:-1], time[1:]
    one, two = recovery_residuals(
        time, arc.cumulative_3d, car_time, speed, (start + end) / 2
    )
    supported = np.isfinite(one) & np.isfinite(two)
    persistent = persistent_residual(one, two, thresholds.persistent_residual_m)
    left, right = reference_arc_m[:-1], reference_arc_m[1:]
    projected = np.isfinite(left) & np.isfinite(right) & (right > left)
    region_start = RECURRENCE_ARC_M - thresholds.recurrence_radius_m
    region_end = RECURRENCE_ARC_M + thresholds.recurrence_radius_m
    region = (right >= region_start) & (left <= region_end)
    position["ver_reference_arc_3d_m"] = reference_arc_m
    horizontal_available = np.isfinite(arc.horizontal_segments)
    spatial_available = np.isfinite(arc.spatial_segments)
    position_speed = intervals["position_speed_kmh"].to_numpy()

    per_interval = pd.DataFrame(
        {
            "interval_s": duration,
            "horizontal_chord_m": arc.horizontal_segments,
            "chord_3d_m": arc.spatial_segments,
            "position_speed_mps": intervals["position_speed_kmh"] / KPH_PER_MPS,
            "matched_car_speed_mps": intervals["car_speed_kmh"] / KPH_PER_MPS,
            "matched_car_distance_m": intervals["car_distance_m"],
            "signed_distance_residual_m": intervals["signed_distance_discrepancy_m"],
            "residual_1s_m": one,
            "residual_2s_m": two,
            "synchronization_status": intervals["status"],
            "flag_invalid_interval_time": pd.array(
                ~np.isfinite(duration) | (duration <= 0), dtype="boolean"
            ),
            "flag_invalid_chord": pd.array(
                ~np.isfinite(arc.spatial_segments), dtype="boolean"
            ),
            "flag_degenerate_segment": nullable_flag(
                arc.horizontal_segments == 0, horizontal_available
            ),
            "flag_short_horizontal_chord": nullable_flag(
                arc.horizontal_segments < thresholds.short_horizontal_m,
                horizontal_available,
            ),
            "flag_large_spatial_gap": nullable_flag(
                arc.spatial_segments > thresholds.large_gap_m, spatial_available
            ),
            "flag_severe_spatial_gap": nullable_flag(
                arc.spatial_segments > thresholds.severe_gap_m, spatial_available
            ),
            "flag_position_speed_high": nullable_flag(
                position_speed > thresholds.speed_kmh, np.isfinite(position_speed)
            ),
            "flag_no_car_coverage": pd.array(
                ~np.isfinite(intervals["car_distance_m"]), dtype="boolean"
            ),
            "flag_persistent_residual": nullable_flag(persistent, supported),
            "flag_recurrent_reference_region": nullable_flag(region, projected),
        },
        index=intervals.index,
    )
    per_interval.index += 1
    position[per_interval.columns] = per_interval.reindex(position.index)


def nullable_flag(value: np.ndarray, available: np.ndarray) -> pd.arrays.BooleanArray:
    return pd.array(np.where(available, value, None), dtype="boolean")


def identify_samples(
    table: pd.DataFrame, split: str, driver: str, lap_number: int | None
) -> pd.DataFrame:
    """Use zero-based native row indices so interval IDs equal ending sample IDs."""
    return table.assign(
        driver=driver,
        split=split,
        lap_number=lap_number,
        sample_index=np.arange(len(table)),
    )[["split", "driver", "lap_number", "sample_index", *table.columns]]
