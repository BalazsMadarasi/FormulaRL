"""Integrate reported speed over requested intervals on a common time axis."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from formularl.geometry import ArcLengths, FloatArray

KPH_PER_MPS = 3.6
LOW_SPEED_RATIO = 0.75
RESIDUAL_THRESHOLD_M = 5.0
RECOVERY_WINDOWS_S = (1.0, 2.0)


def integrate_speed(
    time_s: NDArray[np.float64],
    speed_mps: NDArray[np.float64],
    starts_s: NDArray[np.float64],
    ends_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return distances for paired bounds, or NaN where coverage is unsupported.

    Speed is linear between adjacent samples. Spans touching duplicate time
    samples or invalid speeds are unavailable. A missing or decreasing source
    timestamp makes the source time axis unavailable.
    """
    distance = np.full(starts_s.shape, np.nan)
    elapsed = np.diff(time_s.astype(float, copy=False))
    if not np.isfinite(time_s).all() or (elapsed < 0).any() or not elapsed.size:
        return distance

    repeated = elapsed == 0
    invalid = ~np.isfinite(speed_mps) | (speed_mps < 0)
    invalid[:-1] |= repeated
    invalid[1:] |= repeated
    usable = ~(invalid[:-1] | invalid[1:])

    in_range = (starts_s >= time_s[0]) & (ends_s <= time_s[-1]) & (ends_s > starts_s)
    rows = np.flatnonzero(in_range)
    left = np.searchsorted(time_s, starts_s[rows], side="right") - 1
    right = np.searchsorted(time_s, ends_s[rows], side="left") - 1
    invalid_count = np.r_[0, np.cumsum(~usable)]
    supported = invalid_count[right + 1] == invalid_count[left]
    rows, left, right = rows[supported], left[supported], right[supported]

    # Prefix trapezoids handle full spans. Integrate the linear speed exactly
    # over the two partial spans at the query boundaries.
    trapezoids = np.zeros_like(elapsed)
    slopes = np.zeros_like(elapsed)
    before = speed_mps[:-1][usable]
    after = speed_mps[1:][usable]
    trapezoids[usable] = elapsed[usable] * (before + after) / 2
    slopes[usable] = (after - before) / elapsed[usable]
    prefix_m = np.r_[0.0, np.cumsum(trapezoids)]

    start_offset = starts_s[rows] - time_s[left]
    end_offset = ends_s[rows] - time_s[right]
    start_partial = start_offset * (speed_mps[left] + slopes[left] * start_offset / 2)
    end_partial = end_offset * (speed_mps[right] + slopes[right] * end_offset / 2)
    distance[rows] = prefix_m[right] - prefix_m[left] + end_partial - start_partial
    return distance


def position_window_distance(
    time_s: FloatArray,
    cumulative_m: FloatArray,
    starts_s: FloatArray,
    ends_s: FloatArray,
) -> FloatArray:
    """Allocate each position chord linearly in time for diagnostic windows."""
    if not len(time_s) or not np.isfinite(time_s).all() or (np.diff(time_s) <= 0).any():
        return np.full(starts_s.shape, np.nan)
    start_m = np.interp(starts_s, time_s, cumulative_m, left=np.nan, right=np.nan)
    end_m = np.interp(ends_s, time_s, cumulative_m, left=np.nan, right=np.nan)
    return np.where(ends_s > starts_s, end_m - start_m, np.nan)


def recovery_residuals(
    time_s: FloatArray,
    cumulative_m: FloatArray,
    car_time_s: FloatArray,
    car_speed_mps: FloatArray,
    centers_s: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compare position and car distances on Task 3B's two recovery windows."""
    distances = []
    for width in RECOVERY_WINDOWS_S:
        start, end = centers_s - width / 2, centers_s + width / 2
        distances.append(
            position_window_distance(time_s, cumulative_m, start, end)
            - integrate_speed(car_time_s, car_speed_mps, start, end)
        )
    return distances[0], distances[1]


def persistent_residual(
    one_second: FloatArray,
    two_seconds: FloatArray,
    threshold_m: float = RESIDUAL_THRESHOLD_M,
) -> NDArray[np.bool_]:
    return (
        (np.abs(one_second) > threshold_m)
        & (np.abs(two_seconds) > threshold_m)
        & (np.sign(one_second) == np.sign(two_seconds))
    )


def time_at_reference_arc(
    time_s: FloatArray, sample_arc: FloatArray, reference_m: FloatArray
) -> FloatArray:
    """Invert only single, forward, supported projected chords."""
    left, right = sample_arc[:-1], sample_arc[1:]
    locations = reference_m[:, None]
    usable = (locations >= left) & (locations < right) & (np.diff(time_s) > 0)
    rows = np.flatnonzero(usable.sum(axis=1) == 1)
    result = np.full(reference_m.shape, np.nan)
    if rows.size:
        i = usable[rows].argmax(axis=1)
        fraction = (reference_m[rows] - left[i]) / (right[i] - left[i])
        result[rows] = time_s[i] + fraction * (time_s[i + 1] - time_s[i])
    return result


def compare_intervals(
    position_time: FloatArray,
    arc: ArcLengths,
    car_time: FloatArray,
    car_speed: FloatArray,
    lap_start: float,
) -> pd.DataFrame:
    starts, ends = position_time[:-1], position_time[1:]
    duration = ends - starts
    valid_time = np.isfinite(duration) & (duration > 0)
    position_speed = (
        np.divide(
            arc.spatial_segments,
            duration,
            out=np.full_like(duration, np.nan),
            where=valid_time,
        )
        * KPH_PER_MPS
    )
    car_distance = integrate_speed(car_time, car_speed, starts, ends)
    car_average_speed = (
        np.divide(
            car_distance,
            duration,
            out=np.full_like(duration, np.nan),
            where=valid_time,
        )
        * KPH_PER_MPS
    )
    valid_car_time = (
        len(car_time) >= 2
        and np.isfinite(car_time).all()
        and (np.diff(car_time) >= 0).all()
    )
    inside_coverage = (
        (starts >= car_time[0]) & (ends <= car_time[-1])
        if valid_car_time
        else np.zeros_like(valid_time)
    )
    status = np.select(
        [
            ~valid_time,
            np.full_like(valid_time, not valid_car_time),
            ~inside_coverage,
            ~np.isfinite(car_distance),
            ~np.isfinite(position_speed),
        ],
        [
            "invalid position timing",
            "unavailable car time axis",
            "outside car coverage",
            "unusable car speed span",
            "unavailable position chord",
        ],
        default="compared",
    )
    distance_error = np.where(valid_time, arc.spatial_segments - car_distance, np.nan)
    speed_error = position_speed - car_average_speed
    low = pd.array(
        position_speed < LOW_SPEED_RATIO * car_average_speed, dtype="boolean"
    )
    low[(status != "compared") | (car_average_speed <= 0)] = pd.NA
    return pd.DataFrame(
        {
            "interval_number": np.arange(1, len(position_time)),
            "start_session_s": starts,
            "end_session_s": ends,
            "start_lap_s": starts - lap_start,
            "end_lap_s": ends - lap_start,
            "end_3d_arc_m": arc.cumulative_3d[1:],
            "duration_s": duration,
            "position_displacement_m": arc.spatial_segments,
            "position_speed_kmh": position_speed,
            "car_distance_m": car_distance,
            "car_speed_kmh": car_average_speed,
            "signed_distance_discrepancy_m": distance_error,
            "absolute_distance_discrepancy_m": np.abs(distance_error),
            "signed_speed_discrepancy_kmh": speed_error,
            "absolute_speed_discrepancy_kmh": np.abs(speed_error),
            "unusually_low": low,
            "status": status,
        }
    )
