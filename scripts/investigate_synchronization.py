"""Compare raw position and car telemetry over the same Spa qualifying intervals."""

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.core import Lap
from matplotlib.collections import LineCollection

from formularl.geometry import ArcLengths, FloatArray, arc_lengths, closure_distances
from formularl.synchronization import (
    KPH_PER_MPS,
    LOW_SPEED_RATIO,
    compare_intervals,
    integrate_speed,
    position_window_distance,
)
from formularl.telemetry import (
    COORDINATES,
    DRIVERS,
    EVENT,
    SPA_LENGTH_M,
    SPEED_FLAG_KMH,
    load_laps,
    read_car,
    read_position,
    select_fastest_valid_lap,
    time_seconds,
)

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_S = (0.5, 1.0, 2.0)
LAGS_S = np.arange(-500, 501, 20) / 1_000
COLORS = {"VER": "tab:blue", "LEC": "tab:red", "HAM": "tab:green"}


@dataclass(frozen=True)
class LapAnalysis:
    position: pd.DataFrame
    arc: ArcLengths
    intervals: pd.DataFrame
    car_cumulative_m: FloatArray
    window_errors: pd.DataFrame


def measurement(value: float, unit: str, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return f"unavailable ({unit})"
    number = f"{value:+.3f}" if signed else f"{value:.3f}"
    return f"{number} {unit}"


def min_median_max(values: FloatArray, unit: str) -> str:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return f"unavailable ({unit})"
    return " / ".join(
        measurement(value, unit)
        for value in (finite.min(), np.median(finite), finite.max())
    )


def report_car(
    car: pd.DataFrame,
    car_time: FloatArray,
    position_time: FloatArray,
    lap_start: float,
    lap_end: float,
) -> None:
    in_lap = (car_time >= lap_start) & (car_time <= lap_end)
    intervals_ms = np.diff(car_time) * 1_000
    speeds = car["Speed"].to_numpy(dtype=float, na_value=np.nan)
    missing = car.isna().sum()
    missing_fields = missing[
        (missing > 0) | missing.index.isin(["SessionTime", "Speed"])
    ]
    duplicates = car["SessionTime"].dropna().duplicated().sum()
    print(
        f"  Car samples: {len(car)} fetched, {in_lap.sum()} within recorded lap boundaries"
    )
    print(f"  Car interval min / median / max: {min_median_max(intervals_ms, 'ms')}")
    print(
        f"  Car timestamps: {duplicates} duplicates, {(intervals_ms <= 0).sum()} "
        f"non-positive intervals, {(~np.isfinite(intervals_ms)).sum()} unavailable intervals"
    )
    print(
        f"  Car missing values ({missing.sum()} across all fetched columns): "
        + ", ".join(f"{key}={count}" for key, count in missing_fields.items())
    )
    print(
        f"  Reported in-lap speed min / median / max: {min_median_max(speeds[in_lap], 'km/h')}"
    )
    flags = (
        (speeds < 0)
        | (speeds > SPEED_FLAG_KMH)
        | np.isinf(speeds)
        | (in_lap & (speeds == 0))
    )
    print(
        f"  Suspicious reported car speeds: {flags.sum()} "
        f"(negative, infinite, >{SPEED_FLAG_KMH:g} km/h, or zero in-lap)"
    )
    for t, speed in zip(car_time[flags], speeds[flags], strict=True):
        print(f"    Session time {measurement(t, 's')}: {speed:g} km/h")

    times = car_time[np.isfinite(car_time)]
    if not times.size:
        print("  Car time coverage: unavailable.")
        return
    first, last = times.min(), times.max()
    print(f"  Car coverage on session clock: {first:.3f} to {last:.3f} s")
    for label, start, end in (
        ("position endpoints", position_time[0], position_time[-1]),
        ("recorded lap boundaries", lap_start, lap_end),
    ):
        first_offset = measurement(first - start, "s", signed=True)
        last_offset = measurement(last - end, "s", signed=True)
        print(
            f"  Car coverage minus {label}: {first_offset} at start, {last_offset} at end"
        )


def report_interval_coverage(intervals: pd.DataFrame) -> None:
    compared = intervals["status"] == "compared"
    print(f"  Synchronized intervals: {compared.sum()}/{len(intervals)}")
    for status, group in intervals.loc[~compared].groupby("status"):
        numbers = ", ".join(group["interval_number"].astype(str))
        print(f"    {status}: {len(group)} intervals ({numbers})")


def anomaly_rows(driver: str, intervals: pd.DataFrame) -> pd.DataFrame:
    neighbor_columns = [
        "duration_s",
        "position_displacement_m",
        "position_speed_kmh",
        "car_speed_kmh",
        "signed_distance_discrepancy_m",
        "unusually_low",
        "status",
    ]
    neighbors = intervals[neighbor_columns]
    with_neighbors = pd.concat(
        [
            intervals,
            neighbors.shift(1).add_prefix("previous_"),
            neighbors.shift(-1).add_prefix("next_"),
        ],
        axis=1,
    )
    anomalies = with_neighbors.loc[
        intervals["position_speed_kmh"] > SPEED_FLAG_KMH
    ].copy()
    anomalies.insert(0, "driver", driver)
    before = anomalies["previous_unusually_low"]
    after = anomalies["next_unusually_low"]
    baseline = intervals["unusually_low"]
    either_low = before.fillna(False) | after.fillna(False)
    either_known = before.notna() | after.notna()
    print(f"  Position speeds >{SPEED_FLAG_KMH:g} km/h: {len(anomalies)}")
    print(
        f"  Neighbor below {LOW_SPEED_RATIO:.0%} of matched car speed:"
        f" previous {before.sum()}/{before.count()}, next {after.sum()}/{after.count()},"
        f" either {either_low.sum()}/{either_known.sum()}."
    )
    print(
        f"  Low-speed baseline across eligible native intervals: {baseline.sum()}/{baseline.count()}"
    )
    return anomalies


def report_lag_scan(
    label: str,
    starts: FloatArray,
    ends: FloatArray,
    position_speed_kmh: FloatArray,
    car_time: FloatArray,
    car_speed: FloatArray,
) -> None:
    duration = ends - starts
    valid = np.isfinite(position_speed_kmh) & (duration > 0)
    starts, ends, duration = starts[valid], ends[valid], duration[valid]
    car_averages = (
        np.array(
            [
                integrate_speed(car_time, car_speed, starts + lag, ends + lag)
                for lag in LAGS_S
            ]
        )
        * KPH_PER_MPS
        / duration
    )
    common = np.isfinite(car_averages).all(axis=0)
    if not common.any():
        print(f"  {label} lag scan: unavailable (no common coverage).")
        return
    errors = position_speed_kmh[valid][common] - car_averages[:, common]
    rmse = np.sqrt(np.mean(errors**2, axis=1))
    tied = np.flatnonzero(np.isclose(rmse, rmse.min(), rtol=0, atol=1e-9))
    best = tied[np.argmin(np.abs(LAGS_S[tied]))]
    zero = np.flatnonzero(LAGS_S == 0)[0]
    limit = " (scan boundary)" if best in (0, len(LAGS_S) - 1) else ""
    print(
        f"  {label} lag scan ({common.sum()} common intervals): best {LAGS_S[best] * 1_000:+.0f} ms{limit},"
        f" RMSE {rmse[zero]:.3f} km/h at zero, {rmse[best]:.3f} km/h at best"
        f", improvement {rmse[zero] - rmse[best]:.3f} km/h"
    )


def time_window_errors(
    intervals: pd.DataFrame,
    position_time: FloatArray,
    arc: ArcLengths,
    car_time: FloatArray,
    car_speed: FloatArray,
) -> pd.DataFrame:
    starts = intervals["start_session_s"].to_numpy()
    ends = intervals["end_session_s"].to_numpy()
    native_speed = intervals["position_speed_kmh"].to_numpy()
    midpoints = (starts + ends) / 2
    errors = {"Native": intervals["signed_speed_discrepancy_kmh"].to_numpy()}
    report_lag_scan("Native", starts, ends, native_speed, car_time, car_speed)

    valid_timeline = (
        np.isfinite(position_time).all() and (np.diff(position_time) > 0).all()
    )
    if not valid_timeline:
        print("  Time-window comparisons unavailable (position time axis is invalid).")
        return pd.DataFrame(
            columns=["Native", *(f"{width:.1f} s" for width in WINDOWS_S)], dtype=float
        )

    for width in WINDOWS_S:
        start = midpoints - width / 2
        end = midpoints + width / 2
        position_avg = (
            position_window_distance(position_time, arc.cumulative_3d, start, end)
            / width
            * KPH_PER_MPS
        )
        car_avg = integrate_speed(car_time, car_speed, start, end) / width * KPH_PER_MPS
        label = f"{width:.1f} s"
        errors[label] = position_avg - car_avg
        if width == 1.0:
            report_lag_scan(label, start, end, position_avg, car_time, car_speed)

    paired = pd.DataFrame(errors).dropna()
    print(
        f"  Time-window errors on {len(paired)}/{len(intervals)} shared midpoint anchors"
    )
    for label, difference in paired.items():
        if difference.empty:
            continue
        absolute = difference.abs()
        rmse = np.sqrt((difference**2).mean())
        print(
            f"    {label}: bias {difference.mean():+.3f} km/h"
            f" | MAE {absolute.mean():.3f} km/h | RMSE {rmse:.3f} km/h"
            f" | p95 absolute {absolute.quantile(0.95):.3f} km/h"
        )
    return paired


def report_distance_comparison(
    position: pd.DataFrame,
    arc: ArcLengths,
    position_time: FloatArray,
    car_time: FloatArray,
    car_speed: FloatArray,
    lap_start: float,
    lap_end: float,
) -> None:
    observed = arc.cumulative_3d[-1]
    _, _, closing_chord = closure_distances(position[COORDINATES].to_numpy())
    closed = observed + closing_chord
    matched_car, full_lap_car = integrate_speed(
        car_time,
        car_speed,
        np.array([position_time[0], lap_start]),
        np.array([position_time[-1], lap_end]),
    )
    print(f"  Official circuit length: {SPA_LENGTH_M:,.0f} m")
    for label, length in (
        ("Observed 3D polyline", observed),
        ("Car integral between position endpoints", matched_car),
        ("Car integral over recorded lap", full_lap_car),
        ("Estimated closed 3D polyline", closed),
    ):
        length_text = measurement(length, "m")
        relation = distance_difference(length, SPA_LENGTH_M)
        print(f"  {label}: {length_text} | vs official: {relation}")
    print(
        f"  Observed position vs matched car integral: {distance_difference(observed, matched_car)}"
    )
    print(
        f"  Closed estimate vs recorded-lap car integral: {distance_difference(closed, full_lap_car)}"
    )


def distance_difference(distance: float, baseline: float) -> str:
    difference = distance - baseline
    if not np.isfinite(difference):
        return "unavailable"
    direction = "below" if difference < 0 else "above" if difference > 0 else "equal"
    percent = (
        f"{abs(difference) / baseline * 100:.3f}%"
        if baseline > 0
        else "percentage unavailable"
    )
    return f"{abs(difference):.3f} m ({percent}, {direction})"


def investigate_lap(
    lap: Lap, position: pd.DataFrame, car: pd.DataFrame | None
) -> tuple[LapAnalysis, pd.DataFrame]:
    position_time = time_seconds(position)
    lap_start = lap["LapStartTime"].total_seconds()
    lap_end = lap["Time"].total_seconds()
    car_time, car_speed = np.empty(0), np.empty(0)
    if car is not None:
        car_time = time_seconds(car)
        car_speed = car["Speed"].to_numpy(dtype=float, na_value=np.nan) / KPH_PER_MPS
        report_car(car, car_time, position_time, lap_start, lap_end)

    arc = arc_lengths(position[COORDINATES].to_numpy())
    intervals = compare_intervals(position_time, arc, car_time, car_speed, lap_start)
    report_interval_coverage(intervals)
    anomalies = anomaly_rows(lap["Driver"], intervals)
    window_errors = time_window_errors(
        intervals, position_time, arc, car_time, car_speed
    )
    report_distance_comparison(
        position, arc, position_time, car_time, car_speed, lap_start, lap_end
    )

    car_cumulative = integrate_speed(
        car_time,
        car_speed,
        np.full_like(position_time, position_time[0]),
        position_time,
    )
    if np.isfinite(car_cumulative[1:]).any():
        car_cumulative[0] = 0.0
    return LapAnalysis(
        position, arc, intervals, car_cumulative, window_errors
    ), anomalies


def save_plots(analyses: dict[str, LapAnalysis], output_dir: Path) -> None:
    matplotlib.use("Agg")
    speed_figure, speed_panels = plt.subplots(
        len(analyses),
        1,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    discrepancy_figure, discrepancy = plt.subplots(
        figsize=(10, 5), layout="constrained"
    )
    distance_figure, (distance, residual) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, layout="constrained"
    )
    map_figure, circuit = plt.subplots(figsize=(9, 7), layout="constrained")
    window_figure, (mae, tail) = plt.subplots(
        1, 2, figsize=(10, 5), sharex=True, layout="constrained"
    )

    for (driver, analysis), speed in zip(
        analyses.items(), speed_panels[:, 0], strict=True
    ):
        intervals = analysis.intervals
        midpoint = (intervals["start_lap_s"] + intervals["end_lap_s"]) / 2
        color = COLORS[driver]
        flagged = intervals["position_speed_kmh"] > SPEED_FLAG_KMH
        speed.plot(
            midpoint,
            intervals["car_speed_kmh"],
            color="0.15",
            linewidth=1.4,
            label="Car interval average",
        )
        speed.scatter(
            midpoint[~flagged],
            intervals.loc[~flagged, "position_speed_kmh"],
            s=9,
            color=color,
            label="Position interval average",
        )
        speed.scatter(
            midpoint[flagged],
            intervals.loc[flagged, "position_speed_kmh"],
            s=20,
            color="crimson",
            marker="x",
            label=f">{SPEED_FLAG_KMH:g} km/h",
            zorder=3,
        )
        speed.set(title=driver, ylabel="Average speed (km/h)")
        if speed is speed_panels[0, 0]:
            speed.legend(ncols=3)
        discrepancy.scatter(
            intervals["duration_s"] * 1_000,
            intervals["signed_speed_discrepancy_kmh"],
            s=10,
            alpha=0.55,
            color=color,
            label=driver,
        )
        time = analysis.position["ElapsedSeconds"]
        position_distance = analysis.arc.cumulative_3d
        distance.plot(time, position_distance, color=color, label=f"{driver} position")
        distance.plot(
            time,
            analysis.car_cumulative_m,
            color=color,
            linestyle="--",
            label=f"{driver} car",
        )
        residual.plot(
            time,
            position_distance - analysis.car_cumulative_m,
            color=color,
            label=driver,
        )

        xy = analysis.position[["X", "Y"]].to_numpy()
        circuit.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.55, label=driver)
        chords = np.stack((xy[:-1], xy[1:]), axis=1)
        circuit.add_collection(
            LineCollection(
                chords[flagged.to_numpy()], colors="crimson", linewidths=2.4, zorder=3
            )
        )

        absolute = analysis.window_errors.abs()
        if not absolute.empty:
            mae.plot(
                absolute.columns, absolute.mean(), marker="o", color=color, label=driver
            )
            tail.plot(
                absolute.columns,
                absolute.quantile(0.95),
                marker="o",
                color=color,
                label=driver,
            )

    speed_panels[-1, 0].set_xlabel("Time since recorded lap start (s)")
    discrepancy.axhline(0, color="0.4", linewidth=0.8)
    discrepancy.set(
        xlabel="Position sampling interval (ms)",
        ylabel="Position minus car speed (km/h)",
        title="Speed discrepancy by position sample interval",
        xscale="log",
    )
    distance.set(
        title="Distance from the first position sample",
        ylabel="Cumulative distance (m)",
    )
    residual.axhline(0, color="0.4", linewidth=0.8)
    residual.set(
        xlabel="Time since recorded lap start (s)", ylabel="Position minus car (m)"
    )
    circuit.plot(
        [],
        [],
        color="crimson",
        linewidth=2.4,
        label=f"Implied speed >{SPEED_FLAG_KMH:g} km/h",
    )
    circuit.set(
        title="Location of anomalous speed intervals", xlabel="X (m)", ylabel="Y (m)"
    )
    circuit.set_aspect("equal", adjustable="box")
    mae.set(
        title="Mean absolute discrepancy",
        xlabel="Time window",
        ylabel="Speed error (km/h)",
    )
    tail.set(
        title="95th percentile absolute discrepancy",
        xlabel="Time window",
        ylabel="Speed error (km/h)",
    )
    for axes in (discrepancy, distance, residual, circuit, mae, tail):
        if axes.get_legend_handles_labels()[0]:
            axes.legend()

    for figure, title, name in (
        (speed_figure, "Position and car interval speeds", "synchronized_speeds.png"),
        (discrepancy_figure, "Native intervals", "speed_discrepancy_by_interval.png"),
        (
            distance_figure,
            "Integrated car speed and position arc length",
            "synchronized_distances.png",
        ),
        (
            map_figure,
            "Position intervals above the diagnostic threshold",
            "speed_anomalies_xy.png",
        ),
        (
            window_figure,
            "Speed agreement by temporal scale",
            "speed_discrepancy_by_window.png",
        ),
    ):
        figure.suptitle(f"{EVENT}\n{title}")
        figure.savefig(output_dir / name, dpi=150)
        plt.close(figure)


def main(*, offline: bool = False) -> int:
    laps = load_laps(ROOT / "cache", offline=offline)
    if laps is None:
        return 1

    analyses: dict[str, LapAnalysis] = {}
    anomalies: list[pd.DataFrame] = []
    print(f"\n{EVENT}")
    for driver in DRIVERS:
        lap = select_fastest_valid_lap(laps, driver)
        if lap is None:
            print(f"{driver}: no valid timed lap found.", file=sys.stderr)
            continue
        print(
            f"\n{driver} | Lap {lap['LapNumber']:.0f} | {lap['LapTime'].total_seconds():.3f} s"
        )
        position = read_position(lap)
        car = read_car(lap)
        if position is None:
            continue
        analysis, anomaly_report = investigate_lap(lap, position, car)
        analyses[driver] = analysis
        anomalies.append(anomaly_report)

    if not analyses:
        print("No position telemetry available for synchronization.", file=sys.stderr)
        return 1
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "speed_anomalies.csv"
    report = pd.concat(anomalies, ignore_index=True)
    report.to_csv(csv_path, index=False, float_format="%.6f")
    save_plots(analyses, output_dir)
    print(f"\nSaved five plots and {len(report)} anomaly rows to {output_dir}")
    complete = sum(
        (analysis.intervals["status"] == "compared").any()
        for analysis in analyses.values()
    )
    return int(complete != len(DRIVERS))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Only use the local FastF1 cache."
    )
    raise SystemExit(main(offline=parser.parse_args().offline))
