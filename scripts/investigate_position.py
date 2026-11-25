"""Inspect raw position samples from qualifying at Spa in 2024."""

import sys
from pathlib import Path

import fastf1
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.core import Lap, Laps
from fastf1.exceptions import DataNotLoadedError, F1AuthError, NoLapDataError

ROOT = Path(__file__).resolve().parents[1]
EVENT = "2024 Belgian Grand Prix qualifying"
DRIVERS = ("VER", "LEC", "HAM")
COORDINATES = ["X", "Y", "Z"]
COLORS = {"VER": "tab:blue", "LEC": "tab:red", "HAM": "tab:green"}


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


def read_position(lap: Lap) -> pd.DataFrame | None:
    """Keep FastF1's original samples and build a separate metric table."""
    driver = lap["Driver"]
    try:
        raw = lap.get_pos_data(pad=0, interpolate_edges=False)
    except (DataNotLoadedError, KeyError) as error:
        print(f"{driver}: position telemetry unavailable ({error}).", file=sys.stderr)
        return None
    if raw.empty:
        print(f"{driver}: position telemetry is empty (0 samples).", file=sys.stderr)
        return None

    missing_columns = {*COORDINATES, "SessionTime"}.difference(raw.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        print(
            f"{driver}: position telemetry is missing columns {names}.", file=sys.stderr
        )
        return None

    timestamps = raw["SessionTime"]
    position = pd.DataFrame(raw[COORDINATES], dtype=float) / 10.0
    position["SessionTime"] = timestamps
    position["ElapsedSeconds"] = (timestamps - lap["LapStartTime"]).dt.total_seconds()
    position["IntervalMillis"] = timestamps.diff().dt.total_seconds() * 1_000
    return position


def format_quantiles(values: pd.Series, quantiles: tuple[float, ...], unit: str) -> str:
    """Format the requested summaries of the available values."""
    if not values.notna().any():
        return f"unavailable ({unit})"
    return " / ".join(f"{values.quantile(q):.3f} {unit}" for q in quantiles)


def report_position(position: pd.DataFrame) -> None:
    """Report quality diagnostics in the original sample order."""
    coordinates = position[COORDINATES]
    timestamps = position["SessionTime"]
    intervals = position["IntervalMillis"].iloc[1:]
    displacements = pd.Series(np.linalg.norm(coordinates.diff().to_numpy(), axis=1))
    displacements = displacements.iloc[1:]

    print(f"  Position samples: {len(position)} samples")
    print(
        "  Sampling interval (min / median / max): "
        f"{format_quantiles(intervals, (0, 0.5, 1), 'ms')}"
    )
    missing = ", ".join(
        f"{axis}={coordinates[axis].isna().sum()}" for axis in COORDINATES
    )
    print(f"  Missing coordinate values: {missing} (values)")
    print(f"  Missing timestamps: {timestamps.isna().sum()} samples")
    print(f"  Duplicate timestamps: {timestamps.dropna().duplicated().sum()} samples")
    print(f"  Non-positive sampling intervals: {intervals.le(0).sum()} intervals")
    print(f"  Unavailable sampling intervals: {intervals.isna().sum()} pairs")
    for axis in COORDINATES:
        bounds = format_quantiles(coordinates[axis], (0, 1), "m")
        print(f"  {axis} coordinate (min / max): {bounds}")
    elevation_range = coordinates["Z"].max() - coordinates["Z"].min()
    elevation = (
        f"{elevation_range:.3f} m" if pd.notna(elevation_range) else "unavailable (m)"
    )
    print(f"  Elevation range: {elevation}")
    print(
        "  Consecutive 3D displacement (median / p95 / max): "
        f"{format_quantiles(displacements, (0.5, 0.95, 1), 'm')}"
    )
    print(f"  Unavailable 3D displacements: {displacements.isna().sum()} pairs")


def save_plots(positions: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save overlays and a common-bin histogram for the available drivers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    xy_figure, xy = plt.subplots(figsize=(9, 7), layout="constrained")
    space_figure = plt.figure(figsize=(9, 7), layout="constrained")
    space = space_figure.add_subplot(projection="3d")
    elevation_figure, elevation = plt.subplots(figsize=(10, 5), layout="constrained")
    intervals_figure, interval_plot = plt.subplots(
        figsize=(10, 5), layout="constrained"
    )

    intervals = {
        driver: position["IntervalMillis"].dropna().to_numpy()
        for driver, position in positions.items()
    }
    bins = np.histogram_bin_edges(np.concatenate(list(intervals.values())), bins=30)
    for driver, position in positions.items():
        x, y, z = (position[axis] for axis in COORDINATES)
        style = {
            "label": driver,
            "color": COLORS[driver],
            "marker": ".",
            "markersize": 2,
        }
        xy.plot(x, y, **style)
        space.plot(x, y, z, **style)
        elevation.plot(position["ElapsedSeconds"], z, **style)
        label = driver if intervals[driver].size else f"{driver} (no intervals)"
        interval_plot.hist(
            intervals[driver],
            bins=bins,
            histtype="step",
            label=label,
            color=COLORS[driver],
        )

    xy.set(xlabel="X (m)", ylabel="Y (m)", title="Raw X/Y trajectories")
    xy.set_aspect("equal", adjustable="box")
    space.set(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Raw 3D trajectories"
    )
    elevation.set(
        xlabel="Elapsed lap time (s)", ylabel="Z (m)", title="Elevation during each lap"
    )
    interval_plot.set(
        xlabel="Consecutive sampling interval (ms)",
        ylabel="Count (intervals)",
        title="Distribution of sampling intervals",
    )

    for figure, axes, filename in (
        (xy_figure, xy, "trajectory_xy.png"),
        (space_figure, space, "trajectory_3d.png"),
        (elevation_figure, elevation, "elevation_vs_time.png"),
        (intervals_figure, interval_plot, "sampling_intervals.png"),
    ):
        axes.legend()
        figure.suptitle(EVENT)
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def main() -> int:
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache_dir))

    try:
        session = fastf1.get_session(2024, "Belgian Grand Prix", "Q")
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
        return 1
    if laps.empty:
        print(f"No lap timing available for {EVENT}.", file=sys.stderr)
        return 1
    if "Deleted" not in laps.columns:
        print(
            "Lap deletion status is unavailable; race control messages are required.",
            file=sys.stderr,
        )
        return 1

    positions: dict[str, pd.DataFrame] = {}
    print(f"\n{EVENT}")
    for driver in DRIVERS:
        lap = select_fastest_valid_lap(laps, driver)
        if lap is None:
            print(f"{driver}: no valid timed lap found.", file=sys.stderr)
            continue
        lap_time = lap["LapTime"].total_seconds()
        print(f"\n{driver} | Lap {lap['LapNumber']:.0f} | {lap_time:.3f} s")
        position = read_position(lap)
        if position is not None:
            report_position(position)
            positions[driver] = position

    if positions:
        save_plots(positions, ROOT / "outputs")
        print(f"\nSaved four plots to {ROOT / 'outputs'}")
    else:
        print("No position telemetry available for plotting.", file=sys.stderr)
    return int(len(positions) != len(DRIVERS))


if __name__ == "__main__":
    raise SystemExit(main())
