"""Inspect raw position samples and their geometry from qualifying at Spa in 2024."""

import sys
from argparse import ArgumentParser
from pathlib import Path

import fastf1
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.core import Lap, Laps
from fastf1.exceptions import DataNotLoadedError, F1AuthError, NoLapDataError
from matplotlib.collections import LineCollection
from matplotlib.ticker import LinearLocator, StrMethodFormatter

from formularl.geometry import (
    Alignment,
    ArcLengths,
    FloatArray,
    align_to_reference,
    arc_lengths,
    closure_distances,
)

ROOT = Path(__file__).resolve().parents[1]
EVENT = "2024 Belgian Grand Prix qualifying"
DRIVERS = ("VER", "LEC", "HAM")
COORDINATES = ["X", "Y", "Z"]
COLORS = {"VER": "tab:blue", "LEC": "tab:red", "HAM": "tab:green"}
SPA_LENGTH_M = 7_004.0
GAP_THRESHOLDS_M = (40.0, 60.0)
SPEED_FLAG_KMH = 400.0


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


def report_position(position: pd.DataFrame, arc: ArcLengths) -> None:
    """Report quality diagnostics in the original sample order."""
    coordinates = position[COORDINATES]
    timestamps = position["SessionTime"]
    intervals = position["IntervalMillis"].iloc[1:]
    displacements = pd.Series(arc.spatial_segments)

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


def format_measurement(value: float, unit: str, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return f"unavailable ({unit})"
    number = f"{value:+.3f}" if signed else f"{value:.3f}"
    return f"{number} {unit}"


def report_arc_length(label: str, segments: FloatArray, total: float) -> None:
    if not np.isfinite(total):
        known = segments[np.isfinite(segments)]
        subtotal = format_measurement(known.sum() if known.size else np.nan, "m")
        print(
            f"  Observed {label} subtotal: {subtotal} ({known.size}/{segments.size} segments)"
        )
        print(
            f"  {label.capitalize()} deviation from Spa: unavailable (incomplete arc)"
        )
        return
    difference = total - SPA_LENGTH_M
    direction = "below" if difference < 0 else "above" if difference > 0 else "at"
    percent = abs(difference) / SPA_LENGTH_M * 100
    print(
        f"  Observed {label} length: {total:.3f} m"
        f" | Spa deviation: {abs(difference):.3f} m ({percent:.3f}%, {direction} official)"
    )


def report_geometry(
    position: pd.DataFrame, arc: ArcLengths, lap_end: pd.Timedelta
) -> None:
    print("  Observed lap geometry")
    horizontal, spatial = arc.cumulative_horizontal[-1], arc.cumulative_3d[-1]
    report_arc_length("horizontal", arc.horizontal_segments, horizontal)
    report_arc_length("3D", arc.spatial_segments, spatial)
    elevation = spatial - horizontal
    elevation_pct = elevation / spatial * 100 if spatial > 0 else np.nan
    print(
        f"  Length added by elevation: {format_measurement(elevation, 'm')}"
        f" ({format_measurement(elevation_pct, '%')} of observed 3D length)"
    )

    xy_gap, z_delta, xyz_gap = closure_distances(position[COORDINATES].to_numpy())
    start_offset = position["ElapsedSeconds"].iloc[0]
    end_offset = (position["SessionTime"].iloc[-1] - lap_end).total_seconds()
    print(
        f"  Endpoint separation: {format_measurement(xy_gap, 'm')} horizontal"
        f" | {format_measurement(z_delta, 'm', signed=True)} end-minus-start Z"
        f" | {format_measurement(xyz_gap, 'm')} in 3D"
    )
    print(
        f"  Sample minus recorded lap boundary: {format_measurement(start_offset, 's', signed=True)} at start"
        f" | {format_measurement(end_offset, 's', signed=True)} at end"
    )
    print(
        f"  Estimated closed polyline: {format_measurement(horizontal + xy_gap, 'm')} horizontal"
        f" | {format_measurement(spatial + xyz_gap, 'm')} in 3D"
    )
    report_segments(position, arc)


def report_segments(position: pd.DataFrame, arc: ArcLengths) -> None:
    lengths = arc.spatial_segments
    valid = lengths[np.isfinite(lengths)]
    if valid.size:
        quantiles = format_quantiles(
            pd.Series(valid), (0, 0.25, 0.5, 0.75, 0.95, 1), "m"
        )
        print(f"  3D segments (min / p25 / median / p75 / p95 / max): {quantiles}")
        print(
            f"  3D segment mean / population SD: {valid.mean():.3f} m / {valid.std():.3f} m"
        )
    else:
        print("  3D segment statistics: unavailable (m)")

    intervals = position["IntervalMillis"].to_numpy()[1:]
    speeds = np.divide(
        lengths * 3_600,
        intervals,
        out=np.full_like(lengths, np.nan),
        where=np.isfinite(intervals) & (intervals > 0),
    )
    smaller, larger = GAP_THRESHOLDS_M
    large_gap = lengths > smaller
    fast = speeds > SPEED_FLAG_KMH
    print(
        f"  3D gaps >{smaller:g} m / >{larger:g} m: "
        f"{np.count_nonzero(large_gap)} / {np.count_nonzero(lengths > larger)} segments"
    )
    print(
        f"  Implied speeds >{SPEED_FLAG_KMH:g} km/h: {np.count_nonzero(fast)} segments"
        f" | unavailable: {np.count_nonzero(~np.isfinite(speeds))} segments"
    )
    for i in np.flatnonzero(large_gap | fast):
        end_arc = format_measurement(arc.cumulative_3d[i + 1], "m")
        interval = format_measurement(intervals[i], "ms")
        speed = format_measurement(speeds[i], "km/h")
        flag = " (potentially implausible)" if fast[i] else ""
        print(
            f"    Segment {i + 1}: s-end {end_arc} | length {lengths[i]:.3f} m"
            f" | interval {interval} | implied speed {speed}{flag}"
        )


def report_alignment(driver: str, alignment: Alignment) -> None:
    matched = alignment.matched
    print(
        f"\n{driver} vs VER | {np.count_nonzero(matched)}/{len(matched)} matched samples"
    )
    print(
        f"  Horizontally degenerate VER segments skipped: {alignment.degenerate_reference_segments}"
    )
    statuses, counts = np.unique(alignment.status[~matched], return_counts=True)
    for status, count in zip(statuses, counts, strict=True):
        print(f"  Samples excluded from error statistics ({status}): {count}")
    for first, second in alignment.progress_breaks:
        print(
            f"  Progress break at samples {first + 1} to {second + 1}: "
            f"projected 3D s {alignment.reference_arc[first]:.3f} m"
            f" to {alignment.reference_arc[second]:.3f} m"
        )
    if not matched.any():
        print("  Alignment errors: unavailable (m)")
        return
    ref_arc = alignment.reference_arc[matched]
    print(
        f"  Matched reference 3D arc span: {ref_arc.min():.3f} to {ref_arc.max():.3f} m"
    )
    horizontal = alignment.horizontal_error[matched]
    vertical = alignment.vertical_error[matched]
    for label, errors in (
        ("Horizontal", horizontal),
        ("Absolute vertical", np.abs(vertical)),
        ("3D", np.hypot(horizontal, vertical)),
    ):
        print(
            f"  {label} error (mean / median / RMSE / p95 / max):"
            f" {errors.mean():.3f} m / {np.median(errors):.3f} m"
            f" / {np.sqrt(np.mean(errors**2)):.3f} m"
            f" / {np.percentile(errors, 95):.3f} m / {errors.max():.3f} m"
        )
    print(f"  Signed vertical bias (driver minus VER): {vertical.mean():+.3f} m")


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
    space.set_title("Raw 3D trajectories")
    space.set_xlabel("X (m)", labelpad=13)
    space.set_ylabel("Y (m)", labelpad=13)
    space.set_zlabel("Z (m)", labelpad=10)
    limits = (space.get_xlim3d(), space.get_ylim3d(), space.get_zlim3d())
    space.set_box_aspect([high - low for low, high in limits])
    space.set_proj_type("ortho")
    space.zaxis.set_major_locator(LinearLocator(2))
    space.zaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    space.tick_params(axis="z", labelsize=9)
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


def save_geometry_plots(
    positions: dict[str, pd.DataFrame],
    arcs: dict[str, ArcLengths],
    alignments: dict[str, Alignment],
    output_dir: Path,
) -> None:
    elevation_figure, elevation = plt.subplots(figsize=(10, 5), layout="constrained")
    segment_figure, segments = plt.subplots(figsize=(10, 5), layout="constrained")
    gap_figure, gaps = plt.subplots(figsize=(9, 7), layout="constrained")
    deviation_figure, (horizontal, vertical) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7), layout="constrained"
    )
    smaller, larger = GAP_THRESHOLDS_M
    for driver, position in positions.items():
        arc = arcs[driver]
        color = COLORS[driver]
        xy = position[["X", "Y"]].to_numpy()
        elevation.plot(arc.cumulative_3d, position["Z"], color=color, label=driver)
        segments.scatter(
            arc.cumulative_3d[1:], arc.spatial_segments, s=8, color=color, label=driver
        )
        gaps.plot(xy[:, 0], xy[:, 1], color=color, label=driver, alpha=0.65)
        chords = np.stack((xy[:-1], xy[1:]), axis=1)
        lengths = arc.spatial_segments
        for mask, gap_color, width in (
            ((lengths > smaller) & (lengths <= larger), "darkorange", 2.5),
            (lengths > larger, "crimson", 3.5),
        ):
            gaps.add_collection(
                LineCollection(chords[mask], colors=gap_color, linewidths=width)
            )
    for limit, color in zip(GAP_THRESHOLDS_M, ("darkorange", "crimson"), strict=True):
        segments.axhline(
            limit, color=color, linestyle="--", label=f"{limit:g} m threshold"
        )
    gaps.plot(
        [],
        [],
        color="darkorange",
        linewidth=2.5,
        label=f"{smaller:g}–{larger:g} m gaps",
    )
    gaps.plot([], [], color="crimson", linewidth=3.5, label=f">{larger:g} m gaps")
    for driver, alignment in alignments.items():
        matched = alignment.matched
        if not matched.any():
            continue
        s = alignment.reference_arc[matched]
        horizontal.scatter(
            s,
            alignment.horizontal_error[matched],
            s=9,
            color=COLORS[driver],
            label=driver,
        )
        vertical.scatter(
            s,
            alignment.vertical_error[matched],
            s=9,
            color=COLORS[driver],
            label=driver,
        )
    vertical.axhline(0, color="0.5", linewidth=0.8)
    horizontal.set(ylabel="Horizontal error (m)", title="Horizontal deviation from VER")
    vertical.set(
        xlabel="VER cumulative 3D arc length (m)",
        ylabel="Driver minus VER Z (m)",
        title="Signed vertical deviation from VER",
    )
    if horizontal.get_legend_handles_labels()[0]:
        horizontal.legend()
        vertical.legend()
    else:
        horizontal.text(
            0.5, 0.5, "No eligible matches", transform=horizontal.transAxes, ha="center"
        )

    elevation.set(
        xlabel="Cumulative 3D arc length (m)",
        ylabel="Z (m)",
        title="Lap elevation by distance",
    )
    segments.set(
        xlabel="Cumulative 3D arc length at segment end (m)",
        ylabel="3D segment length (m)",
        title="Spatial sampling by distance",
    )
    gaps.set(xlabel="X (m)", ylabel="Y (m)", title="Large spatial sampling gaps")
    gaps.set_aspect("equal", adjustable="box")
    for axes in (elevation, segments, gaps):
        axes.legend()
    for figure, filename in (
        (elevation_figure, "elevation_vs_arc_length.png"),
        (deviation_figure, "aligned_deviations.png"),
        (segment_figure, "segment_lengths.png"),
        (gap_figure, "spatial_gaps_xy.png"),
    ):
        figure.suptitle(EVENT)
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def main(*, offline: bool = False) -> int:
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache_dir))
    if offline:
        fastf1.Cache.offline_mode(enabled=True)

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
    arcs: dict[str, ArcLengths] = {}
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
            arc = arc_lengths(position[COORDINATES].to_numpy())
            report_position(position, arc)
            report_geometry(position, arc, lap["Time"])
            positions[driver] = position
            arcs[driver] = arc

    if positions:
        alignments: dict[str, Alignment] = {}
        if "VER" in positions:
            reference = positions["VER"][COORDINATES].to_numpy()
            for driver, position in positions.items():
                if driver != "VER":
                    alignment = align_to_reference(
                        reference,
                        arcs["VER"],
                        position[COORDINATES].to_numpy(),
                        arcs[driver],
                    )
                    report_alignment(driver, alignment)
                    alignments[driver] = alignment
        else:
            print(
                "Cross-driver alignment unavailable because VER has no position telemetry."
            )
        output_dir = ROOT / "outputs"
        save_plots(positions, output_dir)
        save_geometry_plots(positions, arcs, alignments, output_dir)
        print(f"\nSaved eight plots to {output_dir}")
    else:
        print("No position telemetry available for plotting.", file=sys.stderr)
    return int(len(positions) != len(DRIVERS))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Only use the local FastF1 cache."
    )
    raise SystemExit(main(offline=parser.parse_args().offline))
