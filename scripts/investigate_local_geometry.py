"""Validate elevation, local derivatives and speed-anomaly recovery at Spa."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.core import Lap
from matplotlib.axes import Axes

from formularl.geometry import (
    GAP_THRESHOLDS_M,
    SHORT_CHORD_M,
    ArcLengths,
    FloatArray,
    align_to_reference,
    arc_lengths,
    horizontal_curvature,
    inclination_and_gradient,
    interpolate_arc,
    vertical_curvature,
    window_gradient,
)
from formularl.synchronization import (
    KPH_PER_MPS,
    RESIDUAL_THRESHOLD_M,
    compare_intervals,
    persistent_residual,
    recovery_residuals,
    time_at_reference_arc,
)
from formularl.telemetry import (
    COORDINATES,
    DRIVERS,
    EVENT,
    SPEED_FLAG_KMH,
    load_laps,
    read_car,
    read_position,
    select_fastest_valid_lap,
    time_seconds,
)

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"VER": "tab:blue", "LEC": "tab:red", "HAM": "tab:green"}
WINDOWS_M = (10, 25, 50)
PUBLISHED_RANGE_M = 102.2
SPARSE_CHORD_M = GAP_THRESHOLDS_M[0]
ELEVATION_INCREMENT_M = 0.1
RADIUS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LapGeometry:
    xyz: FloatArray
    arc: ArcLengths
    time_s: FloatArray
    car_time_s: FloatArray
    car_speed_mps: FloatArray
    intervals: pd.DataFrame
    inclination_deg: FloatArray
    gradients: pd.DataFrame
    horizontal_k: FloatArray
    vertical_k: FloatArray

    @property
    def centers_m(self) -> FloatArray:
        s = self.arc.cumulative_3d
        return (s[:-1] + s[1:]) / 2


def measure_lap(lap: Lap, position: pd.DataFrame) -> LapGeometry:
    xyz = position[COORDINATES].to_numpy()
    arc = arc_lengths(xyz)
    time_s = time_seconds(position)
    car = read_car(lap)
    car_time = time_seconds(car) if car is not None else np.empty(0)
    car_speed = (
        car["Speed"].to_numpy(dtype=float, na_value=np.nan) / KPH_PER_MPS
        if car is not None
        else np.empty(0)
    )
    intervals = compare_intervals(
        time_s, arc, car_time, car_speed, lap["LapStartTime"].total_seconds()
    )
    angles, native = inclination_and_gradient(
        arc.horizontal_segments, np.diff(xyz[:, 2])
    )
    s = arc.cumulative_3d
    centers = (s[:-1] + s[1:]) / 2
    gradients = pd.DataFrame({"Native": native})
    for width in WINDOWS_M:
        gradients[f"{width} m"] = window_gradient(arc, xyz[:, 2], centers, width)
    horizontal = horizontal_curvature(xyz[:, :2])
    vertical = vertical_curvature(angles, arc.spatial_segments)
    return LapGeometry(
        xyz,
        arc,
        time_s,
        car_time,
        car_speed,
        intervals,
        angles,
        gradients,
        horizontal,
        vertical,
    )


def window_residuals(
    lap: LapGeometry, centers_s: FloatArray
) -> tuple[FloatArray, FloatArray]:
    return recovery_residuals(
        lap.time_s, lap.arc.cumulative_3d, lap.car_time_s, lap.car_speed_mps, centers_s
    )


def matched_arc(lap: LapGeometry, reference: LapGeometry) -> FloatArray:
    if lap is reference:
        return lap.arc.cumulative_3d
    alignment = align_to_reference(reference.xyz, reference.arc, lap.xyz, lap.arc)
    return np.where(alignment.matched, alignment.reference_arc, np.nan)


def projected_centers(sample_arc: FloatArray) -> FloatArray:
    left, right = sample_arc[:-1], sample_arc[1:]
    return np.where(right > left, (left + right) / 2, np.nan)


def recover_anomalies(
    driver: str,
    lap: LapGeometry,
    laps: dict[str, LapGeometry],
    reference_arcs: dict[str, FloatArray],
) -> pd.DataFrame:
    native = lap.intervals["signed_distance_discrepancy_m"]
    anomalous = lap.intervals["position_speed_kmh"] > SPEED_FLAG_KMH
    report = lap.intervals.loc[
        anomalous,
        [
            "interval_number",
            "start_session_s",
            "end_session_s",
            "end_3d_arc_m",
            "position_speed_kmh",
            "signed_distance_discrepancy_m",
            "status",
        ],
    ].copy()
    report.insert(0, "driver", driver)
    report["reference_3d_m"] = projected_centers(reference_arcs[driver])[anomalous]
    report["with_previous_m"] = (native.shift(1) + native)[anomalous]
    report["with_next_m"] = (native + native.shift(-1))[anomalous]
    report["with_both_neighbors_m"] = (native.shift(1) + native + native.shift(-1))[
        anomalous
    ]
    center_s = ((report["start_session_s"] + report["end_session_s"]) / 2).to_numpy()
    one, two = window_residuals(lap, center_s)
    report["residual_1s_m"], report["residual_2s_m"] = one, two
    supported = np.isfinite(one) & np.isfinite(two)
    persistent = persistent_residual(one, two)
    report["persistent"] = pd.array(
        np.where(supported, persistent, None), dtype="boolean"
    )
    print(f"  Speed anomalies: {len(report)} over {SPEED_FLAG_KMH:g} km/h")
    for name in (
        "signed_distance_discrepancy_m",
        "with_previous_m",
        "with_next_m",
        "with_both_neighbors_m",
        "residual_1s_m",
        "residual_2s_m",
    ):
        absolute = report[name].abs().dropna()
        within = (absolute <= RESIDUAL_THRESHOLD_M).sum()
        median = f"{absolute.median():.3f} m" if len(absolute) else "unavailable"
        print(
            f"    {name}: median absolute {median}, within {RESIDUAL_THRESHOLD_M:g} m {within}/{len(absolute)}"
        )
    print(
        f"  Persistent residuals: {persistent.sum()}/{supported.sum()} with both windows"
    )
    locations = report["reference_3d_m"].to_numpy()
    for other, comparison in laps.items():
        if other == driver:
            continue
        mapped = time_at_reference_arc(
            comparison.time_s, reference_arcs[other], locations[persistent]
        )
        peer_one, peer_two = window_residuals(comparison, mapped)
        peer_persistent = persistent_residual(peer_one, peer_two)
        peer_supported = np.isfinite(peer_one) & np.isfinite(peer_two)
        same_sign = peer_persistent & (np.sign(peer_two) == np.sign(two[persistent]))
        report.loc[persistent, f"{other}_matched_session_s"] = mapped
        report.loc[persistent, f"{other}_residual_1s_m"] = peer_one
        report.loc[persistent, f"{other}_residual_2s_m"] = peer_two
        print(
            f"    {other} at those VER arc locations: {peer_supported.sum()}/{persistent.sum()} supported,"
            f" {same_sign.sum()} persistent in the same direction,"
            f" {(peer_persistent & ~same_sign).sum()} persistent in the opposite direction"
        )
        for location in locations[persistent][same_sign]:
            print(f"      {other} same-direction recurrence at VER s {location:.1f} m")
    for s, r1, r2 in zip(
        locations[persistent], one[persistent], two[persistent], strict=True
    ):
        location = f"{s:.1f} m" if np.isfinite(s) else "unavailable"
        print(f"    Persistent at VER s {location}: 1 s {r1:+.2f} m, 2 s {r2:+.2f} m")
    return report


def report_elevation(lap: LapGeometry) -> None:
    z = lap.xyz[:, 2]
    known = z[np.isfinite(z)]
    if known.size:
        span = np.ptp(known)
        print(
            f"  Elevation {known.min():.1f} to {known.max():.1f} m"
            f" | range {span:.1f} m | minus published {PUBLISHED_RANGE_M:.1f} m: {span - PUBLISHED_RANGE_M:+.1f} m"
        )
    else:
        print("  Elevation extrema unavailable.")
    rise = np.diff(z)
    finite_rises = np.isfinite(rise)
    gain = rise[finite_rises & (rise > 0)].sum()
    loss = rise[finite_rises & (rise < 0)].sum()
    print(
        f"  Elevation change over {finite_rises.sum()}/{len(rise)} measured chords"
        f" | gain +{gain:.1f} m | descent {loss:.1f} m | endpoint net {z[-1] - z[0]:+.1f} m"
    )
    angles = lap.inclination_deg[np.isfinite(lap.inclination_deg)]
    if angles.size:
        print(f"  Inclination extremes: {angles.min():+.2f}° to {angles.max():+.2f}°")
    zero_run = lap.arc.horizontal_segments == 0
    print(
        f"  Degenerate segments: {(lap.arc.spatial_segments == 0).sum()} zero 3D length,"
        f" {(zero_run & (lap.arc.spatial_segments > 0)).sum()} vertical-only"
    )
    for width, (scale, values) in zip(
        (None, *WINDOWS_M), lap.gradients.items(), strict=True
    ):
        grade = values.dropna()
        if grade.empty:
            print(f"    {scale} grade: unavailable.")
            continue
        minimum, maximum = grade.idxmin(), grade.idxmax()
        print(
            f"    {scale} grade ({len(grade)}/{len(values)} centers)"
            f" | downhill {grade[minimum]:+.2f}% at {lap.centers_m[minimum]:.1f} m"
            f" | uphill {grade[maximum]:+.2f}% at {lap.centers_m[maximum]:.1f} m"
            f" | median / p95 abs {grade.abs().median():.2f}% / {grade.abs().quantile(0.95):.2f}%"
        )
        if width is not None:
            inside_one_chord = np.count_nonzero(
                lap.arc.spatial_segments[grade.index] >= width
            )
            print(
                f"      Entire window within one native chord: {inside_one_chord}/{len(grade)}"
            )


def native_gradient_at(lap: LapGeometry, centers_m: FloatArray) -> FloatArray:
    ends = lap.arc.cumulative_3d[1:]
    indices = np.searchsorted(ends, centers_m, side="right")
    eligible = (indices < len(ends)) & np.isfinite(centers_m) & (centers_m >= 0)
    rows = np.flatnonzero(eligible)
    rows = rows[np.isfinite(ends[indices[rows]])]
    result = np.full(centers_m.shape, np.nan)
    result[rows] = lap.gradients["Native"].to_numpy()[indices[rows]]
    return result


def compare_gradients(
    driver: str, lap: LapGeometry, reference: LapGeometry, sample_arc: FloatArray
) -> pd.DataFrame:
    centers = projected_centers(sample_arc)
    reference_gradient = {"Native": native_gradient_at(reference, centers)}
    for width in WINDOWS_M:
        reference_gradient[f"{width} m"] = window_gradient(
            reference.arc, reference.xyz[:, 2], centers, width
        )
    difference = lap.gradients - pd.DataFrame(reference_gradient)
    difference.insert(0, "reference_3d_m", centers)
    common = difference.dropna()
    print(f"  {driver} vs VER on {len(common)}/{len(centers)} common segment centers")
    for label, values in common.drop(columns="reference_3d_m").items():
        absolute = values.abs()
        if not absolute.empty:
            print(
                f"    {label} gradient difference (percentage points)"
                f" | MAE {absolute.mean():.3f} | p95 {absolute.quantile(0.95):.3f}"
                f" | max {absolute.max():.3f}"
            )
    return common


def report_curvature(lap: LapGeometry) -> None:
    s = lap.arc.cumulative_3d[1:-1]
    xy = lap.arc.horizontal_segments
    spatial = lap.arc.spatial_segments
    dz = np.diff(lap.xyz[:, 2])
    short = np.minimum(xy[:-1], xy[1:]) < SHORT_CHORD_M
    sparse = np.maximum(spatial[:-1], spatial[1:]) > SPARSE_CHORD_M
    small_z = (
        np.minimum(np.abs(dz[:-1]), np.abs(dz[1:])) <= ELEVATION_INCREMENT_M + 1e-9
    )
    all_vertices = np.ones(s.shape, dtype=bool)
    for name, curvature in (
        ("Horizontal", lap.horizontal_k),
        ("Vertical", lap.vertical_k),
    ):
        finite = np.isfinite(curvature)
        print(f"  {name} curvature: {finite.sum()}/{len(curvature)} defined vertices")
        if name == "Horizontal":
            radii = 1 / np.abs(
                curvature[finite & (np.abs(curvature) > RADIUS_TOLERANCE)]
            )
            if radii.size:
                print(
                    f"    Equivalent radii: minimum {radii.min():.2f} m, median {np.median(radii):.2f} m"
                    f" ({len(radii)}/{finite.sum()} above the curvature tolerance)"
                )
        for label, mask in (
            ("all", all_vertices),
            (f"short (adjacent XY <{SHORT_CHORD_M:g} m)", short),
            (f"sparse (adjacent 3D >{SPARSE_CHORD_M:g} m)", sparse),
            (f"adjacent |dz| <= {ELEVATION_INCREMENT_M:g} m", small_z),
            ("neither short nor sparse", ~short & ~sparse),
        ):
            absolute = np.abs(curvature[finite & mask])
            if absolute.size:
                print(
                    f"    {label}: n={len(absolute)}, |curvature| p95 {np.percentile(absolute, 95):.5f} m^-1"
                    f", max {absolute.max():.5f} m^-1"
                )
        largest = np.flatnonzero(finite)
        largest = largest[np.argsort(np.abs(curvature[largest]))[::-1][:3]]
        for i in largest:
            radius = (
                f", radius {1 / abs(curvature[i]):.2f} m"
                if name == "Horizontal" and abs(curvature[i]) > RADIUS_TOLERANCE
                else ""
            )
            print(
                f"    At s {s[i]:.1f} m: {curvature[i]:+.5f} m^-1{radius}"
                f" | XY chords {xy[i]:.2f}, {xy[i + 1]:.2f} m"
                f" | rises {dz[i]:+.1f}, {dz[i + 1]:+.1f} m"
            )


def plot_steepest(axes: Axes, driver: str, lap: LapGeometry) -> None:
    width = WINDOWS_M[-1]
    gradients = lap.gradients[f"{width} m"].dropna()
    x, y = lap.xyz[:, :2].T
    s = lap.arc.cumulative_3d
    axes.plot(x, y, color="0.55", label=driver)
    if not gradients.empty:
        for i, name, color in (
            (gradients.idxmax(), "Uphill", "tab:red"),
            (gradients.idxmin(), "Downhill", "tab:blue"),
        ):
            center = lap.centers_m[i]
            left, right = center - width / 2, center + width / 2
            knots = np.r_[left, s[(s > left) & (s < right)], right]
            axes.plot(
                interpolate_arc(s, x, knots),
                interpolate_arc(s, y, knots),
                color=color,
                linewidth=4,
                label=f"{name} {gradients[i]:+.1f}% at {center:,.0f} m",
            )
    axes.set(
        xlabel="X (m)", ylabel="Y (m)", title=f"{driver} steepest {width} m windows"
    )
    axes.set_aspect("equal", adjustable="box")
    axes.legend()


def save_plots(
    laps: dict[str, LapGeometry],
    comparisons: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    matplotlib.use("Agg")
    gradient_fig, gradient = plt.subplots(figsize=(11, 5), layout="constrained")
    window_fig, (extremes, tails) = plt.subplots(
        1, 2, figsize=(11, 5), layout="constrained"
    )
    aligned_fig, aligned_axes = plt.subplots(
        2, 2, figsize=(12, 7), sharex=True, sharey=True, layout="constrained"
    )
    horizontal_fig, horizontal = plt.subplots(figsize=(11, 5), layout="constrained")
    vertical_fig, vertical = plt.subplots(figsize=(11, 5), layout="constrained")
    steep_fig, steep = plt.subplots(figsize=(9, 7), layout="constrained")

    scales = ["Native", *(f"{width} m" for width in WINDOWS_M)]
    for driver, lap in laps.items():
        color = COLORS[driver]
        gradient.scatter(
            lap.centers_m, lap.gradients["Native"], s=9, color=color, label=driver
        )
        extremes.plot(
            scales, lap.gradients.max(), marker="o", color=color, label=driver
        )
        extremes.plot(
            scales, lap.gradients.min(), marker="v", linestyle="--", color=color
        )
        tails.plot(
            scales,
            lap.gradients.abs().quantile(0.95),
            marker="o",
            color=color,
            label=driver,
        )
        s = lap.arc.cumulative_3d[1:-1]
        horizontal.scatter(s, lap.horizontal_k, s=9, color=color, label=driver)
        vertical.scatter(s, lap.vertical_k, s=9, color=color, label=driver)

    for axes, scale in zip(aligned_axes.flat, scales, strict=True):
        for driver, differences in comparisons.items():
            axes.scatter(
                differences["reference_3d_m"],
                differences[scale],
                s=9,
                color=COLORS[driver],
                label=driver,
            )
        axes.axhline(0, color="0.5", linewidth=0.8)
        axes.set_title(scale)
    aligned_fig.supxlabel("VER cumulative 3D arc (m)")
    aligned_fig.supylabel("Driver minus VER (percentage points)")
    if comparisons:
        aligned_axes[0, 0].legend()
    else:
        aligned_axes[0, 0].text(
            0.5,
            0.5,
            "VER comparison unavailable",
            transform=aligned_axes[0, 0].transAxes,
            ha="center",
        )

    gradient.set(
        title="Native segment gradients",
        xlabel="Cumulative 3D arc (m)",
        ylabel="Gradient (%)",
    )
    extremes.set(
        title="Uphill maxima and downhill minima",
        xlabel="Spatial window",
        ylabel="Gradient (%)",
    )
    tails.set(
        title="95th percentile absolute gradient",
        xlabel="Spatial window",
        ylabel="Gradient (%)",
    )
    horizontal.set(
        title="Signed XY circumcircle curvature",
        xlabel="Cumulative 3D arc (m)",
        ylabel="Curvature (m⁻¹)",
    )
    vertical.set(
        title="Inclination change per metre",
        xlabel="Cumulative 3D arc (m)",
        ylabel="Curvature (m⁻¹)",
    )
    for axes in (gradient, extremes, tails, horizontal, vertical):
        axes.legend()
    map_driver = "VER" if "VER" in laps else next(iter(laps))
    plot_steepest(steep, map_driver, laps[map_driver])
    for figure, name, title in (
        (gradient_fig, "gradient_vs_arc_length.png", "Native gradient"),
        (
            window_fig,
            "gradient_window_comparison.png",
            "Gradient sensitivity to spatial window",
        ),
        (
            aligned_fig,
            "aligned_gradient_differences.png",
            "Gradient difference at corresponding locations",
        ),
        (horizontal_fig, "horizontal_curvature.png", "Horizontal curvature"),
        (vertical_fig, "vertical_curvature.png", "Vertical curvature"),
        (steep_fig, "steep_sections_xy.png", "Steepest uphill and downhill sections"),
    ):
        figure.suptitle(f"{EVENT}\n{title}")
        figure.savefig(output_dir / name, dpi=150)
        plt.close(figure)


def main(*, offline: bool = False) -> int:
    laps = load_laps(ROOT / "cache", offline=offline)
    if laps is None:
        return 1
    measured: dict[str, LapGeometry] = {}
    for driver in DRIVERS:
        lap = select_fastest_valid_lap(laps, driver)
        if lap is None:
            print(f"{driver}: no valid timed lap found.")
            continue
        position = read_position(lap)
        if position is not None:
            measured[driver] = measure_lap(lap, position)
    if not measured:
        print("No local geometry could be analyzed.")
        return 1

    reference = measured.get("VER")
    reference_arcs = {
        driver: matched_arc(lap, reference)
        if reference is not None
        else np.full(len(lap.xyz), np.nan)
        for driver, lap in measured.items()
    }
    comparisons: dict[str, pd.DataFrame] = {}
    anomalies: list[pd.DataFrame] = []
    print(f"\n{EVENT}")
    for driver, lap in measured.items():
        print(f"\n{driver}")
        report_elevation(lap)
        report_curvature(lap)
        anomalies.append(recover_anomalies(driver, lap, measured, reference_arcs))
        if reference is not None and driver != "VER":
            comparisons[driver] = compare_gradients(
                driver, lap, reference, reference_arcs[driver]
            )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_plots(measured, comparisons, output_dir)
    report = pd.concat(anomalies, ignore_index=True)
    report.to_csv(output_dir / "anomaly_recovery.csv", index=False, float_format="%.6f")
    print(f"\nSaved six plots and {len(report)} anomaly recovery rows to {output_dir}")
    return int(
        len(measured) != len(DRIVERS)
        or any(lap.car_time_s.size < 2 for lap in measured.values())
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Only use the local FastF1 cache."
    )
    raise SystemExit(main(offline=parser.parse_args().offline))
