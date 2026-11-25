"""Build and compare the Task 5A periodic cubic spline baselines."""

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LinearLocator
from scipy.interpolate import CubicSpline

from formularl.geometry import GAP_THRESHOLDS_M, FloatArray, interpolate_arc
from formularl.periodic import circular_delta, close_reference, wrap_arc
from formularl.reconstruction_data import IDENTITY, XYZ, load_reconstruction
from formularl.spline import (
    TANGENT_EPS,
    arc_map,
    coordinate_extrema,
    differential_geometry,
    periodic_linear,
    periodic_spline,
    project_xy,
)
from formularl.telemetry import SPA_LENGTH_M

ROOT = Path(__file__).resolve().parents[1]
TASK4 = ROOT / "outputs" / "reconstruction"
OUTPUT = TASK4 / "baselines"
SPACINGS_M = (5, 10, 20)
MAX_DONOR_GAP_M = GAP_THRESHOLDS_M[1]
PROJECTION_RADIUS_M = 50.0
GRADE = "gradient_pct"
HORIZONTAL = "horizontal_curvature_inv_m"
VERTICAL = "vertical_curvature_inv_m"
DERIVATIVES = (GRADE, HORIZONTAL, VERTICAL)


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    frames, source = load_reconstruction(ROOT / "outputs" / "datasets")
    metadata = json.loads((TASK4 / "manifest.json").read_bytes())
    if metadata["schema_version"] != 1 or source != metadata["inputs"]:
        raise ValueError(
            "Task 4 provenance differs from the current prepared reconstruction inputs."
        )
    tables = []
    for name in ("observations.csv", "reference_polyline.csv"):
        raw = (TASK4 / name).read_bytes()
        if sha256(raw).hexdigest() != metadata["table_sha256"][name]:
            raise ValueError(f"Task 4 checksum mismatch for {name}.")
        tables.append(pd.read_csv(BytesIO(raw), float_precision="round_trip"))
    observations, reference_csv = tables
    reference = close_reference(frames["VER"][XYZ].to_numpy())
    if (
        not np.isclose(
            reference.period, metadata["reference"]["period_3d_m"], atol=1e-9, rtol=0
        )
        or not np.array_equal(
            reference_csv[[f"start_{k}" for k in XYZ]], reference.vertices[:-1]
        )
        or not np.array_equal(
            reference_csv[[f"end_{k}" for k in XYZ]], reference.vertices[1:]
        )
    ):
        raise ValueError("Task 4 reference differs from the closed native VER path.")
    native = pd.concat(frames.values(), ignore_index=True)
    pd.testing.assert_frame_equal(
        observations[[*IDENTITY, *XYZ]], native[[*IDENTITY, *XYZ]]
    )
    return observations, frames, metadata


def fit_candidates(
    observations: pd.DataFrame, ver: pd.DataFrame, period: float
) -> dict:
    candidates = {
        "VER": (
            periodic_spline(ver["arc_3d_m"].to_numpy(), ver[XYZ].to_numpy(), period),
            None,
        )
    }
    valid = observations.loc[observations["correspondence_valid"]]
    drivers = tuple(observations["driver"].unique())
    for spacing in SPACINGS_M:
        grid = np.linspace(0, period, round(period / spacing), endpoint=False)
        donors = []
        for driver in drivers:
            group = valid.loc[valid["driver"] == driver]
            if group.empty:
                raise ValueError(f"{driver} has no supported donor observations.")
            donors.append(
                periodic_linear(
                    group["reference_arc_3d_m"].to_numpy(),
                    group[XYZ].to_numpy(),
                    grid,
                    period,
                    MAX_DONOR_GAP_M,
                )
            )
        values = np.stack(donors)
        count = np.isfinite(values).all(axis=2).sum(axis=0)
        if np.any(count == 0):
            raise ValueError(
                f"The {spacing} m grid has locations without donor support."
            )
        candidates[f"mean_{spacing}m"] = (
            periodic_spline(grid, np.nansum(values, axis=0) / count[:, None], period),
            count,
        )
    return candidates


def geometry_table(
    curve: CubicSpline,
) -> tuple[pd.DataFrame, float, int, tuple[float, ...]]:
    arc = arc_map(curve)
    s, q = arc.uniform()
    xyz = curve(q)
    v, a = curve(q, 1), curve(q, 2)
    data = {"arc_3d_m": s, "reference_q_m": q, "periodic_u": q / curve.x[-1]}
    for fields, values in (
        (XYZ, xyz),
        (("dx_dq", "dy_dq", "dz_dq"), v),
        (("d2x_dq2", "d2y_dq2", "d2z_dq2"), a),
    ):
        data.update(zip(fields, values.T, strict=True))
    data.update(differential_geometry(v, a))
    return pd.DataFrame(data), arc.length_m, arc.order, arc.differences


def finite(values: FloatArray) -> FloatArray:
    return values[np.isfinite(values)]


def summary(values: FloatArray) -> str:
    values = finite(values)
    if not values.size:
        return "unavailable"
    lo, median, p95, hi = np.percentile(values, [0, 50, 95, 100])
    return f"{lo:.6g} / {median:.6g} / {p95:.6g} / {hi:.6g}"


def max_abs(values: FloatArray) -> float:
    result = finite(values)
    return float(np.abs(result).max()) if result.size else np.nan


def evaluate(
    name: str,
    curve: CubicSpline,
    table: pd.DataFrame,
    length: float,
    order: int,
    convergence: tuple[float, ...],
    donors: FloatArray | None,
    observations: pd.DataFrame,
    regions: dict[str, tuple[float, float]],
    ver: pd.DataFrame,
) -> list[str]:
    period = curve.x[-1]
    anchor = observations["reference_arc_3d_m"].to_numpy()
    anchor = np.where(observations["correspondence_valid"], anchor, np.nan)
    raw = observations[XYZ].to_numpy()
    projected = project_xy(curve, raw, anchor, PROJECTION_RADIUS_M)
    delta = raw - curve(projected)
    xy, z = np.linalg.norm(delta[:, :2], axis=1), delta[:, 2]
    low, high, overshoot = coordinate_extrema(curve)
    ends = [
        np.linalg.norm(
            curve(period, n, extrapolate=False) - curve(0, n, extrapolate=False)
        )
        for n in range(3)
    ]
    q = table["reference_q_m"].to_numpy()
    span = np.searchsorted(curve.x, q, side="right") - 1
    direction = curve(curve.x[span + 1])[:, :2] - curve(curve.x[span])[:, :2]
    against = np.einsum("ij,ij->i", table[["dx_dq", "dy_dq"]], direction) < 0
    lines = [
        f"{name} | length {length:.6f} m | minus 7004 m {length - SPA_LENGTH_M:+.3f} m",
        f"  Elevation {low[2]:.3f} to {high[2]:.3f} m, range {high[2] - low[2]:.3f} m.",
        f"  Gauss order {order}, successive sums of absolute span length differences (m) {', '.join(f'{d:.3g}' for d in convergence)}.",
        f"  True arc sample increment {length / len(table):.6f} m, {len(table)} rows.",
        f"  Seam continuity position / first / second: {ends[0]:.3g} m / {ends[1]:.3g} m per q-m / {ends[2]:.3g} m per q-m squared.",
        f"  XY projection {np.isfinite(projected).sum()}/{len(projected)}, corridor-boundary hits {np.isclose(np.abs(projected - anchor), PROJECTION_RADIUS_M, atol=1e-7, rtol=0).sum()}.",
    ]
    if donors is not None:
        counts = np.bincount(donors, minlength=4)
        lines.append(
            f"  Reference grid step {period / len(donors):.4f} m; nodes with 1 / 2 / 3 donors: {counts[1]} / {counts[2]} / {counts[3]}."
        )
    for driver in (*observations["driver"].unique(), "all"):
        mask = (
            observations["driver"].eq(driver).to_numpy()
            if driver != "all"
            else np.ones(len(raw), dtype=bool)
        )
        lines.append(
            f"  {driver} in-sample residual min / median / p95 / max in m, horizontal {summary(xy[mask])}; signed Z {summary(z[mask])}; absolute Z {summary(np.abs(z[mask]))}."
        )
    for field in ("inclination_deg", *DERIVATIVES):
        values = table[field].to_numpy()
        lines.append(
            f"  {field} signed {summary(values)}; absolute p95 {np.nanpercentile(np.abs(values), 95):.6g}."
        )
    lines.extend(
        [
            f"  Minimum XYZ tangent {table['tangent_magnitude'].min():.6g}; near-zero XYZ / XY samples {(table['tangent_magnitude'] <= TANGENT_EPS).sum()} / {(np.linalg.norm(table[['dx_dq', 'dy_dq']], axis=1) <= TANGENT_EPS).sum()}.",
            f"  Maximum local XYZ knot-box overshoot (m) {', '.join(f'{d:.3f}' for d in overshoot)}; XY tangents opposing local knot chords {against.sum()}.",
        ]
    )
    native_q = np.r_[ver["arc_3d_m"], period]
    native_xyz = np.vstack([ver[XYZ], ver[XYZ].iloc[0]])
    for region, (center, radius) in regions.items():
        local = center + np.linspace(-radius, radius, int(radius * 8) + 1)
        metrics = differential_geometry(curve(local, 1), curve(local, 2))
        base = np.column_stack(
            [
                interpolate_arc(native_q, native_xyz[:, i], wrap_arc(local, period))
                for i in range(3)
            ]
        )
        excursion = np.linalg.norm(curve(local) - base, axis=1).max()
        lines.append(
            f"  {region}: max absolute grade {max_abs(metrics[GRADE]):.3f}%; max |horizontal / vertical curvature| {max_abs(metrics[HORIZONTAL]):.5f} / {max_abs(metrics[VERTICAL]):.5f} m^-1; max same-q departure from VER chords {excursion:.3f} m."
        )
    return lines


def plot_results(
    observations: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    curves: dict[str, CubicSpline],
    regions: dict[str, tuple[float, float]],
    gap: dict,
) -> None:
    matplotlib.use("Agg")
    plan, xy = plt.subplots(figsize=(8, 7), layout="constrained")
    spatial = plt.figure(figsize=(9, 7), layout="constrained")
    xyz = spatial.add_subplot(projection="3d")
    elevation, heights = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7), layout="constrained"
    )
    curvature, bends = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7), layout="constrained"
    )
    for driver, group in observations.groupby("driver", sort=False):
        xy.scatter(
            group["x_m"], group["y_m"], s=5, alpha=0.3, label=f"{driver} samples"
        )
    colors = dict(
        zip(tables, ("black", "tab:orange", "tab:blue", "tab:purple"), strict=True)
    )
    for name, data in tables.items():
        style = {"color": colors[name], "linewidth": 1, "label": name}
        closed = pd.concat([data, data.iloc[:1]], ignore_index=True)
        xy.plot(closed["x_m"], closed["y_m"], **style)
        xyz.plot(closed["x_m"], closed["y_m"], closed["z_m"], **style)
        for ax, field in zip((*heights, *bends), ("z_m", *DERIVATIVES), strict=True):
            ax.plot(data["arc_3d_m"], data[field], **style)
    xy.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        aspect="equal",
        title="Baseline curves and native observations",
    )
    xy.legend(ncols=2)
    xyz.set_xlabel("X (m)", labelpad=13)
    xyz.set_ylabel("Y (m)", labelpad=13)
    xyz.set_zlabel("Z (m)", labelpad=8)
    xyz.set_box_aspect(
        [hi - lo for lo, hi in (xyz.get_xlim3d(), xyz.get_ylim3d(), xyz.get_zlim3d())]
    )
    xyz.set_proj_type("ortho")
    xyz.zaxis.set_major_locator(LinearLocator(2))
    xyz.set_title("Baseline reconstructions on a common physical scale")
    xyz.legend()
    for ax, label in zip(
        (*heights, *bends),
        ("Elevation (m)", "Grade (%)", "Horizontal (m⁻¹)", "Vertical (m⁻¹)"),
        strict=True,
    ):
        ax.set_ylabel(label)
    for axes in (heights, bends):
        axes[0].legend(ncols=4)
        axes[-1].set_xlabel("Fitted 3D arc length (m)")
    figures = [
        (plan, "xy"),
        (spatial, "3d"),
        (elevation, "elevation_gradient"),
        (curvature, "curvature"),
    ]
    period = next(iter(curves.values())).x[-1]
    for key in ("seam", "largest_gap", "residual_region"):
        center, radius = regions[key]
        if key == "largest_gap":
            radius += 15
        local, (map_ax, *profiles) = plt.subplots(
            4, 1, figsize=(9, 11), layout="constrained"
        )
        offsets = np.linspace(-radius, radius, 401)
        q = center + offsets
        matched = (
            np.abs(circular_delta(observations["reference_arc_3d_m"] - center, period))
            <= radius
        )
        for driver, data in observations.loc[matched].groupby("driver", sort=False):
            map_ax.scatter(
                data["x_m"], data["y_m"], s=11, alpha=0.55, label=f"{driver} samples"
            )
        for name, curve in curves.items():
            points = curve(q)
            metrics = differential_geometry(curve(q, 1), curve(q, 2))
            style = {"color": colors[name], "linewidth": 1.2, "label": name}
            map_ax.plot(points[:, 0], points[:, 1], **style)
            for ax, field in zip(profiles, DERIVATIVES, strict=True):
                ax.plot(offsets, metrics[field], **style)
        if key == "largest_gap":
            half_gap = gap["maximum_gap_m"] / 2
            for ax in profiles:
                ax.axvspan(-half_gap, half_gap, color="goldenrod", alpha=0.2)
        elif key == "seam":
            for ax in profiles:
                ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
        map_ax.set(aspect="equal", xlabel="X (m)", ylabel="Y (m)")
        map_ax.legend(ncols=3, fontsize="small")
        for ax, label in zip(
            profiles,
            ("Grade (%)", "Horizontal curvature (m⁻¹)", "Vertical curvature (m⁻¹)"),
            strict=True,
        ):
            ax.set(ylabel=label, xlim=(-radius, radius))
        profiles[-1].set_xlabel("Offset in reference parameter q (m)")
        local.suptitle(
            f"{key.replace('_', ' ').capitalize()} | reference q = {center:.2f} m"
        )
        figures.append((local, key))
    for figure, name in figures:
        figure.savefig(OUTPUT / f"{name}.png", dpi=150)
        plt.close(figure)


def main() -> None:
    observations, frames, manifest = load_inputs()
    period = manifest["reference"]["period_3d_m"]
    gap = manifest["evaluation"]["combined_coverage"]
    regions = {
        "seam": (0.0, manifest["seam_radius_m"]),
        "largest_gap": (
            (gap["gap_start_m"] + gap["gap_end_unwrapped_m"]) / 2,
            gap["maximum_gap_m"] / 2,
        ),
        "residual_region": (
            manifest["known_residual_region_center_m"],
            manifest["known_residual_region_radius_m"],
        ),
    }
    candidates = fit_candidates(observations, frames["VER"], period)
    OUTPUT.mkdir(exist_ok=True)
    lines = [
        "Spa Task 5A | Periodic cubic spline baselines",
        f"Input: {len(observations)} reconstruction observations, Task 4 reference period {period:.6f} m.",
        f"Averaging uses per-driver cyclic linear interpolation up to {MAX_DONOR_GAP_M:g} m per donor span.",
        f"Residuals are in-sample XY projections within +/-{PROJECTION_RADIUS_M:g} reference metres of Task 4 matches, with signed Z taken at the same point.",
        f"Largest gap: {gap['gap_start_m']:.3f} to {gap['gap_end_unwrapped_m']:.3f} reference m. Previously flagged region: {regions['residual_region'][0]:.3f} m.",
        "",
    ]
    tables = {}
    for name, (curve, donors) in candidates.items():
        table, length, order, convergence = geometry_table(curve)
        table.to_csv(OUTPUT / f"{name}.csv", index=False, lineterminator="\n")
        tables[name] = table
        lines.extend(
            evaluate(
                name,
                curve,
                table,
                length,
                order,
                convergence,
                donors,
                observations,
                regions,
                frames["VER"],
            )
        )
        lines.append("")
    (OUTPUT / "report.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    plot_results(
        observations,
        tables,
        {name: curve for name, (curve, _) in candidates.items()},
        regions,
        gap,
    )
    print("\n".join(line for line in lines if " | length " in line))
    print(f"Saved four geometry tables, seven plots and the report to {OUTPUT}")


if __name__ == "__main__":
    main()
