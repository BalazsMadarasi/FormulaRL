"""Build periodic correspondence and fitting observations from prepared Spa laps."""

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from formularl.dataset import RECURRENCE_ARC_M, REFERENCE_DRIVER
from formularl.geometry import FloatArray
from formularl.periodic import (
    ARC_EPSILON_M,
    SITE_TOLERANCE_M,
    SUPPORT_CAP_MULTIPLE,
    ClosedReference,
    ProjectionSettings,
    circular_delta,
    circular_gaps,
    close_reference,
    observation_weights,
    project_ordered,
    wrap_arc,
)
from formularl.reconstruction_data import XYZ, load_reconstruction

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "datasets"
OUTPUT = ROOT / "outputs" / "reconstruction"
COLORS = {"VER": "tab:blue", "LEC": "tab:red", "HAM": "tab:green"}
WEIGHT_COLUMNS = ["weight_uniform", "weight_per_driver", "weight_density"]
BIN_TARGET_M = 100.0
SEAM_RADIUS_M = 50.0
RESIDUAL_REGION_RADIUS_M = 25.0


def reference_table(reference: ClosedReference) -> pd.DataFrame:
    count = len(reference.vertices) - 1
    return pd.DataFrame(
        {
            "segment_index": np.arange(count),
            "start_sample_index": np.arange(count),
            "end_sample_index": np.roll(np.arange(count), -1),
            "closing_chord": np.arange(count) == reference.closing_segment,
            "projectable_in_xy": reference.projectable,
            "start_arc_3d_m": reference.arc.cumulative_3d[:-1],
            "end_arc_3d_m": reference.arc.cumulative_3d[1:],
            "length_xy_m": reference.arc.horizontal_segments,
            "length_3d_m": reference.arc.spatial_segments,
            **{
                f"{side}_{name}": vertices[:, axis]
                for side, vertices in (
                    ("start", reference.vertices[:-1]),
                    ("end", reference.vertices[1:]),
                )
                for axis, name in enumerate(XYZ)
            },
        }
    )


def prepare_observations(
    frames: dict[str, pd.DataFrame],
    reference: ClosedReference,
    settings: ProjectionSettings,
) -> tuple[pd.DataFrame, dict, float]:
    pieces = []
    phase = {}
    for driver, source in frames.items():
        xyz = source[XYZ].to_numpy()
        match = project_ordered(
            reference, xyz, source["arc_horizontal_m"].to_numpy(), settings
        )
        result = source.copy()
        result["reference_arc_3d_m"] = match.arc_m
        result["periodic_u"] = wrap_arc(match.arc_m / reference.period, 1.0)
        result["unwrapped_arc_3d_m"] = match.unwrapped_m
        result["unwrapped_u"] = match.unwrapped_m / reference.period
        result["reference_segment"] = pd.array(
            np.where(match.valid, match.segment, None), dtype="Int64"
        )
        result["reference_segment_fraction"] = match.fraction
        result[["reference_x_m", "reference_y_m", "reference_z_m"]] = match.xyz
        result["horizontal_residual_m"] = match.horizontal_m
        result["vertical_residual_m"] = xyz[:, 2] - match.xyz[:, 2]
        result["residual_3d_m"] = np.hypot(
            result["horizontal_residual_m"], result["vertical_residual_m"]
        )
        result["correspondence_status"] = match.status
        result["correspondence_valid"] = match.valid
        result["correspondence_ambiguous"] = match.status == "ambiguous projection"
        result["correspondence_resolved_by_order"] = match.order_resolved
        result["projection_endpoint"] = (match.fraction == 0) | (match.fraction == 1)
        result["projection_on_closing_chord"] = (
            match.segment == reference.closing_segment
        )
        result["projection_backward"] = match.backward
        result["projection_progress_discontinuity"] = match.order_alert
        pieces.append(result)
        phase[driver] = {
            "offset_horizontal_m": float(match.phase_horizontal_m)
            if np.isfinite(match.phase_horizontal_m)
            else None,
            "anchor_count": int(match.phase_anchors),
            "consensus_count": match.phase_inliers,
        }
    observations = pd.concat(pieces, ignore_index=True)
    weights = observation_weights(
        observations["reference_arc_3d_m"].to_numpy(),
        observations["driver"].to_numpy(),
        reference.period,
    )
    observations[WEIGHT_COLUMNS] = np.column_stack(
        (weights.uniform, weights.per_driver, weights.density)
    )
    observations["density_support_per_observation_m"] = weights.support_m
    observations["density_capped_support_per_observation_m"] = weights.capped_support_m
    observations["density_site_members"] = weights.site_size
    return observations, phase, weights.support_cap_m


def distribution(values: FloatArray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {}
    return {
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def coverage(group: pd.DataFrame, period: float, bins: FloatArray) -> dict:
    valid = group.loc[group["correspondence_valid"]]
    arc = valid["reference_arc_3d_m"].to_numpy()
    starts, gaps = circular_gaps(arc, period)
    counts = np.histogram(arc, bins=bins)[0]
    largest = gaps.argmax() if len(gaps) else None
    return {
        "maximum_gap_m": float(gaps[largest]) if largest is not None else None,
        "gap_start_m": float(starts[largest]) if largest is not None else None,
        "gap_end_unwrapped_m": float(starts[largest] + gaps[largest])
        if largest is not None
        else None,
        "observations_per_km": len(valid) * 1000 / period,
        "empty_bins": int((counts == 0).sum()),
        "min_median_max_count_per_bin": [
            int(counts.min()),
            float(np.median(counts)),
            int(counts.max()),
        ],
    }


def evaluate(
    observations: pd.DataFrame, reference: ClosedReference, bins: FloatArray
) -> dict:
    period = reference.period
    result = {}
    for driver, group in observations.groupby("driver", sort=False):
        valid = group.loc[group["correspondence_valid"]]
        arc = valid["reference_arc_3d_m"].to_numpy()
        seam = valid.loc[np.abs(circular_delta(arc, period)) <= SEAM_RADIUS_M]
        known_region = valid.loc[
            np.abs(circular_delta(arc - RECURRENCE_ARC_M, period))
            <= RESIDUAL_REGION_RADIUS_M
        ]
        wraps = np.floor(valid["unwrapped_u"])
        adjacent = valid["sample_index"].diff().eq(1)
        step_error = (
            valid["unwrapped_arc_3d_m"].diff() - valid["arc_3d_m"].diff()
        ).abs()
        result[driver] = {
            "observations": len(group),
            "matched": len(valid),
            "ambiguous": int(group["correspondence_ambiguous"].sum()),
            "unsupported": int(
                (
                    ~group["correspondence_valid"] & ~group["correspondence_ambiguous"]
                ).sum()
            ),
            "status_counts": {
                str(key): int(value)
                for key, value in group["correspondence_status"].value_counts().items()
            },
            "progress_discontinuities": int(
                group["projection_progress_discontinuity"].sum()
            ),
            "accepted_backsteps": int(group["projection_backward"].sum()),
            "order_resolved": int(group["correspondence_resolved_by_order"].sum()),
            "wrap_transitions": int(np.count_nonzero(np.diff(wraps))),
            "absolute_adjacent_3d_step_error_m": distribution(
                step_error[adjacent].to_numpy()
            ),
            "endpoints": int(valid["projection_endpoint"].sum()),
            "unwrapped_span_m": distribution(valid["unwrapped_arc_3d_m"].to_numpy()),
            "horizontal_m": distribution(valid["horizontal_residual_m"].to_numpy()),
            "signed_vertical_m": distribution(valid["vertical_residual_m"].to_numpy()),
            "absolute_vertical_m": distribution(
                valid["vertical_residual_m"].abs().to_numpy()
            ),
            "vertical_bias_m": float(valid["vertical_residual_m"].mean())
            if len(valid)
            else None,
            "residual_3d_m": distribution(valid["residual_3d_m"].to_numpy()),
            "coverage": coverage(group, period, bins),
            "seam": {
                "observations": len(seam),
                "chord_interior": int(
                    (
                        seam["projection_on_closing_chord"]
                        & ~seam["projection_endpoint"]
                    ).sum()
                ),
                "residual_3d_m": distribution(seam["residual_3d_m"].to_numpy()),
            },
            "known_residual_region": {
                "observations": len(known_region),
                "native_temporal_flags": int(
                    known_region["flag_persistent_residual"].eq(True).sum()
                ),
                "residual_3d_m": distribution(known_region["residual_3d_m"].to_numpy()),
            },
            "weight_mass": {key: float(group[key].sum()) for key in WEIGHT_COLUMNS},
        }
    return {
        "drivers": result,
        "combined_coverage": coverage(observations, period, bins),
        "weights": {
            key: {
                "valid_weight_distribution": distribution(
                    observations.loc[
                        observations["correspondence_valid"], key
                    ].to_numpy()
                ),
                "effective_sample_size": float(1 / (observations[key] ** 2).sum())
                if observations[key].any()
                else None,
            }
            for key in WEIGHT_COLUMNS
        },
        "observations_sharing_a_density_site": int(
            (observations["density_site_members"] > 1).sum()
        ),
        "density_sites": round(
            (
                1
                / observations.loc[
                    observations["density_site_members"] > 0, "density_site_members"
                ]
            ).sum()
        ),
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "unavailable"


def stats_text(stats: dict) -> str:
    return " / ".join(fmt(stats.get(key)) for key in ("min", "median", "p95", "max"))


def report_text(
    reference: ClosedReference, evaluation: dict, support_cap: float, bin_count: int
) -> str:
    i = reference.closing_segment
    closing = reference.vertices[-1] - reference.vertices[-2]
    lines = [
        "Spa qualifying | periodic reconstruction correspondence",
        f"3D period {reference.period:.3f} m; XY length {reference.horizontal_period:.3f} m.",
        (
            f"Origin VER sample 0. Artificial last-to-first chord {reference.arc.spatial_segments[i]:.3f} m in 3D,"
            f" {reference.arc.horizontal_segments[i]:.3f} m horizontally, signed rise {closing[2]:+.3f} m."
        ),
        f"Reference segments {len(reference.projectable)}; horizontally degenerate {np.count_nonzero(~reference.projectable)}.",
        f"Equal-width density bins: {bin_count} at {reference.period / bin_count:.2f} m.",
        "",
    ]
    for driver, info in evaluation["drivers"].items():
        cov = info["coverage"]
        seam, region = info["seam"], info["known_residual_region"]
        lines.extend(
            [
                (
                    f"{driver}: {info['matched']}/{info['observations']} matched;"
                    f" {info['ambiguous']} ambiguous, {info['unsupported']} unsupported."
                ),
                f"  Status counts {info['status_counts']}.",
                (
                    f"  Order-resolved {info['order_resolved']}; rejected progress discontinuities {info['progress_discontinuities']};"
                    f" accepted backsteps {info['accepted_backsteps']}; wrap transitions {info['wrap_transitions']}; endpoint projections {info['endpoints']}."
                ),
                f"  Unwrapped 3D span {fmt(info['unwrapped_span_m'].get('min'))} to {fmt(info['unwrapped_span_m'].get('max'))} m.",
                f"  Maximum adjacent 3D step difference from native arc {fmt(info['absolute_adjacent_3d_step_error_m'].get('max'))} m.",
                (
                    f"  Residual min / median / p95 / max in metres: XY {stats_text(info['horizontal_m'])};"
                    f" signed Z {stats_text(info['signed_vertical_m'])}; 3D {stats_text(info['residual_3d_m'])}."
                ),
                f"  Absolute Z {stats_text(info['absolute_vertical_m'])} m; signed Z mean {fmt(info['vertical_bias_m'])} m.",
                (
                    f"  Density {cov['observations_per_km']:.1f}/km; {cov['empty_bins']} empty bins;"
                    f" max circular gap {fmt(cov['maximum_gap_m'])} m."
                ),
                (
                    f"  Seam +/-{SEAM_RADIUS_M:g} m: {seam['observations']} samples, {seam['chord_interior']} in closing chord interior;"
                    f" 3D residual median {fmt(seam['residual_3d_m'].get('median'))} m."
                ),
                (
                    f"  s={RECURRENCE_ARC_M:.2f} +/-{RESIDUAL_REGION_RADIUS_M:g} m: {region['observations']} samples,"
                    f" {region['native_temporal_flags']} persistent temporal flags;"
                    f" 3D residual median {fmt(region['residual_3d_m'].get('median'))} m."
                ),
                "  Weight mass (uniform / per driver / density): "
                + " / ".join(
                    f"{info['weight_mass'][key]:.2%}" for key in WEIGHT_COLUMNS
                ),
                "",
            ]
        )
    combined = evaluation["combined_coverage"]
    lines.extend(
        [
            (
                f"Combined: density {combined['observations_per_km']:.1f}/km; {combined['empty_bins']} empty bins;"
                f" max circular gap {fmt(combined['maximum_gap_m'])} m from s={fmt(combined['gap_start_m'])}"
                f" to {fmt(combined['gap_end_unwrapped_m'])} m (forward unwrapped end)."
            ),
            (
                f"Density groups: {evaluation['density_sites']} sites,"
                f" {evaluation['observations_sharing_a_density_site']} observations in shared sites,"
                f" cap {fmt(support_cap)} m per site."
            ),
        ]
    )
    for label, stats in evaluation["weights"].items():
        weight_range = " / ".join(
            fmt(stats["valid_weight_distribution"].get(k), 7)
            for k in ("min", "median", "p95", "max")
        )
        lines.append(
            f"  {label}: min / median / p95 / max {weight_range}; effective sample size {fmt(stats['effective_sample_size'], 1)}."
        )
    lines.extend(
        [
            "",
            "VER residuals are measured against its own polyline. Greater coverage describes sampling density.",
            "Absolute positional accuracy requires independent evidence. The closing chord bridges unobserved space.",
        ]
    )
    return "\n".join(lines) + "\n"


def draw_region(
    axes: Axes,
    reference: ClosedReference,
    center: float,
    radius: float,
    *,
    gap: bool = False,
) -> None:
    s = reference.arc.cumulative_3d
    xy = reference.vertices[:, :2]
    for shift in (-reference.period, 0.0, reference.period):
        start, end = s[:-1] + shift - center, s[1:] + shift - center
        selected = (start < radius) & (end > -radius) & (end > start)
        i = np.flatnonzero(selected)
        lo = (np.maximum(start[i], -radius) - start[i]) / (end[i] - start[i])
        hi = (np.minimum(end[i], radius) - start[i]) / (end[i] - start[i])
        vector = xy[i + 1] - xy[i]
        segments = np.stack(
            (xy[i] + lo[:, None] * vector, xy[i] + hi[:, None] * vector), axis=1
        )
        closes = i == reference.closing_segment
        if gap:
            axes.add_collection(
                LineCollection(segments, colors="goldenrod", linewidths=4, zorder=2)
            )
        else:
            axes.add_collection(
                LineCollection(segments[~closes], colors="0.5", linewidths=1.5)
            )
            axes.add_collection(
                LineCollection(
                    segments[closes], colors="0.3", linestyles="dashed", linewidths=1.4
                )
            )


def detail_plot(
    reference: ClosedReference,
    matched: pd.DataFrame,
    center: float,
    radius: float,
    gap_m: float | None = None,
) -> plt.Figure:
    figure, (plan, arc_plot) = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    draw_region(plan, reference, center, radius)
    if gap_m is not None:
        draw_region(plan, reference, center, gap_m / 2, gap=True)
        arc_plot.axvspan(
            -gap_m / 2,
            gap_m / 2,
            color="goldenrod",
            alpha=0.2,
            label="Largest combined gap",
        )
    else:
        first = reference.vertices[0]
        plan.scatter(
            first[0],
            first[1],
            s=85,
            marker="+",
            color="black",
            label="Provisional origin",
            zorder=5,
        )
        plan.plot([], [], color="0.3", linestyle="dashed", label="Closing chord")
        arc_plot.axvline(0, color="0.35", linestyle="dashed")
    for driver, group in matched.groupby("driver", sort=False):
        offsets = circular_delta(
            group["reference_arc_3d_m"].to_numpy() - center, reference.period
        )
        show = np.abs(offsets) <= radius
        sample = group.loc[show]
        xy = sample[["x_m", "y_m"]].to_numpy()
        projected = sample[["reference_x_m", "reference_y_m"]].to_numpy()
        plan.add_collection(
            LineCollection(
                np.stack((xy, projected), axis=1), colors=COLORS[driver], alpha=0.55
            )
        )
        plan.scatter(*xy.T, s=21, color=COLORS[driver], label=driver, zorder=3)
        arc_plot.scatter(
            offsets[show],
            sample["horizontal_residual_m"],
            s=21,
            color=COLORS[driver],
            label=driver,
        )
    plan.autoscale_view()
    plan.set(
        aspect="equal",
        xlabel="X (m)",
        ylabel="Y (m)",
        title="Observations and XY projections",
    )
    arc_plot.set(
        xlabel="Arc offset from origin (m)"
        if gap_m is None
        else "Arc offset from gap midpoint (m)",
        ylabel="Horizontal residual (m)",
        xlim=(-radius, radius),
        title="Projected positions along the reference",
    )
    plan.legend(fontsize="small")
    arc_plot.legend(fontsize="small")
    return figure


def save_plots(
    observations: pd.DataFrame,
    reference: ClosedReference,
    evaluation: dict,
    bins: FloatArray,
    output: Path,
) -> None:
    matplotlib.use("Agg")
    native_fig, native = plt.subplots(figsize=(8, 7), layout="constrained")
    density_fig, density = plt.subplots(figsize=(10, 4), layout="constrained")
    residual_fig, residuals = plt.subplots(
        3, 1, sharex=True, figsize=(10, 7), layout="constrained"
    )
    weight_fig, weights = plt.subplots(
        3, 1, sharex=True, figsize=(10, 7), layout="constrained"
    )
    matched = observations.loc[observations["correspondence_valid"]]
    for driver, group in observations.groupby("driver", sort=False):
        color = COLORS[driver]
        native.scatter(
            group["x_m"], group["y_m"], s=9, alpha=0.7, color=color, label=driver
        )
        valid = group.loc[group["correspondence_valid"]]
        density.stairs(
            np.histogram(valid["reference_arc_3d_m"], bins=bins)[0],
            bins,
            color=color,
            label=driver,
        )
        u = valid["periodic_u"]
        for ax, key in zip(
            residuals,
            ("horizontal_residual_m", "vertical_residual_m", "residual_3d_m"),
            strict=True,
        ):
            ax.scatter(u, valid[key], s=9, color=color, label=driver)
        for ax, key in zip(weights, WEIGHT_COLUMNS, strict=True):
            ax.scatter(u, valid[key], s=9, color=color, label=driver)
    native.set(aspect="equal", xlabel="X (m)", ylabel="Y (m)")
    native.legend()
    density.stairs(
        np.histogram(matched["reference_arc_3d_m"], bins=bins)[0],
        bins,
        color="black",
        label="Combined",
    )
    density.set(
        xlabel="Reference 3D arc (m)",
        ylabel=f"Observations per {np.diff(bins)[0]:.1f} m bin",
    )
    density.legend()
    for ax, label in zip(
        residuals, ("Horizontal (m)", "Signed vertical (m)", "3D (m)"), strict=True
    ):
        ax.set_ylabel(label)
        ax.set_xlim(0, 1)
    residuals[0].legend(loc="upper left", ncols=3)
    residuals[-1].set_xlabel("Periodic coordinate")
    for ax, title in zip(
        weights, ("Uniform", "Equal mass per driver", "Spatial support"), strict=True
    ):
        ax.set(ylabel=title, xlim=(0, 1))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    weights[0].legend(loc="upper left", ncols=3)
    weights[-1].set_xlabel("Periodic coordinate")
    seam = detail_plot(reference, matched, 0, SEAM_RADIUS_M)
    coverage_metrics = evaluation["combined_coverage"]
    gap = coverage_metrics["maximum_gap_m"]
    if gap is None:
        gap_fig, ax = plt.subplots(layout="constrained")
        ax.text(
            0.5,
            0.5,
            "Combined correspondence unavailable",
            ha="center",
            transform=ax.transAxes,
        )
    else:
        center = float(
            wrap_arc(coverage_metrics["gap_start_m"] + gap / 2, reference.period)
        )
        gap_fig = detail_plot(
            reference, matched, center, min(gap / 2 + 30, reference.period * 0.49), gap
        )
    for fig, name, title in (
        (native_fig, "native_observations.png", "Combined native observations"),
        (
            density_fig,
            "sampling_density.png",
            "Sampling density on the provisional reference",
        ),
        (residual_fig, "projection_residuals.png", "Projection residuals by driver"),
        (
            weight_fig,
            "candidate_weights.png",
            "Candidate observation weights (unit total mass)",
        ),
        (seam, "periodic_seam.png", "Periodic seam correspondence"),
        (gap_fig, "largest_combined_gap.png", f"Largest combined gap ({fmt(gap)} m)"),
    ):
        fig.suptitle(title)
        fig.savefig(output / name, dpi=150)
        plt.close(fig)


def main(input_dir: Path = INPUT, output: Path = OUTPUT) -> int:
    settings = ProjectionSettings()
    try:
        frames, provenance = load_reconstruction(input_dir)
        reference = close_reference(frames[REFERENCE_DRIVER][XYZ].to_numpy())
    except (KeyError, TypeError, OSError, ValueError) as error:
        print(f"Cannot prepare periodic observations: {error}", file=sys.stderr)
        return 1
    observations, phase, cap = prepare_observations(frames, reference, settings)
    bins = np.linspace(
        0, reference.period, int(np.ceil(reference.period / BIN_TARGET_M)) + 1
    )
    evaluation = evaluate(observations, reference, bins)
    report = report_text(reference, evaluation, cap, len(bins) - 1)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "observations.csv": observations,
        "reference_polyline.csv": reference_table(reference),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, lineterminator="\n")
    (output / "report.txt").write_text(report, encoding="utf-8", newline="\n")
    save_plots(observations, reference, evaluation, bins, output)
    manifest = {
        "schema_version": 1,
        "inputs": provenance,
        "reference": {
            "driver": REFERENCE_DRIVER,
            "origin_sample_index": 0,
            "period_3d_m": reference.period,
            "period_horizontal_m": reference.horizontal_period,
            "closing_segment_index": reference.closing_segment,
            "closing_chord_3d_m": float(reference.arc.spatial_segments[-1]),
            "closing_chord_xy_m": float(reference.arc.horizontal_segments[-1]),
            "closing_chord_rise_m": float(
                reference.vertices[-1, 2] - reference.vertices[-2, 2]
            ),
            "degenerate_xy_segments": int((~reference.projectable).sum()),
        },
        "correspondence": {
            "method": "Near-nearest XY projection, majority circular phase consensus and ordered horizontal step gates.",
            "phase_by_driver": phase,
            "thresholds": asdict(settings),
            "reference_xy_degeneracy_epsilon_m": ARC_EPSILON_M,
            "phase_radius_cap_fraction_of_xy_period": 0.25,
            "progress_window_cap_fraction_of_xy_period": 0.49,
            "vertical_residual_sign": "observed minus reference",
        },
        "weights": {
            "normalization": "Unit total mass across matched rows for each candidate; unmatched rows have zero mass.",
            "site_width_m": SITE_TOLERANCE_M,
            "density_description": "Pooled cyclic Voronoi support per site, capped before division among its observations.",
            "cap_multiple_of_median_site_support": SUPPORT_CAP_MULTIPLE,
            "site_support_cap_m": cap if np.isfinite(cap) else None,
        },
        "evaluation": evaluation,
        "density_bin_width_m": float(np.diff(bins)[0]),
        "seam_radius_m": SEAM_RADIUS_M,
        "known_residual_region_center_m": RECURRENCE_ARC_M,
        "known_residual_region_radius_m": RESIDUAL_REGION_RADIUS_M,
        "limitations": [
            "Origin is the first observed VER position; exact official start/finish is unknown.",
            "The straight closing chord approximates an unsampled boundary. Geometry is in original metres.",
            "Order checks are local after phase estimation. Close branches and sparse samples can limit identifiability.",
            "VER self-residuals are reference-internal. Combined coverage provides no independent accuracy estimate.",
            "Candidate weights describe sample balance and spatial support, with no inference of measurement variance.",
        ],
        "table_sha256": {
            name: sha256((output / name).read_bytes()).hexdigest() for name in tables
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(report, end="")
    print(f"Saved {len(observations)} native observations and six plots to {output}")
    return 0 if observations["correspondence_valid"].any() else 1


if __name__ == "__main__":
    ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
