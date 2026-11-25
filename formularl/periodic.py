"""Ordered horizontal projection and observation weights for closed polylines."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from formularl.geometry import (
    AMBIGUITY_DISTANCE_M,
    AMBIGUITY_SEPARATION_M,
    BACKWARD_TOLERANCE_M,
    MAX_MATCH_DISTANCE_M,
    PROGRESS_STEP_TOLERANCE_M,
    PROGRESS_WINDOW_M,
    ArcLengths,
    FloatArray,
    IntArray,
    arc_lengths,
    project_onto_segments,
)

BoolArray = NDArray[np.bool_]
ARC_EPSILON_M = 1e-9
SITE_TOLERANCE_M = 0.25
SUPPORT_CAP_MULTIPLE = 3.0


@dataclass(frozen=True)
class ProjectionSettings:
    max_distance_m: float = MAX_MATCH_DISTANCE_M
    phase_tolerance_m: float = PROGRESS_STEP_TOLERANCE_M
    progress_window_m: float = PROGRESS_WINDOW_M
    near_tie_m: float = AMBIGUITY_DISTANCE_M
    branch_separation_m: float = AMBIGUITY_SEPARATION_M
    backward_tolerance_m: float = BACKWARD_TOLERANCE_M
    step_tolerance_m: float = PROGRESS_STEP_TOLERANCE_M


@dataclass(frozen=True)
class ClosedReference:
    vertices: FloatArray
    arc: ArcLengths

    @property
    def period(self) -> float:
        return float(self.arc.cumulative_3d[-1])

    @property
    def horizontal_period(self) -> float:
        return float(self.arc.cumulative_horizontal[-1])

    @property
    def closing_segment(self) -> int:
        return len(self.vertices) - 2

    @property
    def projectable(self) -> BoolArray:
        return self.arc.horizontal_segments > ARC_EPSILON_M


@dataclass(frozen=True)
class Correspondence:
    status: NDArray[np.str_]
    segment: IntArray
    fraction: FloatArray
    arc_m: FloatArray
    unwrapped_m: FloatArray
    xyz: FloatArray
    horizontal_m: FloatArray
    order_alert: BoolArray
    backward: BoolArray
    order_resolved: BoolArray
    phase_horizontal_m: float
    phase_anchors: int
    phase_inliers: int

    @property
    def valid(self) -> BoolArray:
        return self.status == "matched"


@dataclass(frozen=True)
class Weights:
    uniform: FloatArray
    per_driver: FloatArray
    density: FloatArray
    support_m: FloatArray
    capped_support_m: FloatArray
    site_size: IntArray
    support_cap_m: float


def wrap_arc(arc: ArrayLike, period: float) -> FloatArray:
    """Represent the seam at zero, even if a remainder rounds to the period."""
    remainder = np.remainder(arc, period)
    return np.where(remainder >= period, 0.0, remainder)


def circular_delta(arc: ArrayLike, period: float) -> FloatArray:
    return wrap_arc(np.asarray(arc) + period / 2, period) - period / 2


def close_reference(xyz: FloatArray) -> ClosedReference:
    if len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError(
            "Closing the reference requires at least three finite VER vertices."
        )
    vertices = np.vstack((xyz, xyz[0]))
    reference = ClosedReference(vertices, arc_lengths(vertices))
    if not np.isfinite(reference.period) or not reference.projectable.any():
        raise ValueError("The closed reference has no usable horizontal segments.")
    return reference


def _phase(
    progress: FloatArray,
    distance: FloatArray,
    native: FloatArray,
    period: float,
    settings: ProjectionSettings,
) -> tuple[float, int, int]:
    best = np.argmin(distance, axis=1)
    row = np.arange(len(native))
    nearest = distance[row, best]
    anchor = progress[row, best]
    distinct = np.abs(circular_delta(progress - anchor[:, None], period))
    competing = (distance <= nearest[:, None] + settings.near_tie_m) & (
        distinct > settings.branch_separation_m
    )
    usable = (
        np.isfinite(native)
        & (nearest <= settings.max_distance_m)
        & ~competing.any(axis=1)
    )
    offsets = circular_delta(anchor[usable] - native[usable], period)
    if not offsets.size:
        return np.nan, 0, 0
    offsets_from_each = circular_delta(offsets[None, :] - offsets[:, None], period)
    neighborhood = np.abs(offsets_from_each) <= min(
        settings.phase_tolerance_m, period / 4
    )
    best_anchor = np.argmax(neighborhood.sum(axis=1))
    inliers = neighborhood[best_anchor]
    if inliers.sum() <= offsets.size / 2:
        return np.nan, offsets.size, int(inliers.sum())
    phase = offsets[best_anchor] + np.median(offsets_from_each[best_anchor, inliers])
    return float(circular_delta(phase, period)), offsets.size, int(inliers.sum())


def project_ordered(
    reference: ClosedReference,
    xyz: FloatArray,
    native_horizontal_m: FloatArray,
    settings: ProjectionSettings,
) -> Correspondence:
    """Select near-nearest XY candidates consistent with native traversal order."""
    count = len(xyz)
    status = np.full(count, "unavailable phase", dtype="U32")
    segment = np.full(count, -1, dtype=np.intp)
    fraction = np.full(count, np.nan)
    arc_3d = np.full(count, np.nan)
    unwrapped = np.full(count, np.nan)
    projected_xyz = np.full((count, 3), np.nan)
    horizontal = np.full(count, np.nan)
    order_alert = np.zeros(count, dtype=bool)
    backward = np.zeros(count, dtype=bool)
    resolved = np.zeros(count, dtype=bool)

    starts, ends = reference.vertices[:-1], reference.vertices[1:]
    shape = (count, len(starts))
    fractions, distances = np.full(shape, np.nan), np.full(shape, np.inf)
    finite = np.isfinite(xyz).all(axis=1)
    status[~finite] = "invalid observation"
    status[finite & ~np.isfinite(native_horizontal_m)] = "unavailable native progress"
    for i in np.flatnonzero(finite):
        t, d = project_onto_segments(xyz[i, :2], starts[:, :2], ends[:, :2])
        usable = reference.projectable
        fractions[i, usable], distances[i, usable] = t[usable], d[usable]

    h_period = reference.horizontal_period
    raw_progress = (
        reference.arc.cumulative_horizontal[:-1]
        + fractions * reference.arc.horizontal_segments
    )
    raw_spatial = (
        reference.arc.cumulative_3d[:-1] + fractions * reference.arc.spatial_segments
    )
    progress = wrap_arc(raw_progress, h_period)
    spatial_progress = wrap_arc(raw_spatial, reference.period)
    status[
        finite
        & np.isfinite(native_horizontal_m)
        & (distances.min(axis=1) > settings.max_distance_m)
    ] = "too far from reference"
    phase, anchors, inliers = _phase(
        progress, distances, native_horizontal_m, h_period, settings
    )
    previous, previous_lift = -1, np.nan
    progress_window = min(settings.progress_window_m, h_period * 0.49)

    if np.isfinite(phase):
        for i in np.flatnonzero(finite & np.isfinite(native_horizontal_m)):
            best = distances[i].argmin()
            closest = distances[i, best]
            if closest > settings.max_distance_m:
                status[i] = "too far from reference"
                continue
            row_progress = progress[i]
            expected = phase + native_horizontal_m[i]
            laps = np.rint((expected - row_progress) / h_period)
            lifted = row_progress + h_period * laps
            near = (distances[i] <= closest + settings.near_tie_m) & (
                distances[i] <= settings.max_distance_m
            )
            candidates = near & (np.abs(lifted - expected) <= progress_window)
            if not candidates.any():
                status[i] = "outside progress window"
                continue
            if previous >= 0:
                observed_step = native_horizontal_m[i] - native_horizontal_m[previous]
                steps = lifted - previous_lift
                candidates &= (steps >= -settings.backward_tolerance_m) & (
                    np.abs(steps - observed_step) <= settings.step_tolerance_m
                )
                if not candidates.any():
                    status[i] = "progress discontinuity"
                    order_alert[i] = True
                    continue
            chosen = np.argmin(np.where(candidates, distances[i], np.inf))
            separation = np.abs(
                circular_delta(row_progress - row_progress[chosen], h_period)
            )
            separated_in_3d = (
                np.abs(
                    circular_delta(
                        spatial_progress[i] - spatial_progress[i, chosen],
                        reference.period,
                    )
                )
                > ARC_EPSILON_M
            )
            separate = (separation > settings.branch_separation_m) | (
                (separation <= ARC_EPSILON_M) & separated_in_3d
            )
            if np.any(candidates & separate):
                status[i] = "ambiguous projection"
                continue

            t = fractions[i, chosen]
            arc_3d[i] = spatial_progress[i, chosen]
            winding = np.rint((lifted[chosen] - raw_progress[i, chosen]) / h_period)
            unwrapped[i] = raw_spatial[i, chosen] + winding * reference.period
            segment[i], fraction[i] = chosen, t
            projected_xyz[i] = starts[chosen] + t * (ends[chosen] - starts[chosen])
            horizontal[i] = distances[i, chosen]
            backward[i] = (
                previous >= 0 and unwrapped[i] < unwrapped[previous] - ARC_EPSILON_M
            )
            resolved[i] = np.any(near & separate)
            status[i] = "matched"
            previous, previous_lift = i, lifted[chosen]

    return Correspondence(
        status,
        segment,
        fraction,
        arc_3d,
        unwrapped,
        projected_xyz,
        horizontal,
        order_alert,
        backward,
        resolved,
        phase,
        anchors,
        inliers,
    )


def circular_gaps(
    positions: FloatArray, period: float
) -> tuple[FloatArray, FloatArray]:
    """Return the starts and lengths of forward gaps, including last to first."""
    ordered = np.sort(wrap_arc(positions, period))
    if not ordered.size:
        return ordered, ordered.copy()
    return ordered, np.diff(np.r_[ordered, ordered[0] + period])


def observation_weights(
    arc: FloatArray, drivers: NDArray[np.str_], period: float
) -> Weights:
    """Give each candidate unit mass; unsupported observations have zero mass."""
    uniform, per_driver, density, support, capped = (
        np.zeros(len(arc)) for _ in range(5)
    )
    site_size = np.zeros(len(arc), dtype=np.intp)
    rows = np.flatnonzero(np.isfinite(arc))
    if not rows.size:
        return Weights(uniform, per_driver, density, support, capped, site_size, np.nan)

    uniform[rows] = 1 / len(rows)
    _, inverse, counts = np.unique(
        drivers[rows], return_inverse=True, return_counts=True
    )
    per_driver[rows] = 1 / (counts[inverse] * len(counts))

    positions = wrap_arc(arc[rows], period)
    order = np.argsort(positions, kind="stable")
    sorted_positions = positions[order]
    _, gap = circular_gaps(sorted_positions, period)
    cut = (int(gap.argmax()) + 1) % len(rows)
    order = np.r_[order[cut:], order[:cut]]
    lifted = np.r_[sorted_positions[cut:], sorted_positions[:cut] + period]
    boundaries = [0]
    for i, position in enumerate(lifted):
        if position - lifted[boundaries[-1]] > SITE_TOLERANCE_M:
            boundaries.append(i)
    groups = np.split(rows[order], boundaries[1:])
    sites = np.array([group.mean() for group in np.split(lifted, boundaries[1:])])
    gap = np.diff(np.r_[sites, sites[0] + period])
    mass = (gap + np.roll(gap, 1)) / 2
    cap = float(SUPPORT_CAP_MULTIPLE * np.median(mass))
    for members, site_mass in zip(groups, mass, strict=True):
        support[members] = site_mass / len(members)
        capped[members] = min(site_mass, cap) / len(members)
        site_size[members] = len(members)
    density = capped / capped.sum()
    return Weights(uniform, per_driver, density, support, capped, site_size, cap)
