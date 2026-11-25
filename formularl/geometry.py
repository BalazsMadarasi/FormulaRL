"""Polyline measurements in metres for position samples in their original order."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

MAX_MATCH_DISTANCE_M = 25.0
PROGRESS_WINDOW_M = 150.0
AMBIGUITY_DISTANCE_M = 1.0
AMBIGUITY_SEPARATION_M = 30.0
BACKWARD_TOLERANCE_M = 10.0
PROGRESS_STEP_TOLERANCE_M = 30.0


@dataclass(frozen=True)
class ArcLengths:
    horizontal_segments: FloatArray
    spatial_segments: FloatArray
    cumulative_horizontal: FloatArray
    cumulative_3d: FloatArray


@dataclass(frozen=True)
class Alignment:
    reference_arc: FloatArray
    horizontal_error: FloatArray
    vertical_error: FloatArray
    status: NDArray[np.str_]
    progress_breaks: IntArray
    degenerate_reference_segments: int

    @property
    def matched(self) -> NDArray[np.bool_]:
        return self.status == "matched"


def arc_lengths(xyz: FloatArray) -> ArcLengths:
    """Measure adjacent chords. Undefined segments interrupt cumulative distance."""
    delta = np.diff(np.asarray(xyz, dtype=float), axis=0)
    delta[~np.isfinite(delta)] = np.nan
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    spatial = np.linalg.norm(delta, axis=1)
    return ArcLengths(
        horizontal,
        spatial,
        np.concatenate(([0.0], np.cumsum(horizontal)))[: len(xyz)],
        np.concatenate(([0.0], np.cumsum(spatial)))[: len(xyz)],
    )


def closure_distances(xyz: FloatArray) -> tuple[float, float, float]:
    """Return horizontal separation, signed end-minus-start Z, and 3D separation."""
    if len(xyz) < 2:
        return np.nan, np.nan, np.nan
    delta = np.asarray(xyz[-1], dtype=float) - xyz[0]
    delta[~np.isfinite(delta)] = np.nan
    horizontal = float(np.linalg.norm(delta[:2]))
    return horizontal, float(delta[2]), float(np.linalg.norm(delta))


def project_onto_segments(
    point_xy: FloatArray, starts_xy: FloatArray, ends_xy: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return clamped segment fractions and horizontal distances for one point."""
    vectors = ends_xy - starts_xy
    lengths_sq = np.sum(vectors**2, axis=1)
    fractions = np.divide(
        np.sum((point_xy - starts_xy) * vectors, axis=1),
        lengths_sq,
        out=np.full_like(lengths_sq, np.nan, dtype=float),
        where=np.isfinite(lengths_sq) & (lengths_sq > 0),
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projected = starts_xy + fractions[:, None] * vectors
    return fractions, np.linalg.norm(point_xy - projected, axis=1)


def _progress_breaks(
    status: NDArray[np.str_],
    reference_progress: FloatArray,
    sample_progress: FloatArray,
) -> IntArray:
    """Return pairs of candidate rows with inconsistent forward progress."""
    rows = np.flatnonzero(status == "matched")
    ref_step = np.diff(reference_progress[rows])
    sample_step = np.diff(sample_progress[rows])
    jumps = (ref_step < -BACKWARD_TOLERANCE_M) | (
        np.abs(ref_step - sample_step) > PROGRESS_STEP_TOLERANCE_M
    )
    return np.column_stack((rows[:-1][jumps], rows[1:][jumps]))


def align_to_reference(
    reference: FloatArray, ref_arc: ArcLengths, samples: FloatArray, arc: ArcLengths
) -> Alignment:
    """Match by horizontal projection and progress, then evaluate Z and 3D arc."""
    reference_s = np.full(len(samples), np.nan)
    reference_progress = np.full(len(samples), np.nan)
    horizontal_error = np.full(len(samples), np.nan)
    vertical_error = np.full(len(samples), np.nan)
    status = np.full(len(samples), "unavailable progress", dtype="U32")

    ref_xy = reference[:, :2]
    ref_progress = ref_arc.cumulative_horizontal
    sample_progress = arc.cumulative_horizontal
    degenerate = int(np.count_nonzero(ref_arc.horizontal_segments == 0))
    if (
        len(reference) >= 2
        and len(samples) >= 2
        and np.isfinite(ref_progress[-1])
        and np.isfinite(sample_progress[-1])
        and min(ref_progress[-1], sample_progress[-1]) > 0
    ):
        ref_length = ref_progress[-1]
        expected_progress = sample_progress * (ref_length / sample_progress[-1])
        for i, (point, expected) in enumerate(
            zip(samples, expected_progress, strict=True)
        ):
            if not np.isfinite(point).all():
                status[i] = "invalid sample"
                continue
            fractions, distances = project_onto_segments(
                point[:2], ref_xy[:-1], ref_xy[1:]
            )
            best = np.nanargmin(distances)
            fraction = fractions[best]
            progress = ref_progress[:-1] + fractions * ref_arc.horizontal_segments
            best_progress = progress[best]
            candidates = distances <= distances[best] + AMBIGUITY_DISTANCE_M
            if distances[best] > MAX_MATCH_DISTANCE_M:
                status[i] = "too far from reference"
                continue
            if np.any(
                candidates & (np.abs(progress - best_progress) > AMBIGUITY_SEPARATION_M)
            ):
                status[i] = "ambiguous projection"
                continue
            near_start = min(best_progress, expected) <= PROGRESS_WINDOW_M
            near_end = max(best_progress, expected) >= ref_length - PROGRESS_WINDOW_M
            if best_progress in (0.0, ref_length) or (
                near_start
                and near_end
                and abs(best_progress - expected) > PROGRESS_WINDOW_M
            ):
                status[i] = "start/end boundary"
                continue
            if abs(best_progress - expected) > PROGRESS_WINDOW_M:
                status[i] = "outside progress window"
                continue
            reference_s[i] = (
                ref_arc.cumulative_3d[best] + fraction * ref_arc.spatial_segments[best]
            )
            if not np.isfinite(reference_s[i]):
                status[i] = "unavailable reference Z/arc"
                continue
            reference_z = reference[best, 2] + fraction * (
                reference[best + 1, 2] - reference[best, 2]
            )
            reference_progress[i] = best_progress
            horizontal_error[i] = distances[best]
            vertical_error[i] = point[2] - reference_z
            status[i] = "matched"

    breaks = _progress_breaks(status, reference_progress, sample_progress)
    status[breaks.ravel()] = "progress discontinuity"
    return Alignment(
        reference_s, horizontal_error, vertical_error, status, breaks, degenerate
    )
