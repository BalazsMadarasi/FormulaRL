import numpy as np
import pytest

from formularl.geometry import (
    align_to_reference,
    arc_lengths,
    closure_distances,
    project_onto_segments,
)


def test_arc_lengths_measure_horizontal_and_3d_chords() -> None:
    xyz = np.array([[0, 0, 0], [3, 4, 12], [3, 4, 0], [3, 8, 3]])

    arc = arc_lengths(xyz)

    np.testing.assert_allclose(arc.horizontal_segments, [5, 0, 4])
    np.testing.assert_allclose(arc.spatial_segments, [13, 12, 5])
    np.testing.assert_allclose(arc.cumulative_horizontal, [0, 5, 5, 9])
    np.testing.assert_allclose(arc.cumulative_3d, [0, 13, 25, 30])


def test_arc_lengths_do_not_bridge_missing_coordinates() -> None:
    xyz = np.array([[0, 0, 0], [3, 4, np.nan], [6, 8, 1], [9, 12, 5]])

    arc = arc_lengths(xyz)

    np.testing.assert_allclose(arc.cumulative_horizontal, [0, 5, 10, 15])
    np.testing.assert_allclose(arc.spatial_segments, [np.nan, np.nan, np.sqrt(41)])
    np.testing.assert_allclose(arc.cumulative_3d, [0, np.nan, np.nan, np.nan])


def test_closure_measures_only_first_and_last_samples() -> None:
    xyz = np.array([[0, 0, 12], [-8, 9, 7], [3, 4, 0]])

    assert closure_distances(xyz) == pytest.approx((5, -12, 13))


def test_projection_clamps_both_ends_and_skips_degenerate_segment() -> None:
    point = np.array([5.0, 3.0])
    starts = np.array([[0, 0], [6, 0], [0, 0], [5, 0]], dtype=float)
    ends = np.array([[10, 0], [10, 0], [4, 0], [5, 0]], dtype=float)

    fractions, distances = project_onto_segments(point, starts, ends)

    np.testing.assert_allclose(fractions, [0.5, 0, 1, np.nan])
    np.testing.assert_allclose(distances, [3, np.sqrt(10), np.sqrt(10), np.nan])


def test_alignment_uses_xy_and_interpolates_3d_arc_and_z() -> None:
    reference = np.array([[0, 0, 10], [0, 0, 10], [100, 0, 110], [200, 0, 210]])
    samples = np.array([[0, 0, 999], [50, 2, 80], [150, 2, -300], [200, 0, 210]])

    match = align_to_reference(
        reference, arc_lengths(reference), samples, arc_lengths(samples)
    )

    assert match.degenerate_reference_segments == 1
    assert match.status.tolist() == [
        "start/end boundary",
        "matched",
        "matched",
        "start/end boundary",
    ]
    np.testing.assert_allclose(
        match.reference_arc[match.matched], np.array([50, 150]) * np.sqrt(2)
    )
    np.testing.assert_allclose(match.horizontal_error[match.matched], [2, 2])
    np.testing.assert_allclose(match.vertical_error[match.matched], [20, -460])


def test_alignment_rejects_a_tie_between_separate_sections() -> None:
    reference = np.array([[0, 0, 0], [100, 0, 0], [100, 2, 0], [0, 2, 0]])
    samples = np.array([[0, 0, 0], [20, 1, 0], [100, 0, 0], [0, 2, 0]])

    match = align_to_reference(
        reference, arc_lengths(reference), samples, arc_lengths(samples)
    )

    assert match.status[1] == "ambiguous projection"
    assert not match.matched[1]


def test_alignment_reports_and_excludes_progress_breaks() -> None:
    reference = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]])
    samples = np.array([[0, 0, 0], [80, 0, 0], [65, 0, 0], [150, 0, 0], [200, 0, 0]])

    match = align_to_reference(
        reference, arc_lengths(reference), samples, arc_lengths(samples)
    )

    np.testing.assert_array_equal(match.progress_breaks, [[1, 2]])
    assert match.status[1:4].tolist() == [
        "progress discontinuity",
        "progress discontinuity",
        "matched",
    ]
