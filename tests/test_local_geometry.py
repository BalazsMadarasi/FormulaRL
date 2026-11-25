import numpy as np

from formularl.geometry import (
    arc_lengths,
    horizontal_curvature,
    inclination_and_gradient,
    vertical_curvature,
    window_gradient,
)


def test_inclination_and_grade_from_known_run_and_rise() -> None:
    run = np.array([4.0, 4.0, 0.0, 0.0])
    rise = np.array([3.0, -3.0, 2.0, 0.0])
    angle, grade = inclination_and_gradient(run, rise)

    uphill = np.degrees(np.arctan(0.75))
    np.testing.assert_allclose(angle, [uphill, -uphill, 90, np.nan])
    np.testing.assert_allclose(grade, [75, -75, np.nan, np.nan])


def test_window_grade_uses_horizontal_path_length_around_a_bend() -> None:
    xyz = np.array([[0.0, 0, 0], [3.0, 0, 4], [3.0, 3, 8]])
    arc = arc_lengths(xyz)

    grade = window_gradient(arc, xyz[:, 2], np.array([5.0, 1.0]), width_m=8.0)

    np.testing.assert_allclose(grade, [100 * 4 / 3, np.nan])


def test_horizontal_curvature_for_circle_straight_and_duplicate() -> None:
    xy = np.array([[10.0, 0], [0.0, 10], [-10.0, 0]])
    np.testing.assert_allclose(horizontal_curvature(xy), [0.1])
    np.testing.assert_allclose(horizontal_curvature(xy[::-1]), [-0.1])
    np.testing.assert_allclose(
        horizontal_curvature(np.array([[0, 0], [1, 0], [2, 0]])), [0]
    )
    np.testing.assert_allclose(horizontal_curvature(xy[[0, 0, 1]]), [np.nan])


def test_vertical_curvature_uses_spacing_between_segment_midpoints() -> None:
    lengths = np.array([1.0, np.sqrt(2), 0.0])
    angle = np.array([0.0, 45.0, 45.0])
    expected = (np.pi / 4) / ((1 + np.sqrt(2)) / 2)

    np.testing.assert_allclose(vertical_curvature(angle, lengths), [expected, np.nan])
