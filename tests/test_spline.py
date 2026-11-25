import numpy as np
import pytest

from formularl.spline import (
    arc_map,
    differential_geometry,
    periodic_linear,
    periodic_spline,
    project_xy,
)


def test_circle_periodicity_length_and_spatial_resampling() -> None:
    radius = 10.0
    period = 2 * np.pi * radius
    q = np.linspace(0, period, 48, endpoint=False)
    curve = periodic_spline(
        q,
        np.column_stack(
            (radius * np.cos(q / radius), radius * np.sin(q / radius), np.zeros_like(q))
        ),
        period,
    )
    for order in range(3):
        np.testing.assert_allclose(
            curve(0, order), curve(period, order, extrapolate=False), atol=1e-12
        )
    arc = arc_map(curve)
    assert arc.length_m == pytest.approx(period, abs=1e-4)
    assert arc.differences[-1] < 1e-6
    s, u = arc.uniform()
    step = arc.length_m / len(s)
    xyz = curve(np.r_[u, period])
    chord = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    np.testing.assert_allclose(
        chord, 2 * radius * np.sin(step / (2 * radius)), atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(
        arc.at_distance(np.array([0.0, arc.length_m])), [0, period]
    )


def test_derivative_geometry_and_stationary_samples() -> None:
    v = np.array([[0.0, 4, 2], [3.0, 0, 4], [0.0, 0, 0], [0.0, 0, 2]])
    a = np.array([[-2.0, 0, 0], [0.0, 0, 5], [1.0, 0, 0], [0.0, 0, 0]])
    geometry = differential_geometry(v, a)
    np.testing.assert_allclose(geometry["gradient_pct"], [50, 400 / 3, np.nan, np.nan])
    np.testing.assert_allclose(
        geometry["horizontal_curvature_inv_m"], [1 / 8, 0, np.nan, np.nan]
    )
    np.testing.assert_allclose(
        geometry["vertical_curvature_inv_m"], [0, 15 / 125, np.nan, np.nan]
    )
    np.testing.assert_allclose(
        geometry["inclination_deg"],
        [np.degrees(np.arctan(0.5)), np.degrees(np.arctan(4 / 3)), np.nan, 90],
    )


def test_periodic_interpolation_groups_duplicates_and_masks_sparse_spans() -> None:
    q = np.array([1.0, 1.0, 4.0, 9.0])
    xyz = np.column_stack(([2.0, 4.0, 6.0, 1.0], np.zeros(4), np.zeros(4)))
    result = periodic_linear(q, xyz, np.array([0.0, 1.0, 4.0, 6.0, 10.0]), 10, 3)
    np.testing.assert_allclose(result[:, 0], [2, 3, 6, np.nan, 2])


def test_xy_projection_can_cross_the_seam_without_using_elevation() -> None:
    q = np.array([0.0, 1, 2, 3])
    curve = periodic_spline(
        q, np.array([[1.0, 0, 1], [0.0, 1, 2], [-1.0, 0, 3], [0.0, -1, 4]]), 4
    )
    point = curve(-0.15)
    point[2] += 10
    actual = project_xy(curve, point[None, :], np.array([0.05]), 0.4)
    np.testing.assert_allclose(actual, [-0.15], atol=1e-10)
