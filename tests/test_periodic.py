import numpy as np
import pytest

from formularl.geometry import arc_lengths
from formularl.periodic import (
    Correspondence,
    ProjectionSettings,
    circular_gaps,
    close_reference,
    observation_weights,
    project_ordered,
    wrap_arc,
)


def project(
    vertices: np.ndarray, samples: np.ndarray, **settings: float
) -> Correspondence:
    return project_ordered(
        close_reference(vertices),
        samples,
        arc_lengths(samples).cumulative_horizontal,
        ProjectionSettings(**settings),
    )


def test_periodic_wrap_and_gap_across_the_origin() -> None:
    np.testing.assert_allclose(
        wrap_arc(np.array([-0.5, 0, 100, 100.5]), 100), [99.5, 0, 0, 0.5]
    )
    assert wrap_arc(np.array([-np.finfo(float).tiny]), 100)[0] < 100
    starts, gaps = circular_gaps(np.array([10.0, 30, 80]), 100)
    np.testing.assert_allclose(starts, [10, 30, 80])
    np.testing.assert_allclose(gaps, [20, 50, 30])


def test_closing_chord_seam_unwrap_and_interpolated_z() -> None:
    vertices = np.array(
        [[0, 0, 10], [100, 0, 20], [100, 100, 0], [0, 100, 0]], dtype=float
    )
    samples = np.array([[0, 20, 10], [0, 0, 10], [20, 0, 12]], dtype=float)
    match = project(vertices, samples)

    assert match.valid.all()
    assert match.segment[0] == 3
    np.testing.assert_allclose(match.xyz[:, 2], [8, 10, 12])
    np.testing.assert_allclose(
        match.unwrapped_m, [-np.hypot(20, 2), 0, np.hypot(20, 2)]
    )
    assert match.arc_m[0] > match.arc_m[2] > match.arc_m[1] == 0


def test_degenerate_seam_preserves_3d_length_and_flags_vertical_ambiguity() -> None:
    square = np.array(
        [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [0, 0, 10]], dtype=float
    )
    samples = square.copy()
    reference = close_reference(square)
    assert not reference.projectable[-1]
    assert reference.period - reference.arc.cumulative_3d[-2] == pytest.approx(10)
    match = project(square, samples)
    assert match.status[0] == match.status[-1] == "ambiguous projection"
    assert match.valid[1:-1].all()
    # A repeated terminal vertex at the same elevation represents the same seam.
    square[-1, 2] = samples[-1, 2] = 0
    flat_match = project(square, samples)
    assert flat_match.valid.all()
    assert flat_match.arc_m[-1] == 0
    assert flat_match.unwrapped_m[-1] == pytest.approx(400)


def test_order_resolves_nearby_branch_and_reports_an_unresolved_tie() -> None:
    rectangle = np.array([[0, 0, 0], [100, 0, 0], [100, 2, 0], [0, 2, 0]], dtype=float)
    samples = np.array(
        [[0, 0, 0], [30, 0, 0], [60, 0, 0], [70, 1.2, 0], [85, 0, 0]], dtype=float
    )
    match = project(rectangle, samples)
    assert match.valid.all() and match.order_resolved[3]
    assert match.segment[3] == 0
    ambiguous = project(rectangle, samples, step_tolerance_m=100)
    assert ambiguous.status[3] == "ambiguous projection"
    assert np.isnan(ambiguous.arc_m[3]) and ambiguous.valid[-1]


def test_order_reports_rejected_backtrack_without_changing_observations() -> None:
    square = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=float)
    samples = np.array(
        [[5, 0, 0], [15, 0, 0], [30, 0, 0], [10, 0, 0], [55, 0, 0]], dtype=float
    )
    original = samples.copy()
    match = project(square, samples, step_tolerance_m=60)
    assert match.status[3] == "progress discontinuity" and match.order_alert[3]
    assert match.valid[-1]
    np.testing.assert_array_equal(samples, original)


def test_weights_split_seam_sites_and_cap_sparse_support() -> None:
    arc = np.array([0.05, 99.95, 20, 60, np.nan])
    drivers = np.array(["VER", "LEC", "LEC", "LEC", "LEC"])
    weights = observation_weights(arc, drivers, 100)
    np.testing.assert_allclose(weights.uniform, [0.25, 0.25, 0.25, 0.25, 0])
    np.testing.assert_allclose(weights.per_driver, [0.5, 1 / 6, 1 / 6, 1 / 6, 0])
    np.testing.assert_allclose(weights.support_m, [15, 15, 30, 40, 0])
    np.testing.assert_allclose(weights.density, [0.15, 0.15, 0.30, 0.40, 0])
    np.testing.assert_array_equal(weights.site_size, [2, 2, 1, 1, 0])
    sparse = observation_weights(np.r_[np.arange(7), 50.0], np.repeat("VER", 8), 100)
    assert sparse.support_cap_m == pytest.approx(3)
    assert sparse.support_m.max() > sparse.capped_support_m.max() == pytest.approx(3)
    assert sparse.density.sum() == pytest.approx(1)
