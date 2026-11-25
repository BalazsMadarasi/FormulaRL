import numpy as np

from formularl.synchronization import integrate_speed


def test_integrates_trapezoids_across_reported_samples() -> None:
    distance = integrate_speed(
        np.array([10.0, 11.0, 13.0]),
        np.array([10.0, 20.0, 30.0]),
        np.array([10.0, 11.0]),
        np.array([13.0, 13.0]),
    )
    np.testing.assert_allclose(distance, [65, 50])


def test_interpolates_both_interval_boundaries() -> None:
    distance = integrate_speed(
        np.array([10.0, 12.0, 14.0]),
        np.array([0.0, 8.0, 0.0]),
        np.array([10.5, 11.0]),
        np.array([11.5, 13.0]),
    )
    np.testing.assert_allclose(distance, [4, 12])


def test_unavailable_intervals_do_not_extrapolate_or_bridge_invalid_speed() -> None:
    distance = integrate_speed(
        np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
        np.array([10.0, 10.0, np.nan, 10.0, 10.0]),
        np.array([9.5, 13.5, 10.0, 11.5, 13.0, 13.0]),
        np.array([11.0, 14.5, 11.0, 12.5, 14.0, 13.0]),
    )
    np.testing.assert_allclose(distance, [np.nan, np.nan, 10, np.nan, 10, np.nan])


def test_duplicate_timestamp_leaves_neighboring_spans_unsupported() -> None:
    distance = integrate_speed(
        np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0]),
        np.array([10.0, 10.0, 15.0, 20.0, 10.0, 10.0]),
        np.array([0.0, 1.2, 2.2, 3.0]),
        np.array([1.0, 1.8, 2.8, 4.0]),
    )
    np.testing.assert_allclose(distance, [10, np.nan, np.nan, 10])
