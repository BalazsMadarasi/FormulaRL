from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from formularl.dataset import (
    DEFAULT_THRESHOLDS,
    SPLITS,
    annotate_position,
    car_table,
    dataset_paths,
    position_table,
)

START, END = pd.to_timedelta(np.array([100, 104]), unit="s")


def test_si_conversion_preserves_raw_values_and_clock() -> None:
    raw_position = pd.DataFrame(
        {
            "X": [123, 153],
            "Y": [-20, 20],
            "Z": [3655, 3655],
            "Date": pd.to_datetime(
                ["2024-07-27 14:00:00.000000123", "2024-07-27 14:00:00.5"],
                format="mixed",
            ),
            "SessionTime": pd.to_timedelta([100.0, 100.5], unit="s"),
            "Time": pd.to_timedelta([0.0, 0.5], unit="s"),
        }
    )
    snapshot = raw_position.copy(deep=True)
    position, arc = position_table(raw_position, START, END)
    car = car_table(
        pd.DataFrame(
            {
                "SessionTime": pd.to_timedelta([99.75, 100.25], unit="s"),
                "Speed": [180, 216],
                "DRS": [0, 42],
            }
        ),
        START,
        END,
        DEFAULT_THRESHOLDS,
    )

    pd.testing.assert_frame_equal(raw_position, snapshot)
    np.testing.assert_allclose(position["x_m"], [12.3, 15.3])
    np.testing.assert_allclose(position["z_m"], [365.5, 365.5])
    np.testing.assert_allclose(arc.cumulative_3d, [0, 5])
    np.testing.assert_allclose(position["lap_time_s"], [0, 0.5])
    assert position["raw_x_dm"].tolist() == [123, 153]
    assert pd.Timestamp(position["raw_date"].iloc[0]) == raw_position["Date"].iloc[0]
    assert (
        pd.to_timedelta(position["raw_session_time"]).iloc[1]
        == raw_position["SessionTime"].iloc[1]
    )
    assert (
        pd.to_timedelta(position["raw_slice_time"]).iloc[1]
        == raw_position["Time"].iloc[1]
    )
    assert car["raw_speed_kmh"].tolist() == [180, 216]
    assert car["drs"].tolist() == [0, 42]
    assert not car["flag_invalid_optional_channel"].any()
    np.testing.assert_allclose(car["speed_mps"], [50, 60])
    np.testing.assert_allclose(car["lap_time_s"], [-0.25, 0.25])
    assert car["within_recorded_lap"].tolist() == [False, True]


def test_quality_flags_leave_duplicate_and_invalid_rows_in_place() -> None:
    raw = pd.DataFrame(
        {
            "X": [0, 10, 10, 670, 680, np.nan, 1000],
            "Y": 0,
            "Z": 0,
            "SessionTime": pd.to_timedelta(
                [100, 100.25, 100.25, 100.5, 100.75, 101, 101.25], unit="s"
            ),
        }
    )
    position, arc = position_table(raw, START, END)
    car = car_table(
        pd.DataFrame(
            {
                "SessionTime": pd.to_timedelta([100.25, 100.75], unit="s"),
                "Speed": [180, 180],
            }
        ),
        START,
        END,
        DEFAULT_THRESHOLDS,
    )
    aligned = np.array([2100.0, 2110, 2110, 2180, 2200, np.nan, np.nan])
    annotate_position(position, arc, car, aligned, 100.0, DEFAULT_THRESHOLDS)

    assert car["drs"].isna().all()
    assert len(position) == len(raw)
    assert position["flag_duplicate_timestamp"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert position.loc[2, "flag_nonpositive_timestamp_step"]
    assert position.loc[2, "flag_degenerate_segment"]
    assert position.loc[1, "flag_short_horizontal_chord"]
    assert (
        position.loc[3, "flag_severe_spatial_gap"]
        and position.loc[3, "flag_large_spatial_gap"]
    )
    assert position.loc[3, "flag_position_speed_high"]
    assert position.loc[1, "synchronization_status"] == "outside car coverage"
    assert position.loc[3, "flag_recurrent_reference_region"]
    assert (
        position.loc[5, "flag_invalid_coordinates"]
        and position.loc[6, "flag_invalid_chord"]
    )
    assert pd.isna(position.loc[6, "flag_position_speed_high"])
    assert position["flag_persistent_residual"].isna().all()
    assert position["arc_3d_m"].iloc[4] == pytest.approx(68.0)
    assert position["arc_3d_m"].iloc[5:].isna().all()


def test_invalid_position_interval_time_flags_zero_negative_and_missing_steps() -> None:
    raw = pd.DataFrame(
        {
            "X": np.arange(7) * 10,
            "Y": 0,
            "Z": 0,
            "SessionTime": pd.to_timedelta(
                [100.0, 101.0, 101.0, 100.5, np.nan, 103.0, 104.0], unit="s"
            ),
        }
    )
    position, arc = position_table(raw, START, END)
    car = car_table(pd.DataFrame(), START, END, DEFAULT_THRESHOLDS)
    annotate_position(position, arc, car, arc.cumulative_3d, 100.0, DEFAULT_THRESHOLDS)

    invalid = position["flag_invalid_interval_time"]
    assert pd.isna(invalid.iloc[0])
    assert invalid.iloc[1:].tolist() == [False, True, True, True, True, False]
    assert np.flatnonzero(position["flag_duplicate_timestamp"]).tolist() == [1, 2]
    assert np.flatnonzero(position["flag_nonpositive_timestamp_step"]).tolist() == [
        2,
        3,
    ]
    assert np.flatnonzero(position["flag_invalid_session_time"]).tolist() == [4]


def test_persistence_applies_to_supported_windows_even_without_speed_spikes() -> None:
    timestamps = pd.to_timedelta(np.arange(100, 105), unit="s")
    position, arc = position_table(
        pd.DataFrame(
            {"X": np.arange(5) * 200, "Y": 0, "Z": 0, "SessionTime": timestamps}
        ),
        START,
        END,
    )
    car = car_table(
        pd.DataFrame({"SessionTime": timestamps, "Speed": 36}),
        START,
        END,
        DEFAULT_THRESHOLDS,
    )
    annotate_position(position, arc, car, arc.cumulative_3d, 100.0, DEFAULT_THRESHOLDS)

    assert position["flag_persistent_residual"].sum() == 2
    assert position["flag_position_speed_high"].sum() == 0
    assert position.loc[2, "residual_1s_m"] == pytest.approx(10)
    assert position.loc[2, "residual_2s_m"] == pytest.approx(20)
    assert pd.isna(position.loc[1, "flag_persistent_residual"])


def test_dataset_paths_keep_the_validation_split_separate(tmp_path: Path) -> None:
    assert SPLITS["reconstruction"] == ("VER", "LEC", "HAM")
    assert SPLITS["validation"] == ("NOR", "PIA", "RUS")
    paths = {
        path.relative_to(tmp_path).as_posix()
        for split, drivers in SPLITS.items()
        for driver in drivers
        for path in dataset_paths(tmp_path, split, driver)
    }
    assert len(paths) == 12
    assert "reconstruction/VER/position.csv" in paths
    assert "validation/NOR/car.csv" in paths
    with pytest.raises(ValueError, match="not assigned"):
        dataset_paths(tmp_path, "reconstruction", "NOR")
