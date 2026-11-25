"""Export native Spa position and car telemetry with traceable quality flags."""

import json
from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import fastf1
import numpy as np
import pandas as pd
from fastf1.core import Lap, Laps

from formularl.dataset import (
    CAR_OPTIONAL,
    CAR_REQUIRED,
    DEFAULT_THRESHOLDS,
    POSITION_REQUIRED,
    RECURRENCE_ARC_M,
    REFERENCE_DRIVER,
    SPLITS,
    QualityThresholds,
    annotate_position,
    car_table,
    dataset_paths,
    identify_samples,
    position_table,
)
from formularl.geometry import (
    ArcLengths,
    FloatArray,
    align_to_reference,
    closure_distances,
)
from formularl.synchronization import (
    RECOVERY_WINDOWS_S,
    integrate_speed,
    persistent_residual,
    recovery_residuals,
    time_at_reference_arc,
)
from formularl.telemetry import (
    CAR_PADDING_SAMPLES,
    EVENT,
    SEASON,
    SESSION_CODE,
    SPA_LENGTH_M,
    load_laps,
    read_car,
    read_raw_position,
    select_fastest_valid_lap,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "datasets"
XYZ = ["x_m", "y_m", "z_m"]


@dataclass
class PreparedLap:
    split: str
    driver: str
    lap: Lap | None
    start: pd.Timedelta
    end: pd.Timedelta
    position_columns: list[str]
    car_columns: list[str]
    raw_position_count: int
    raw_car_count: int
    position: pd.DataFrame
    car: pd.DataFrame
    arc: ArcLengths


def prepare_lap(
    laps: Laps, split: str, driver: str, thresholds: QualityThresholds
) -> PreparedLap:
    lap = select_fastest_valid_lap(laps, driver)
    position = read_raw_position(lap) if lap is not None else None
    car = read_car(lap, require_channels=False) if lap is not None else None
    start, end = (
        (lap["LapStartTime"], lap["Time"]) if lap is not None else (pd.NaT, pd.NaT)
    )
    position = position if position is not None else pd.DataFrame()
    car = car if car is not None else pd.DataFrame()
    position_normalized, arc = position_table(position, start, end)
    return PreparedLap(
        split,
        driver,
        lap,
        start,
        end,
        list(position.columns),
        list(car.columns),
        len(position),
        len(car),
        position_normalized,
        car_table(car, start, end, thresholds),
        arc,
    )


def projected_arc(lap: PreparedLap, reference: PreparedLap) -> FloatArray:
    if lap is reference:
        return lap.arc.cumulative_3d
    alignment = align_to_reference(
        reference.position[XYZ].to_numpy(),
        reference.arc,
        lap.position[XYZ].to_numpy(),
        lap.arc,
    )
    return np.where(alignment.matched, alignment.reference_arc, np.nan)


def number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def observed_range(values: pd.Series) -> list[float] | None:
    valid = values[np.isfinite(values)]
    return [float(valid.min()), float(valid.max())] if len(valid) else None


def describe_stream(
    table: pd.DataFrame,
    raw_count: int,
    source_columns: list[str],
    required: tuple[str, ...],
) -> dict:
    if len(table) != raw_count:
        raise ValueError("Prepared sample count differs from the raw stream.")
    flags = table.filter(regex=r"^flag_")
    return {
        "source_columns": source_columns,
        "missing_required_columns": [
            column for column in required if column not in source_columns
        ],
        "raw_samples": raw_count,
        "prepared_samples": len(table),
        "rejected_samples": 0,
        "rejection_reasons": {},
        "within_recorded_lap": int(table["within_recorded_lap"].sum()),
        "session_time_range_s": observed_range(table["session_time_s"]),
        "lap_time_range_s": observed_range(table["lap_time_s"]),
        "flag_counts": {name: int(values.sum()) for name, values in flags.items()},
        "flag_unavailable_counts": {
            name: int(values.isna().sum())
            for name, values in flags.items()
            if values.hasnans
        },
    }


def summarize_lap(lap: PreparedLap, thresholds: QualityThresholds) -> dict:
    start, end = lap.start.total_seconds(), lap.end.total_seconds()
    position, car, arc = lap.position, lap.car, lap.arc
    position_time = position["session_time_s"].to_numpy()
    car_time, speed = car["session_time_s"].to_numpy(), car["speed_mps"].to_numpy()
    horizontal = arc.cumulative_horizontal[-1] if len(position) else np.nan
    spatial = arc.cumulative_3d[-1] if len(position) else np.nan
    _, _, closing_chord = closure_distances(position[XYZ].to_numpy())
    elevation_range = observed_range(position["z_m"])
    match_start, match_end = (
        position_time[[0, -1]] if len(position) else (np.nan, np.nan)
    )
    matched, full_lap = integrate_speed(
        car_time, speed, np.array([match_start, start]), np.array([match_end, end])
    )
    mapped = time_at_reference_arc(
        position_time,
        position["ver_reference_arc_3d_m"].to_numpy(),
        np.array([RECURRENCE_ARC_M]),
    )
    one, two = recovery_residuals(
        position_time, arc.cumulative_3d, car_time, speed, mapped
    )
    recurrence_supported = np.isfinite(one[0]) and np.isfinite(two[0])

    position_summary = describe_stream(
        position, lap.raw_position_count, lap.position_columns, POSITION_REQUIRED
    )
    car_summary = describe_stream(car, lap.raw_car_count, lap.car_columns, CAR_REQUIRED)
    car_summary["missing_optional_columns"] = [
        name for name in CAR_OPTIONAL if name not in lap.car_columns
    ]
    comparable = int(position["synchronization_status"].eq("compared").sum())
    issues = []
    if lap.lap is None:
        status = "unavailable"
        issues.append("No lap met the existing fastest-valid-lap criteria.")
    else:
        if len(position) < 2 or position_summary["missing_required_columns"]:
            issues.append(
                "Position stream has fewer than two rows or missing required channels."
            )
        if len(car) < 2 or car_summary["missing_required_columns"]:
            issues.append(
                "Car stream has fewer than two rows or missing required channels."
            )
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            issues.append("Recorded lap boundary timestamps are invalid.")
        if not np.isfinite(spatial) or spatial <= 0:
            issues.append("A complete, positive observed 3D arc is unavailable.")
        if comparable == 0:
            issues.append("No consecutive position interval can be synchronized.")
        status = "unsuitable" if issues else "prepared"
        if not issues and (comparable < len(position) - 1 or not np.isfinite(full_lap)):
            status = "prepared_with_limits"
            issues.append(
                "Car synchronization has partial coverage. See interval status and car distance."
            )
    return {
        "split": lap.split,
        "driver": lap.driver,
        "lap_number": int(lap.lap["LapNumber"]) if lap.lap is not None else None,
        "lap_time_s": number(lap.lap["LapTime"].total_seconds())
        if lap.lap is not None
        else None,
        "official_start_session_s": number(start),
        "official_end_session_s": number(end),
        "status": status,
        "issues": issues,
        "position": position_summary,
        "car": car_summary,
        "geometry": {
            "elevation_range_m": elevation_range,
            "elevation_span_m": elevation_range[1] - elevation_range[0]
            if elevation_range
            else None,
            "arc_3d_coverage_m": observed_range(position["arc_3d_m"]),
            "observed_horizontal_m": number(horizontal),
            "observed_3d_m": number(spatial),
            "estimated_closed_3d_m": number(spatial + closing_chord),
            "reference_aligned_samples": int(
                position["ver_reference_arc_3d_m"].notna().sum()
            ),
        },
        "synchronization": {
            "compared_intervals": comparable,
            "position_intervals": max(len(position) - 1, 0),
            "interval_status_counts": {
                key: int(value)
                for key, value in position["synchronization_status"]
                .value_counts()
                .items()
            },
            "car_distance_over_position_m": number(matched),
            "car_distance_over_official_lap_m": number(full_lap),
            "position_minus_car_over_position_m": number(spatial - matched),
            "persistent_high_speed_intervals": int(
                (
                    position["flag_persistent_residual"]
                    & position["flag_position_speed_high"]
                ).sum()
            ),
            "at_recurring_reference_location": {
                "session_time_s": number(mapped[0]),
                "residual_1s_m": number(one[0]),
                "residual_2s_m": number(two[0]),
                "persistent": bool(
                    persistent_residual(one, two, thresholds.persistent_residual_m)[0]
                )
                if recurrence_supported
                else None,
            },
        },
    }


def write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_lap(lap: PreparedLap, output: Path, summary: dict) -> None:
    paths = dataset_paths(output, lap.split, lap.driver)
    for stream, frame, path in zip(
        ("position", "car"), (lap.position, lap.car), paths, strict=True
    ):
        samples = identify_samples(frame, lap.split, lap.driver, summary["lap_number"])
        contents = samples.to_csv(index=False, lineterminator="\n")
        write_atomic(path, contents)
        summary[stream]["csv"] = path.relative_to(output).as_posix()
        summary[stream]["sha256"] = sha256(contents.encode("utf-8")).hexdigest()


def fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "unavailable"


def span(values: list[float] | None) -> str:
    return f"{fmt(values[0])} to {fmt(values[1])}" if values else "unavailable"


def quality_report(entries: list[dict]) -> str:
    lines = [f"{EVENT} | dataset preparation", ""]
    for item in entries:
        label = (
            f"{item['driver']} lap {item['lap_number']}"
            if item["lap_number"] is not None
            else item["driver"]
        )
        lines.append(
            f"{item['split']} | {label} | {fmt(item['lap_time_s'])} s | {item['status']}"
        )
        lines.extend(f"  {issue}" for issue in item["issues"])
        lines.append(
            f"  Recorded session clock {fmt(item['official_start_session_s'])}"
            f" to {fmt(item['official_end_session_s'])} s"
        )
        for stream in ("position", "car"):
            data = item[stream]
            lines.append(
                f"  {stream}: {data['raw_samples']} raw / {data['prepared_samples']} prepared"
                f" / {data['rejected_samples']} rejected, {data['within_recorded_lap']} inside lap"
            )
            lines.append(
                f"    Session {span(data['session_time_range_s'])} s"
                f" | relative to lap start {span(data['lap_time_range_s'])} s"
            )
            flags = ", ".join(
                f"{key.removeprefix('flag_')}={count}"
                for key, count in data["flag_counts"].items()
                if count
            )
            lines.append(f"    True flags: {flags or 'none'}")
            unavailable = ", ".join(
                f"{key.removeprefix('flag_')}={count}"
                for key, count in data["flag_unavailable_counts"].items()
            )
            if unavailable:
                lines.append(f"    Blank interval flags: {unavailable}")
            if data["missing_required_columns"]:
                lines.append(
                    f"    Missing required channels: {', '.join(data['missing_required_columns'])}"
                )
            if data.get("missing_optional_columns"):
                lines.append(
                    f"    Missing optional channels: {', '.join(data['missing_optional_columns'])}"
                )
            for reason, count in data["rejection_reasons"].items():
                lines.append(f"    Rejected {count} rows: {reason}")
        geometry, sync = item["geometry"], item["synchronization"]
        lines.append(
            f"  Elevation {span(geometry['elevation_range_m'])} m (span {fmt(geometry['elevation_span_m'])} m)"
            f" | observed XY {fmt(geometry['observed_horizontal_m'])} m"
            f" | observed 3D {fmt(geometry['observed_3d_m'])} m | closed estimate {fmt(geometry['estimated_closed_3d_m'])} m"
        )
        lines.append(
            f"  Known 3D arc {span(geometry['arc_3d_coverage_m'])} m"
            f" | VER alignment {geometry['reference_aligned_samples']}/{item['position']['prepared_samples']} rows"
            f" | synchronized {sync['compared_intervals']}/{sync['position_intervals']} intervals"
        )
        lines.append(
            f"  Car distance over position / official lap"
            f" {fmt(sync['car_distance_over_position_m'])} / {fmt(sync['car_distance_over_official_lap_m'])} m"
            f" | position minus matched car {fmt(sync['position_minus_car_over_position_m'])} m"
        )
        recurrent = sync["at_recurring_reference_location"]
        lines.append(
            f"  VER s={RECURRENCE_ARC_M:.1f} m: 1 s / 2 s residuals"
            f" {fmt(recurrent['residual_1s_m'])} / {fmt(recurrent['residual_2s_m'])} m"
            f" | persistent: {recurrent['persistent'] if recurrent['persistent'] is not None else 'unavailable'}"
        )
        lines.append("")

    lines.append("Split comparison (quality and coverage only)")
    for split in SPLITS:
        group = [entry for entry in entries if entry["split"] == split]
        lengths = [
            entry["geometry"]["observed_3d_m"]
            for entry in group
            if entry["geometry"]["observed_3d_m"] is not None
        ]
        elevations = [
            entry["geometry"]["elevation_span_m"]
            for entry in group
            if entry["geometry"]["elevation_span_m"] is not None
        ]
        pos_n = sum(entry["position"]["prepared_samples"] for entry in group)
        car_n = sum(entry["car"]["prepared_samples"] for entry in group)
        high = sum(
            entry["position"]["flag_counts"]["flag_position_speed_high"]
            for entry in group
        )
        persistent_high = sum(
            entry["synchronization"]["persistent_high_speed_intervals"]
            for entry in group
        )
        length_range = span([min(lengths), max(lengths)] if lengths else None)
        elevation_range = span(
            [min(elevations), max(elevations)] if elevations else None
        )
        lines.append(
            f"  {split}: {pos_n} position rows, {car_n} car rows"
            f" | observed 3D lengths {length_range} m | elevation spans {elevation_range} m"
            f" | high position speed {high} intervals ({persistent_high} persistent)"
        )
    return "\n".join(lines) + "\n"


def manifest_for(
    laps: Laps, entries: list[dict], thresholds: QualityThresholds, offline: bool
) -> dict:
    session = laps.session
    return {
        "schema_version": 2,
        "session": {
            "year": SEASON,
            "event": session.event["EventName"],
            "round": int(session.event["RoundNumber"]),
            "name": session.name,
            "code": SESSION_CODE,
            "fastf1_date": str(session.date),
            "official_circuit_length_m": SPA_LENGTH_M,
        },
        "provenance": {
            "source": "FastF1 position and car telemetry",
            "fastf1_version": fastf1.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "fastf1_session_path": session.api_path,
            "cache_only": offline,
            "position_method": "Lap.get_pos_data",
            "position_padding_rows_per_side": 0,
            "car_method": "Lap.get_car_data",
            "car_padding_rows_per_side": CAR_PADDING_SAMPLES,
            "fastf1_edge_interpolation": False,
            "position_units_from_source": "decimetres",
            "speed_units_from_source": "km/h",
        },
        "selection": {
            "criterion": "Shortest recorded lap time per driver, stable FastF1 row order for ties.",
            "required": [
                "Deleted=False",
                "IsAccurate=True",
                "PitInTime missing",
                "PitOutTime missing",
                "LapTime present",
            ],
            "reconstruction_drivers": SPLITS["reconstruction"],
            "validation_drivers": SPLITS["validation"],
            "validation_policy": "Held out from future reconstruction fitting; fixed VER alignment for diagnostics only.",
        },
        "units": {
            "normalized_position_and_distance": "m",
            "normalized_time": "s",
            "normalized_speed": "m/s",
            "primary_spatial_coordinate": "arc_3d_m",
            "secondary_spatial_coordinate": "arc_horizontal_m",
            "raw_timestamp_encoding": "FastF1 Date, SessionTime and slice Time as text (date/time or duration)",
            "throttle_pct": "%",
            "brake": "FastF1 boolean",
            "rpm": "rev/min",
            "gear": "gear number",
            "drs": "uninterpreted FastF1 DRS state code",
        },
        "quality": {
            "thresholds": asdict(thresholds),
            "interval_row": "Ending sample, zero-based sample_index; first-row interval fields are blank.",
            "duplicate_timestamp": "Flags every row sharing a finite session timestamp.",
            "invalid_interval_time": "Flags position intervals with non-finite or non-positive duration.",
            "persistent_residual_windows_s": RECOVERY_WINDOWS_S,
            "persistent_residual_rule": "Signed position-minus-car residual exceeds the magnitude threshold in both centered windows and has the same sign.",
            "reference_driver": REFERENCE_DRIVER,
            "recurring_residual_center_3d_m": RECURRENCE_ARC_M,
            "recurrence_rule": "A supported forward projected interval whose VER arc span intersects the diagnostic band.",
            "retention": "Every returned raw row is retained in original order, including invalid or flagged rows.",
        },
        "known_limitations": [
            "Raw refers to the FastF1 slices after upstream parsing; unavailable source samples may be absent or zero-filled.",
            "Raw slice Time follows FastF1's lap slicing. Numeric timestamps use SessionTime and the recorded lap start.",
            "Arc lengths follow observed consecutive chords and become unavailable after a missing chord. The closed estimate adds a straight closing chord for reporting.",
            "Absolute circuit geometry remains unverified. Position quantization and sparse samples limit local derivatives.",
            "Time-window residuals use linear chord allocation and trapezoidal speed integration, without extrapolation or timestamp shifts.",
            "Reference alignment uses VER's sampled path and can leave arc correspondence unavailable.",
            "Temporal flags do not reject geometry. No smoothing, speed clipping or rescaling is applied.",
        ],
        "laps": entries,
    }


def main(
    *,
    offline: bool = False,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
    output: Path = OUTPUT,
) -> int:
    laps = load_laps(ROOT / "cache", offline=offline)
    if laps is None:
        return 1
    prepared = [
        prepare_lap(laps, split, driver, thresholds)
        for split, drivers in SPLITS.items()
        for driver in drivers
    ]
    reference = next(
        item
        for item in prepared
        if item.driver == REFERENCE_DRIVER and item.split == "reconstruction"
    )
    reference_available = (
        len(reference.position) >= 2
        and np.isfinite(reference.arc.cumulative_3d[-1])
        and reference.arc.cumulative_3d[-1] > 0
    )
    entries = []
    for item in prepared:
        mapped = (
            projected_arc(item, reference)
            if reference_available
            else np.full(len(item.position), np.nan)
        )
        annotate_position(
            item.position,
            item.arc,
            item.car,
            mapped,
            item.start.total_seconds(),
            thresholds,
        )
        summary = summarize_lap(item, thresholds)
        if not reference_available:
            summary["issues"].append(
                "VER reference is unavailable; aligned flags are left blank."
            )
            if summary["status"] == "prepared":
                summary["status"] = "prepared_with_limits"
        write_lap(item, output, summary)
        entries.append(summary)
    report = quality_report(entries)
    write_atomic(output / "quality_report.txt", report)
    manifest = manifest_for(laps, entries, thresholds, offline)
    write_atomic(
        output / "manifest.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(report, end="")
    print(f"Datasets and manifest saved to {output}")
    return int(any(item["status"] in ("unavailable", "unsuitable") for item in entries))


def positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise ArgumentTypeError("expected a finite number greater than zero")
    return value


if __name__ == "__main__":
    defaults = DEFAULT_THRESHOLDS
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Only use the local FastF1 cache."
    )
    parser.add_argument(
        "--short-chord-m", type=positive_float, default=defaults.short_horizontal_m
    )
    parser.add_argument(
        "--gap-thresholds-m",
        type=positive_float,
        nargs=2,
        default=(defaults.large_gap_m, defaults.severe_gap_m),
        metavar=("LARGE", "SEVERE"),
    )
    parser.add_argument(
        "--speed-threshold-kmh", type=positive_float, default=defaults.speed_kmh
    )
    parser.add_argument(
        "--persistent-residual-m",
        type=positive_float,
        default=defaults.persistent_residual_m,
    )
    parser.add_argument(
        "--recurrence-radius-m",
        type=positive_float,
        default=defaults.recurrence_radius_m,
    )
    args = parser.parse_args()
    large, severe = args.gap_thresholds_m
    if large >= severe:
        parser.error(
            "the large gap threshold must be smaller than the severe gap threshold"
        )
    raise SystemExit(
        main(
            offline=args.offline,
            thresholds=QualityThresholds(
                short_horizontal_m=args.short_chord_m,
                large_gap_m=large,
                severe_gap_m=severe,
                speed_kmh=args.speed_threshold_kmh,
                persistent_residual_m=args.persistent_residual_m,
                recurrence_radius_m=args.recurrence_radius_m,
            ),
        )
    )
