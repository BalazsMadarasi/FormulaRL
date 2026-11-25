"""Read just the checksummed reconstruction position split from Task 3C."""

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from formularl.dataset import SPLITS

XYZ = ["x_m", "y_m", "z_m"]
ARC_COLUMNS = ["arc_horizontal_m", "arc_3d_m"]
IDENTITY = ["split", "driver", "lap_number", "sample_index"]
REQUIRED_COLUMNS = {
    *XYZ,
    *ARC_COLUMNS,
    *IDENTITY,
    "flag_invalid_coordinates",
    "flag_invalid_interval_time",
    "flag_degenerate_segment",
    "flag_short_horizontal_chord",
    "flag_large_spatial_gap",
    "flag_severe_spatial_gap",
    "flag_position_speed_high",
    "flag_persistent_residual",
    "flag_recurrent_reference_region",
}


def load_reconstruction(root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Validate manifest assignments before opening any of the three position files."""
    root = root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema_version"] != 2:
        raise ValueError("Task 4 requires prepared dataset schema version 2.")

    names = list(SPLITS["reconstruction"])
    if any(
        manifest["selection"][f"{split}_drivers"] != list(drivers)
        for split, drivers in SPLITS.items()
    ):
        raise ValueError(
            "Manifest split assignments differ from the prepared dataset contract."
        )
    entries = manifest["laps"]
    assignments = [(item["split"], item["driver"]) for item in entries]
    expected = {
        (split, driver) for split, drivers in SPLITS.items() for driver in drivers
    }
    if len(assignments) != len(expected) or set(assignments) != expected:
        raise ValueError(
            "Manifest lap entries have missing, duplicate or incorrect assignments."
        )

    prepared = {
        item["driver"]: item for item in entries if item["split"] == "reconstruction"
    }
    frames = {}
    sources = {}
    for driver in names:
        item = prepared[driver]
        if item["status"] not in {"prepared", "prepared_with_limits"}:
            raise ValueError(f"{driver} has no prepared reconstruction lap.")
        asset = item["position"]
        name = f"reconstruction/{driver}/position.csv"
        if asset["csv"] != name or (root / name).resolve() != root / name:
            raise ValueError(
                f"{driver} position path does not identify its reconstruction CSV."
            )
        raw = (root / name).read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != asset["sha256"]:
            raise ValueError(f"{driver} position CSV checksum mismatch.")
        frame = pd.read_csv(BytesIO(raw), float_precision="round_trip")
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{driver} position CSV missing columns: {', '.join(sorted(missing))}."
            )
        if (
            not len(frame)
            or len(frame) != asset["prepared_samples"]
            or not frame["split"].eq("reconstruction").all()
            or not frame["driver"].eq(driver).all()
            or item["lap_number"] is None
            or not frame["lap_number"].eq(item["lap_number"]).all()
            or not np.array_equal(frame["sample_index"], np.arange(len(frame)))
        ):
            raise ValueError(
                f"{driver} CSV identity, sample count or native order differs from the manifest."
            )
        for field in [*XYZ, *ARC_COLUMNS]:
            frame[field] = pd.to_numeric(frame[field], errors="raise").astype(float)
        frames[driver] = frame
        sources[driver] = {
            "path": name,
            "sha256": digest,
            "lap_number": item["lap_number"],
            "samples": len(frame),
        }

    provenance = {
        "dataset_schema_version": manifest["schema_version"],
        "dataset_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "session": manifest["session"],
        "reconstruction_position_files": sources,
        "held_out_drivers": list(SPLITS["validation"]),
    }
    return frames, provenance
