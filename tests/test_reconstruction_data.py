import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from formularl.dataset import SPLITS
from formularl.reconstruction_data import REQUIRED_COLUMNS, load_reconstruction


def test_loader_checks_reconstruction_files_without_opening_heldout_files(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": 2,
        "session": {},
        "selection": {
            f"{split}_drivers": list(drivers) for split, drivers in SPLITS.items()
        },
        "laps": [],
    }
    for split, drivers in SPLITS.items():
        for driver in drivers:
            entry = {"split": split, "driver": driver}
            manifest["laps"].append(entry)
            if split == "reconstruction":
                row = dict.fromkeys(sorted(REQUIRED_COLUMNS), 0)
                row.update(driver=driver, split=split, lap_number=1)
                csv = pd.DataFrame([row]).to_csv(index=False).encode()
                path = tmp_path / split / driver / "position.csv"
                path.parent.mkdir(parents=True)
                path.write_bytes(csv)
                entry.update(
                    status="prepared",
                    lap_number=1,
                    position={
                        "csv": path.relative_to(tmp_path).as_posix(),
                        "sha256": sha256(csv).hexdigest(),
                        "prepared_samples": 1,
                    },
                )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded, provenance = load_reconstruction(tmp_path)
    assert tuple(loaded) == SPLITS["reconstruction"]
    assert provenance["held_out_drivers"] == list(SPLITS["validation"])

    (tmp_path / "reconstruction/VER/position.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="VER position CSV checksum mismatch"):
        load_reconstruction(tmp_path)
