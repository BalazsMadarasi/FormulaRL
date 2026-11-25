# FormulaRL

Task 1 investigates the position telemetry for the fastest valid qualifying
laps of Max Verstappen (VER), Charles Leclerc (LEC), and Lewis Hamilton (HAM)
at the 2024 Belgian Grand Prix.

## Windows PowerShell setup

Run these commands from the repository root with Python 3.11 or newer. The
commands use the virtual environment's Python directly, so activation is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe .\scripts\investigate_position.py
```

Downloading uncached session data requires network access and an active F1 TV
Access, Pro, or Premium subscription. FastF1 prompts for sign-in in an
interactive terminal when needed. You can also sign in before running the
script using the command below. [FastF1 authentication guide](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/accounts_auth.rst)

```powershell
.\.venv\Scripts\python.exe -m fastf1 auth f1tv --authenticate
```

The script creates `cache/` for FastF1 and creates `outputs/` when there is
telemetry to plot. It reports unavailable lap timing or telemetry and uses a
nonzero exit code if any of the three drivers could not be analyzed. When only
some drivers have telemetry, the plots show the available drivers.

## What the script measures

The script chooses the shortest recorded lap time after excluding deleted,
inaccurate, pit-entry, and pit-exit laps. It uses FastF1's `get_pos_data()`
with zero padding and edge interpolation disabled. It preserves the returned
rows, their order, and their values, and converts X, Y, and Z from decimetres
to metres in a separate table. “Raw” refers to the samples as returned by
FastF1, including its upstream parsing and timestamp handling.
FastF1 can zero-fill coordinates for missing source samples, so a zero
missing-value count only describes the returned table.

For each lap the script prints the lap number, lap time, and sample count.
Minimum, median, and maximum intervals in milliseconds use consecutive rows
in their original order and include zero and negative intervals. The report
counts repeated occurrences after the first for each non-missing timestamp,
non-positive intervals, and missing values for each coordinate. It also gives
coordinate minima and maxima, the elevation range (maximum Z minus minimum Z),
and the median, 95th percentile, and maximum Euclidean displacement between
consecutive samples, all in metres. Pairs with missing inputs are counted as
unavailable for the affected statistic. Displacements use adjacent rows even
when their timestamps are repeated or out of order.

## Plots

| File in `outputs/` | What it shows |
| --- | --- |
| `trajectory_xy.png` | X/Y overlay with equal scaling on both axes. |
| `trajectory_3d.png` | The same samples in three dimensions with their original coordinate origin. |
| `elevation_vs_time.png` | Z on a shared coordinate scale against seconds since each lap's recorded start. |
| `sampling_intervals.png` | Interval counts with shared bins for comparing cadence, gaps, and non-positive intervals. |
