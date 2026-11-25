# FormulaRL

The investigation uses the fastest valid qualifying laps of Max Verstappen
(VER), Charles Leclerc (LEC), and Lewis Hamilton (HAM) at the 2024 Belgian Grand
Prix. It reports the raw position telemetry diagnostics and measures the
geometry of the observed lap polylines.

## Windows PowerShell setup

Run these commands from the repository root with Python 3.11 or newer. The
commands use the virtual environment's Python directly, so activation is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe -m pytest
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

Use `--offline` to run against an already populated FastF1 cache.

```powershell
.\.venv\Scripts\python.exe .\scripts\investigate_position.py --offline
```

## Raw telemetry

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

## Arc length and closure

The script sums the straight chords between consecutive position samples in
XY and in XYZ. Cumulative XYZ chord length is the primary spatial coordinate,
starting at zero for the first sample. The elevation contribution is the 3D
length minus the horizontal length, as a distance and as a percentage of
the observed 3D length. Each observed length is compared with Spa's official
7,004 m circuit length. The sampled driving line, sparse chords through
corners, missing endpoint time, and telemetry errors can all affect that
comparison. The script keeps the original coordinate scale.
[Official Spa circuit length](https://www.formula1.com/en/information/belgium-circuit-de-spa-francorchamps.3LltuYaAXVRU8iezEsjzGw)

Closure uses the first and last samples. The report gives their horizontal and
3D separation, signed end-minus-start elevation, and signed sample-minus-lap
time offsets using FastF1's recorded `LapStartTime` and `Time` boundaries.
A positive start offset and negative end offset show incomplete temporal
coverage. The separate closed-length estimates add one straight endpoint
chord. This estimates closure without adding any samples or recovering the
missing path.

An undefined coordinate leaves affected adjacent chords undefined. The report
gives the available subtotal, and cumulative distance stops at the first
undefined chord in that dimension. No bridge is inserted. Segment statistics
cover the measured chord lengths, including zero-length segments.

## Spatial sampling

The thresholds are strictly greater than **40 m** and **60 m**. The latter is
a subset of the former. Each flagged line shows the segment number, cumulative
3D distance at its end, chord length, sample interval, and implied average
speed. The implied speed is chord length divided by the time interval,
converted to km/h. Speeds above **400 km/h** are flagged even for shorter
chords. Missing or non-positive intervals yield unavailable speeds. These
flags help diagnose the samples and timestamps and leave them in the data.

## Cross-driver alignment

The script projects LEC and HAM samples onto VER's XY segments with the
projection fraction clamped to the segment. That fraction gives the VER
reference Z and cumulative 3D arc length at the projected location. The report
gives horizontal, absolute vertical, and combined 3D errors, with signed
vertical bias calculated as the mean of driver Z minus reference Z.

The matching checks use horizontal geometry and progress. Zero-length XY
segments are counted and skipped. The nearest projection is excluded if it
is more than 25 m away, if another candidate is within 1 m of its distance
and over 30 m away along the horizontal reference polyline, or if it is over
150 m away from the target's normalized horizontal lap progress mapped onto
VER. Clamped start and end points and jumps to the opposite end of the lap are
reported as boundary cases. There is no lap wraparound or extrapolation.
Between successive otherwise eligible matches, the script excludes both
samples if reference progress goes backwards by more than 10 m or its step
differs from the target step by more than 30 m. It reports the affected 3D arc
positions and the number of samples used in the error statistics.

The checks are diagnostic heuristics. VER's polyline follows one sampled
driving line, and long reference chords can cut across curves or simplify
elevation changes. The reported deviations measure agreement with that
polyline. Common telemetry bias and absolute positional accuracy cannot be
determined from this comparison.

## Plots

| File in `outputs/` | What it shows |
| --- | --- |
| `trajectory_xy.png` | X/Y overlay with equal scaling on both axes. |
| `trajectory_3d.png` | Raw trajectories with one physical scale for X, Y, and Z. |
| `elevation_vs_time.png` | Z on a shared coordinate scale against seconds since each lap's recorded start. |
| `sampling_intervals.png` | Interval counts with shared bins for comparing cadence, gaps, and non-positive intervals. |
| `elevation_vs_arc_length.png` | Z against each driver's observed cumulative 3D arc length. |
| `aligned_deviations.png` | Horizontal and signed vertical deviations for eligible matches along VER's 3D arc. |
| `segment_lengths.png` | Individual 3D chord lengths at segment-end arc positions, with both gap thresholds. |
| `spatial_gaps_xy.png` | Equal-scale X/Y trajectories with the chords above the gap thresholds highlighted. |
