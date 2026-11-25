# FormulaRL

The investigation uses the fastest valid qualifying laps of Max Verstappen
(VER), Charles Leclerc (LEC), and Lewis Hamilton (HAM) at the 2024 Belgian Grand
Prix. It reports the raw position telemetry diagnostics and measures the
geometry of the observed lap polylines. A separate investigation compares
position-derived motion with the raw car speed stream. The local geometry
investigation measures elevation, grade, curvature, and anomaly recovery.
Dataset preparation packages the three reconstruction laps and three
held-out validation laps using the same diagnostic methods.

## Windows PowerShell setup

Run these commands from the repository root with Python 3.11 or newer. The
commands use the virtual environment's Python directly, so activation is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe .\scripts\investigate_position.py
.\.venv\Scripts\python.exe .\scripts\investigate_synchronization.py
.\.venv\Scripts\python.exe .\scripts\investigate_local_geometry.py
.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py
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
.\.venv\Scripts\python.exe .\scripts\investigate_synchronization.py --offline
.\.venv\Scripts\python.exe .\scripts\investigate_local_geometry.py --offline
.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py --offline
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

## Telemetry synchronization

`investigate_synchronization.py` reuses the lap selection above and fetches
raw car rows through `get_car_data()` with four sample rows of padding on each
side where available. Both streams use FastF1's shared `SessionTime`. The
plots subtract the recorded lap start from that common clock. FastF1 edge
interpolation is disabled for both raw slices. Car counts and interval
statistics include the fetched padding, and the speed range and median use
samples within the recorded lap. Negative, infinite, or above-400 km/h
reported speeds and zeros within the lap are flagged.

For every adjacent pair of position samples, the script divides 3D chord
length by elapsed time and compares that speed with the average from the car
stream over the same bounds. The car calculation converts km/h to m/s and
integrates piecewise-linear reported speed using full and partial
trapezoids. It leaves intervals unsupported when they fall outside coverage,
have invalid timing, or overlap a car span with missing, nonfinite, negative,
or duplicate-time endpoint samples. Missing or decreasing car timestamps
make the source time axis unusable. Positive reported speeds stay in the
calculation, including suspicious high values. Signed discrepancies are
position minus car.

The anomaly CSV includes every position interval above **400 km/h**, its
session time bounds, segment-end 3D arc length, displacement, matched car
speed, distance and speed discrepancies, and both neighbors' measurements.
A neighbor is unusually low when its position speed is below **75%** of its
positive matched car speed. The report compares the before and after counts
with the overall low-speed rate. Blank CSV values indicate unavailable
measurements or neighbors, and the status columns explain unsupported intervals.

The temporal comparison uses **0.5, 1.0, and 2.0 s** windows centered on native
position interval midpoints. It sums chord distance within each window, with
fractional segments assigned in proportion to time. This only distributes
distance for the diagnostic calculation. The comparison uses the same
eligible midpoints at all four scales, including the native intervals, and
does not extrapolate either stream. The diagnostic lag scan tests **−500 to
+500 ms in 20 ms steps** at native and 1.0 s scales. A positive lag means
evaluating car speed at later session times. Each scan uses the coverage
shared by all candidate lags and compares RMSE with the zero-lag result.
The analysis uses the original timestamps for every reported comparison
outside the lag scan.

The distance report compares the observed 3D polyline and car integral over
the first-to-last position time span, then separately reports the car
integral over the recorded lap, the straight-chord closure estimate, and
Spa's official 7,004 m length. Percentage differences use the stated
comparison baseline as their denominator. Neither stream is rescaled.
Both reported speed and position are sampled measurements. Long car sample
gaps require a linear speed assumption, and the position chords can shorten
curves. Local disagreement, lag fits, and integrated distances therefore
cannot establish which stream is in error or identify a timestamp fault.

## Elevation and local geometry

### Anomaly recovery

`investigate_local_geometry.py` uses the same raw slices and zero-lag
synchronization as Task 3A. For each position speed above 400 km/h it sums
signed position-minus-car distance residuals with the preceding interval,
the following interval, and both. A sum needs all of its component residuals.
It also measures the residual in exact **1 s** and **2 s** windows centered on
the anomalous interval's midpoint, using time-proportional partial position
chords and trapezoidal car distance as above.

A persistent residual is strictly greater than **5 m** in magnitude in both
windows, with the same sign. For each such anomaly, the script locates the
midpoint on VER's projected 3D arc. It maps that position back to session
time for each peer using a single forward span between adjacent accepted
projections. A peer comparison can therefore detect persistence even when
that driver had no above-threshold speed at the location. Ambiguous spans
and missing coverage produce unavailable values.

On the cached laps, the 74 anomalies had a median absolute native residual
of **9.53 m**. This fell to **0.35 m** with the following interval and **1.82 m**
with both neighbors. The 1 s and 2 s window medians were **3.31 m** and
**3.50 m**. Six anomalies met the persistence criterion. At VER arc
**2,182.1 m**, HAM and VER both had persistent negative residuals. LEC's
residuals at that location were within 5 m. Anomaly windows can overlap.
The cause is unresolved.

### Grade and curvature

Native grade is `100 × dz / horizontal chord length`. Inclination is
`atan2(dz, horizontal chord length)` in degrees. A vertical-only chord has
a signed 90° inclination and undefined grade. A zero-length chord has
undefined inclination and grade. Elevation gain and descent sum positive
and negative changes separately from the endpoint net change.

Spatial windows of **10, 25, and 50 m** are centered on native segment
midpoints along the 3D arc. Piecewise-linear interpolation at each boundary
gives the elevation and cumulative horizontal path length. The grade uses
the elevation change divided by that horizontal path length, including
bends. Windows require coverage at both ends and leave the raw samples
unchanged. A window contained inside one raw chord returns that chord's
grade. The report counts those windows so their effective support is visible.
For cross-driver comparisons, the script projects both endpoints of a
target chord, uses the mapped midpoint on VER's arc, and compares all
scales on the same eligible centers. Each driver's window is measured in
metres along its own 3D arc.

All three laps span **365.5–467.9 m**, a **102.4 m** range. The strongest
native uphill estimates were **+14.71% to +15.71%** and the strongest
downhill estimates were **−13.33% to −11.74%**. The corresponding 50 m
extremes were **+14.27% to +14.60%** and **−9.09% to −8.25%**.
**951 of 1,291** eligible 10 m windows fell inside a single native chord,
compared with **5 of 1,284** eligible 50 m windows. The 95th
percentile absolute aligned grade difference versus VER fell from
**2.447 to 0.351 percentage points** for LEC and **1.829 to 0.330** for HAM,
from native chords to 50 m windows. These diagnostics assess sampling
sensitivity and cross-driver consistency.

Horizontal curvature is the signed inverse circumcircle radius through
three consecutive XY samples. Straight triples have zero curvature.
Triples with repeated XY points are undefined. Radius is reported for
`|curvature| > 1e-12 m⁻¹`. Vertical curvature is the change in inclination
in radians divided by the 3D distance between adjacent segment midpoints.
The report examines curvature next to XY chords shorter than **2 m**,
3D chords longer than **40 m**, and elevation steps no larger than
the **0.1 m** reporting increment. For example, LEC's largest horizontal
estimate implied a **7.50 m** radius beside a **0.41 m** chord, and VER's
largest vertical estimate was **0.03411 m⁻¹** beside a **0.61 m** chord.
Short chords magnify coordinate quantization, while long chords can
hide localized changes. These are raw derivative estimates.

### Published circuit references

Pirelli reports a **102.2 m** difference between the high point at Malmedy
and the low point at Paul Frère. The telemetry range is **0.2 m larger**
on each lap. The circuit operator describes Raidillon as **17%**, while
Formula 1's corner guide describes a **15%** maximum. The operator's access
map also marks distances and spot altitudes. These sources provide context
for the overall relief and climb. A common datum, measurement line, and
averaging distance would be needed for sample-by-sample validation.
[Pirelli](https://press.pirelli.com/two-races-for-spa-this-year/) · [Circuit operator](https://www.spa-francorchamps.be/en/news/417_spa-francorchamps-cyclon-track-first-edition) · [Formula 1 corner guide](https://www.formula1.com/en/latest/article/explained-how-every-corner-at-the-circuit-de-spa-francorchamps-got-names.1o6cyXf5F6q9VFFcgMqTqs) · [Circuit access map](https://www.spa-francorchamps.be/assets/3541f0b5-47e5-483b-a5e6-059d99346600/all-plans-acces-csf.pdf)

For future reconstruction, keep the raw chords and use separate quality
flags for persistent residuals, short chords, and sparse sections. Check
recovered speed spikes over their neighboring intervals before changing
geometry. Use 25–50 m grade agreement and held-out laps to evaluate
candidate reconstruction scales, and seek a surveyed longitudinal
profile and its datum before making absolute accuracy claims.

## Prepared datasets

Run `prepare_dataset.py` from the repository root using the PowerShell
commands above. It applies the existing fastest-valid-lap criteria to
each driver. VER, LEC, and HAM are assigned to reconstruction. NOR, PIA,
and RUS are held out from future reconstruction fitting. Diagnostic
alignment projects onto the fixed VER trajectory from the reconstruction
split.

```text
outputs/datasets/
  manifest.json
  quality_report.txt
  reconstruction/{VER,LEC,HAM}/{position,car}.csv
  validation/{NOR,PIA,RUS}/{position,car}.csv
```

The manifest (schema version 2) records the session, FastF1 version and
source methods, lap criteria, split membership, thresholds, coverage, flag counts, CSV paths
and SHA-256 digests. Generated datasets are covered by the `outputs/`
Git ignore rule. The script reports any unavailable or unsuitable selected
lap, exports the rows that are available, and exits nonzero when a lap
cannot be prepared. An unavailable lap has header-only CSVs. Partial
synchronization coverage is reported separately when a usable lap remains.

| Columns | Meaning |
| --- | --- |
| `split`, `driver`, `lap_number`, `sample_index` | Assignment and identity. The index is zero-based within each native slice. |
| `raw_date`, `raw_session_time`, `raw_slice_time` | Original FastF1 `Date`, `SessionTime`, and slice `Time` values as text. |
| `session_time_s`, `lap_time_s`, `within_recorded_lap` | Seconds on the shared session clock, seconds since the recorded lap start, and recorded-lap membership. |
| `raw_x_dm`, `raw_y_dm`, `raw_z_dm`, `x_m`, `y_m`, `z_m` | Original position coordinates and their metre conversions on the same coordinate origin. The original position status and source labels are also retained. |
| `arc_3d_m`, `arc_horizontal_m`, `ver_reference_arc_3d_m` | Primary observed 3D arc, horizontal diagnostic arc, and VER-projected 3D position, where supported. |
| Position interval measurements | Duration, horizontal and 3D chord lengths, position and car speeds in m/s, matched car distance and signed position-minus-car distance residuals in metres. |
| `raw_speed_kmh`, `speed_mps`, `throttle_pct`, `brake`, `rpm`, `gear` | Original car speed and its m/s conversion, plus other reported car channels where present. |
| `drs` | Original FastF1 `DRS` state code, preserved as supplied and blank when unavailable. |

The position and car files preserve separate native sample orders and rates.
FastF1 car slices include up to four original rows on each side of the
recorded lap for boundary coverage. Use `within_recorded_lap` to distinguish
those rows. The original FastF1 slice `Time` is retained for provenance.
Normalized times use the common `SessionTime` clock and recorded lap start.
Arc lengths start at the first observed position row and stop where the
next chord becomes unavailable.

Sample flags describe invalid coordinates, timestamps, and car channels.
Duplicate-time flags mark every row sharing the timestamp. Non-positive
steps refer to the preceding sample. The position interval flag
`flag_invalid_interval_time` covers both non-finite and non-positive durations.
Geometric interval flags cover
unavailable XYZ chords, zero horizontal run, short XY chords under **2 m**,
and 3D gaps above **40 m** and **60 m**. Timing and synchronization flags
cover unsupported intervals, position speeds above **400 km/h**, and
persistent distance residuals. Every supported position interval can be
checked for persistence, including intervals whose speed is below the spike
threshold. Persistence requires a signed residual above **5 m** in magnitude
in the same direction in both centered **1 s** and **2 s** windows.
Interval columns attach to the ending sample. A blank marks a missing or
unevaluable value, including interval fields on the first row.

The recurrence flag marks forward projected intervals that intersect the
diagnostic band within **25 m** of VER arc **2,182.050412 m**. The manifest
also records the 1 s and 2 s residuals at that exact aligned location for
each lap. A region flag records location independently of whether that lap
shows persistence. The CLI accepts `--short-chord-m`,
`--gap-thresholds-m LARGE SEVERE`, `--speed-threshold-kmh`,
`--persistent-residual-m`, and `--recurrence-radius-m`. All flags are
annotations. The export retains every returned sample, including rows with
geometric and temporal flags.

The cached run prepared VER 17, LEC 21, HAM 20, NOR 22, PIA 18, and RUS 15.
Reconstruction has **1,294 position / 1,286 car rows** and validation has
**1,326 position / 1,297 car rows**. Every position interval synchronized,
and all six recorded laps had full car-speed integration coverage. The
three reconstruction laps had **74** speed spikes, including **6** with
persistent residuals. Validation had **66**, including **4** persistent.
The known residual at 2,182.1 m persisted for VER and HAM and fell below
the two-window criterion on all three validation laps. RUS's observed
polyline exceeded its car integral by **33.13 m** over the matched time
span. Absolute geometry and the cause of the remaining synchronization
differences are unresolved.

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
| `synchronized_speeds.png` | Native interval position speeds and matched car averages against lap time. |
| `speed_discrepancy_by_interval.png` | Signed speed discrepancies against native sampling intervals. |
| `synchronized_distances.png` | Cumulative distance from the first position sample and the signed residual. |
| `speed_anomalies_xy.png` | Equal-scale X/Y trajectories with above-threshold speed intervals highlighted. |
| `speed_discrepancy_by_window.png` | Mean and 95th-percentile absolute speed discrepancies at each time scale. |
| `gradient_vs_arc_length.png` | Native grade against cumulative 3D arc for each driver. |
| `gradient_window_comparison.png` | Uphill and downhill extremes and the absolute grade tail by window size. |
| `aligned_gradient_differences.png` | Grade differences relative to VER on a common vertical scale. |
| `horizontal_curvature.png` | Signed three-point XY curvature against 3D arc. |
| `vertical_curvature.png` | Signed change in inclination per metre against 3D arc. |
| `steep_sections_xy.png` | VER's uphill and downhill 50 m grade extrema on the X/Y trajectory. |

`outputs/speed_anomalies.csv` contains the Task 3A anomaly report. The
synchronization script writes its own files alongside the earlier plots.
`outputs/anomaly_recovery.csv` contains the signed neighbor and window
residuals from Task 3B, with aligned peer residuals for persistent cases.
