"""Numerical operations on periodic XYZ splines parameterized in reference metres."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.special import roots_legendre

from formularl.geometry import FloatArray
from formularl.periodic import wrap_arc

TANGENT_EPS = 1e-8
LENGTH_TOLERANCE_M = 1e-6
DUPLICATE_TOLERANCE_M = 1e-8


def unique_observations(
    q: FloatArray, xyz: FloatArray, period: float
) -> tuple[FloatArray, FloatArray]:
    """Average coincident parameters within a driver, including near the seam."""
    wrapped = wrap_arc(q, period)
    order = np.argsort(wrapped, kind="stable")
    wrapped = wrapped[order]
    gaps = np.diff(np.r_[wrapped, wrapped[0] + period])
    cut = (int(gaps.argmax()) + 1) % len(wrapped)
    u = np.r_[wrapped[cut:], wrapped[:cut] + period]
    values = xyz[np.r_[order[cut:], order[:cut]]]
    breaks = [0]
    for i in range(1, len(u)):
        if u[i] - u[breaks[-1]] > DUPLICATE_TOLERANCE_M:
            breaks.append(i)
    q = wrap_arc(np.array([g.mean() for g in np.split(u, breaks[1:])]), period)
    order = np.argsort(q)
    return q[order], np.array([g.mean(axis=0) for g in np.split(values, breaks[1:])])[
        order
    ]


def periodic_linear(
    q: FloatArray, xyz: FloatArray, grid: FloatArray, period: float, max_gap: float
) -> FloatArray:
    """Interpolate on the two cyclic neighbors when their span has support."""
    if not len(q):
        return np.full((len(grid), 3), np.nan)
    grid = wrap_arc(grid, period)
    knots, values = unique_observations(q, xyz, period)
    knots = np.r_[knots[-1] - period, knots, knots[0] + period]
    values = np.vstack((values[-1], values, values[0]))
    right = np.searchsorted(knots, grid, side="right")
    left = right - 1
    spans = knots[right] - knots[left]
    fraction = (grid - knots[left]) / spans
    support = spans <= max_gap
    interpolated = values[left] + fraction[:, None] * (values[right] - values[left])
    return np.where((support | (grid == knots[left]))[:, None], interpolated, np.nan)


def periodic_spline(q: FloatArray, xyz: FloatArray, period: float) -> CubicSpline:
    """Close explicitly at the origin so both seam limits can be evaluated."""
    if len(q) < 3 or q[0] != 0 or q[-1] >= period or np.any(np.diff(q) <= 0):
        raise ValueError(
            "Periodic knots must increase from zero and end before the period."
        )
    return CubicSpline(np.r_[q, period], np.vstack((xyz, xyz[0])), bc_type="periodic")


def _length(
    curve: CubicSpline,
    left: FloatArray,
    right: FloatArray,
    nodes: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    width = right - left
    q = left[:, None] + width[:, None] * nodes
    return width * (np.linalg.norm(curve(q, 1), axis=-1) @ weights)


@dataclass(frozen=True)
class ArcMap:
    curve: CubicSpline
    cumulative_m: FloatArray
    order: int
    differences: tuple[float, ...]

    @property
    def length_m(self) -> float:
        return float(self.cumulative_m[-1])

    def at_distance(self, distance: FloatArray) -> FloatArray:
        """Invert length inside its own spline span by monotone bisection."""
        if np.any((distance < 0) | (distance > self.length_m)):
            raise ValueError("Requested distance lies outside the fitted lap.")
        span = np.minimum(
            np.searchsorted(self.cumulative_m, distance, side="right") - 1,
            len(self.cumulative_m) - 2,
        )
        start, stop = self.curve.x[span], self.curve.x[span + 1]
        target = distance - self.cumulative_m[span]
        nodes, weights = roots_legendre(self.order)
        nodes, weights = (nodes + 1) / 2, weights / 2
        lower, upper = start.copy(), stop.copy()
        for _ in range(32):
            middle = (lower + upper) / 2
            before = _length(self.curve, start, middle, nodes, weights) < target
            lower, upper = (
                np.where(before, middle, lower),
                np.where(before, upper, middle),
            )
        return np.where(
            target == 0,
            start,
            np.where(distance == self.length_m, stop, (lower + upper) / 2),
        )

    def uniform(self, spacing_m: float = 1.0) -> tuple[FloatArray, FloatArray]:
        n = max(1, round(self.length_m / spacing_m))
        distance = np.linspace(0, self.length_m, n, endpoint=False)
        return distance, self.at_distance(distance)


def arc_map(curve: CubicSpline) -> ArcMap:
    """Integrate tangent magnitude span by span, with an order-doubling check."""
    previous = None
    differences = []
    for order in (8, 16, 32, 64, 128):
        nodes, weights = roots_legendre(order)
        lengths = _length(
            curve, curve.x[:-1], curve.x[1:], (nodes + 1) / 2, weights / 2
        )
        if previous is not None:
            differences.append(float(np.abs(lengths - previous).sum()))
            if differences[-1] <= LENGTH_TOLERANCE_M:
                cumulative = np.r_[0.0, np.cumsum(lengths)]
                if np.any(np.diff(cumulative) <= 0):
                    raise ValueError(
                        "A zero-length spline span makes the arc inverse non-unique."
                    )
                return ArcMap(curve, cumulative, order, tuple(differences))
        previous = lengths
    raise ValueError(
        f"Spline arc integration did not converge: last difference {differences[-1]:.3g} m."
    )


def differential_geometry(v: FloatArray, a: FloatArray) -> dict[str, FloatArray]:
    """Derivatives of grade and curvature for an arbitrary regular parameter."""
    horizontal = np.linalg.norm(v[:, :2], axis=1)
    speed = np.linalg.norm(v, axis=1)
    usable = (speed > TANGENT_EPS) & (horizontal > TANGENT_EPS)
    grade = np.divide(
        100 * v[:, 2], horizontal, out=np.full_like(speed, np.nan), where=usable
    )
    h_rate = np.divide(
        np.sum(v[:, :2] * a[:, :2], axis=1),
        horizontal,
        out=np.full_like(speed, np.nan),
        where=usable,
    )
    return {
        "tangent_magnitude": speed,
        "inclination_deg": np.where(
            speed > TANGENT_EPS, np.degrees(np.arctan2(v[:, 2], horizontal)), np.nan
        ),
        "gradient_pct": grade,
        "horizontal_curvature_inv_m": np.divide(
            v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0],
            horizontal**3,
            out=np.full_like(speed, np.nan),
            where=usable,
        ),
        "vertical_curvature_inv_m": np.divide(
            horizontal * a[:, 2] - v[:, 2] * h_rate,
            speed**3,
            out=np.full_like(speed, np.nan),
            where=usable,
        ),
    }


def project_xy(
    curve: CubicSpline, xyz: FloatArray, expected: FloatArray, radius: float
) -> FloatArray:
    """Find continuous XY minima inside the periodic correspondence corridor."""
    widths = np.diff(curve.x)
    # Scale each polynomial to t in [0,1] for the degree-five distance derivative.
    xy = (
        curve.c[:, :, :2] * widths[None, :, None] ** np.arange(3, -1, -1)[:, None, None]
    )[::-1]
    xy = np.moveaxis(xy, 1, 0)
    diff = xy[:, 1:] * np.arange(1, 4)[None, :, None]
    base = np.array(
        [
            np.convolve(c[:, 0], d[:, 0]) + np.convolve(c[:, 1], d[:, 1])
            for c, d in zip(xy, diff, strict=True)
        ]
    )
    period = curve.x[-1]
    starts = np.r_[curve.x[:-1] - period, curve.x[:-1], curve.x[:-1] + period]
    ends = np.r_[curve.x[1:] - period, curve.x[1:], curve.x[1:] + period]
    result = np.full(len(xyz), np.nan)
    for row, (point, anchor) in enumerate(zip(xyz[:, :2], expected, strict=True)):
        if not np.isfinite(anchor) or not np.isfinite(point).all():
            continue
        low, high = anchor - radius, anchor + radius
        first, last = (
            np.searchsorted(ends, low, side="right"),
            np.searchsorted(starts, high),
        )
        best = np.inf
        for span in range(first, last):
            i = span % len(widths)
            lo = max(0.0, (low - starts[span]) / widths[i])
            hi = min(1.0, (high - starts[span]) / widths[i])
            derivative = base[i].copy()
            derivative[:3] -= diff[i] @ point
            roots = np.polynomial.polynomial.polyroots(derivative)
            interior = roots.real[
                (np.abs(roots.imag) < 1e-8) & (roots.real > lo) & (roots.real < hi)
            ]
            candidates = np.r_[lo, interior, hi]
            offsets = np.polynomial.polynomial.polyval(
                candidates, (xy[i] - np.r_[point[None], np.zeros((3, 2))])
            ).T
            distances = np.sum(offsets**2, axis=1)
            winner = distances.argmin()
            if distances[winner] < best:
                best = distances[winner]
                result[row] = starts[span] + candidates[winner] * widths[i]
    return result


def coordinate_extrema(curve: CubicSpline) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return exact axis extrema and maximum excess outside each knot interval box."""
    low, high, excess = np.empty(3), np.empty(3), np.zeros(3)
    for axis in range(3):
        coordinate = PPoly(curve.c[:, :, axis], curve.x, extrapolate=False)
        knots = coordinate(curve.x)
        q = coordinate.derivative().roots(extrapolate=False)
        q = q[np.isfinite(q) & (q > curve.x[0]) & (q < curve.x[-1])]
        values = coordinate(q)
        low[axis], high[axis] = (
            np.min(np.r_[knots, values]),
            np.max(np.r_[knots, values]),
        )
        if values.size:
            i = np.searchsorted(curve.x, q, side="right") - 1
            below = np.minimum(knots[i], knots[i + 1]) - values
            above = values - np.maximum(knots[i], knots[i + 1])
            excess[axis] = max(0.0, below.max(), above.max())
    return low, high, excess
