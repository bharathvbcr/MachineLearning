"""Student-t critical values and paired intervals, exact at every sample size.

The canonical owner of "how wide is the interval around this paired mean". Every
reader that prints a confidence interval reads it from here, so that a board, a
crossing estimate and a funnel ranking cannot disagree about what 95% means.

Why this exists rather than a table
-----------------------------------
Three readers each carried their own t constants and each was wrong somewhere:

  * ``scripts/paired_board.py`` hardcoded the **4-dof** row and applied it at any
    sample size. Exact at five seeds, and wrong in both directions away from it:
    at n=3 its 95% interval is 35.5% too NARROW (overconfident), at n=32 it is
    36% too wide (so a real effect fails to clear a margin it should clear).
  * ``scripts/crossing_token.py`` carried five dof values and raised "add it" for
    anything else -- loud, but it means the estimator simply cannot run at a
    sample size nobody anticipated.
  * ``native_funnel.T_CRITICAL_95`` tabulates df 1..30 and then falls back to the
    NORMAL quantile. That cliff is not harmless: df=31 is 2.039513 and the normal
    is 1.959964, so every interval past the table is ~4% too narrow, and the
    error grows with the confidence level, which is exactly where a table cliff
    hurts most.

A confirmation design at 12-32 seeds (df 11-31) straddles all three failures at
once. So the multiplier is computed, not looked up: any df >= 1 and any
confidence in (0, 1), to full double precision.

Method
------
For T with ``df`` degrees of freedom, Abramowitz & Stegun 26.7.1 gives the
two-sided tail directly as a regularized incomplete beta:

    P(|T| > t) = I_{df/(df+t^2)}(df/2, 1/2)

``I`` is evaluated by the Lentz continued fraction (Numerical Recipes 6.4), which
is standard and converges in a few dozen terms over the whole range used here.
The critical value inverts that tail by bisection -- the tail is strictly
monotone in ``t``, so bisection cannot miss the root and needs no derivative or
starting guess. 200 halvings of a bracket that starts at [0, 1e7] resolve ``t``
to well under one ulp of the printed precision.

No SciPy. This module is imported by the analysis CLIs, and SciPy is not a
dependency of this package (it appears only in a dated audit probe); adding one
to a core reader to fetch a single quantile is not a trade worth making.
"""
from __future__ import annotations

import math
import statistics

__all__ = [
    "student_t_critical",
    "student_t_two_sided_sf",
    "paired_interval",
    "MIN_SAMPLES_FOR_INTERVAL",
]

# Below two samples there is no spread to estimate, so there is no interval. The
# honest width is infinite, never zero: a single observation must not be able to
# look infinitely precise and separate itself from every rival.
MIN_SAMPLES_FOR_INTERVAL = 2

_MAX_T = 1e7          # bisection bracket; t(df=1, conf=1-1e-9) is ~6.4e8 but no
                      # caller asks for that, and the bracket is checked below.
_BISECT_STEPS = 200   # halving 1e7 two hundred times is far past double precision


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4, Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            return h
    raise ArithmeticError(
        f"incomplete beta continued fraction did not converge for a={a}, b={b}, x={x}")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), for a, b > 0 and x in [0, 1]."""
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    # The fraction converges fast only on the side where x is small relative to
    # the shape; the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) moves it there.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_sf(t: float, df: int) -> float:
    """P(|T| > t) for T with ``df`` degrees of freedom. Exact, not tabulated."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if t < 0.0:
        raise ValueError(f"t must be >= 0, got {t}")
    if math.isinf(t):
        return 0.0
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def student_t_critical(df: int, conf: float = 0.95) -> float:
    """Two-sided Student-t multiplier at ``conf`` for ``df`` degrees of freedom.

    ``student_t_critical(4, 0.95) == 2.776445...``, the constant three readers in
    this tree used to carry. Unlike those, it is also right at df=3 and df=31.

    ``df < 1`` has no interval and returns infinity rather than a number: an
    interval that could not be computed must never be reported as a narrow one.
    """
    if df < 1:
        return math.inf
    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    alpha = 1.0 - conf
    lo, hi = 0.0, _MAX_T
    if student_t_two_sided_sf(hi, df) > alpha:
        raise ValueError(
            f"conf={conf} at df={df} needs a t beyond the {_MAX_T:g} bracket")
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:      # bracket collapsed to adjacent doubles
            break
        if student_t_two_sided_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_interval(values, conf: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) for paired differences, at the dof the sample actually has.

    ``values`` are per-pair differences already -- one number per seed, not two
    groups. Fewer than two of them yields an infinite interval around the mean,
    because one pair carries no estimate of spread.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("paired_interval needs at least one value")
    # A non-finite difference means a run went non-finite or a terminal value is
    # missing. Averaging it yields nan and prints as a number; dropping it
    # silently changes the denominator and reports a capped sample as a complete
    # one. Name the offenders and refuse.
    bad = [(i, v) for i, v in enumerate(vals) if not math.isfinite(v)]
    if bad:
        raise ValueError(
            "paired_interval refuses non-finite differences at index(es) "
            + ", ".join(f"{i} ({v})" for i, v in bad)
            + "; exclude those pairs explicitly and say so, or rerun them")
    mean = statistics.fmean(vals)
    if len(vals) < MIN_SAMPLES_FOR_INTERVAL:
        return mean, -math.inf, math.inf
    half = (student_t_critical(len(vals) - 1, conf)
            * statistics.stdev(vals) / math.sqrt(len(vals)))
    return mean, mean - half, mean + half
