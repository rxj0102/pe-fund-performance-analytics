"""Commitment pacing and liquidity projection model.

A fund-of-funds doesn't just pick managers — it has to plan *when* to commit
to them. Because capital calls and distributions from any single fund are
unpredictable and lag the commitment by years, an LP that wants to keep a
steady amount of capital deployed has to commit steadily every vintage year,
long before it knows how earlier vintages will perform. This module answers
the liquidity-planning question a FoF analyst is actually asked: "if we
commit $X per year across a few new funds annually, what does our aggregate
called capital, distributions, and NAV look like over the next decade — and
when do we need cash on hand to meet capital calls?"

The single-vintage cash flow shape is generated with a simplified model in
the spirit of the widely used Takahashi & Alexander (2001) "Illiquid
Alternative Asset Fund Modeling" framework: contributions are called as a
fraction of *remaining* uncalled commitment each period (so calls
naturally front-load and then taper), distributions ramp up over fund life
as an S-curve applied to a base distribution rate against NAV, and NAV
compounds at an assumed gross growth rate between calls and distributions.
This is a deliberately simplified adaptation for illustrative pacing
purposes, not a reproduction of the original paper's exact parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


def _s_curve(frac: np.ndarray, steepness: float = 5.0) -> np.ndarray:
    raw = 1.0 / (1.0 + np.exp(-steepness * (frac - 0.5)))
    lo, hi = 1.0 / (1.0 + np.exp(steepness * 0.5)), 1.0 / (1.0 + np.exp(-steepness * 0.5))
    return (raw - lo) / (hi - lo)


def ta_vintage_schedule(
    commitment: float,
    vintage_year: int,
    fund_life_years: float = 11.0,
    contribution_rate_annual: float = 0.32,
    distribution_rate_annual: float = 0.28,
    growth_rate_annual: float = 0.13,
    bow_steepness: float = 4.5,
) -> pd.DataFrame:
    """Project a single vintage-year commitment's quarterly cash flows and NAV.

    - ``contribution_rate_annual``: the fraction of *remaining* uncalled
      commitment drawn down each year (so calls are largest early and taper
      as the commitment gets used up) — typical small-cap buyout funds
      substantially finish calling capital within 4-6 years.
    - ``distribution_rate_annual``: the base annual distribution rate against
      NAV once the fund is in harvest mode; it is scaled by an S-curve ramp
      over fund life so early-life distributions are minimal (the J-curve).
    - ``growth_rate_annual``: assumed gross annual growth rate of unrealized
      portfolio value.

    Returns a DataFrame with one row per quarter: ``date``, ``contributions``,
    ``distributions``, ``net_cash_flow``, ``nav`` (all in dollars, calls
    shown as negative contributions from the LP's perspective).
    """
    n_quarters = int(round(fund_life_years * 4))
    rc_q = 1.0 - (1.0 - contribution_rate_annual) ** 0.25
    g_q = (1.0 + growth_rate_annual) ** 0.25 - 1.0
    rd_base_q = distribution_rate_annual / 4.0

    dates = pd.date_range(start=date(vintage_year, 3, 31), periods=n_quarters, freq="QE")

    paid_in = 0.0
    nav = 0.0
    rows = []
    for q in range(n_quarters):
        life_frac = q / max(n_quarters - 1, 1)

        contrib = rc_q * (commitment - paid_in)
        contrib = max(contrib, 0.0)
        paid_in += contrib

        ramp = float(_s_curve(np.array([life_frac]), steepness=bow_steepness)[0])
        dist = rd_base_q * ramp * nav
        dist = min(dist, nav + contrib)

        nav = nav * (1.0 + g_q) + contrib - dist

        # fully liquidate in the final quarter, as pacing models conventionally assume
        if q == n_quarters - 1 and nav > 0:
            dist += nav
            nav = 0.0

        rows.append(
            {
                "date": dates[q],
                "contributions": -contrib,
                "distributions": dist,
                "net_cash_flow": dist - contrib,
                "nav": nav,
            }
        )

    return pd.DataFrame(rows)


@dataclass
class PacingAssumptions:
    fund_life_years: float = 11.0
    contribution_rate_annual: float = 0.32
    distribution_rate_annual: float = 0.28
    growth_rate_annual: float = 0.13


def project_commitment_pacing(
    start_year: int,
    n_commitment_years: int,
    annual_commitment_target: float,
    commitments_per_year: int = 3,
    assumptions: PacingAssumptions = PacingAssumptions(),
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project a fund-of-funds' aggregate cash flow and NAV under a steady
    annual commitment pace.

    Commits ``annual_commitment_target`` per year, split across
    ``commitments_per_year`` new fund commitments, for ``n_commitment_years``
    consecutive vintage years starting at ``start_year``. Each commitment is
    projected forward through its full fund life using
    :func:`ta_vintage_schedule`, and all cohorts are summed by calendar
    quarter to produce the FoF's aggregate liquidity projection.

    Returns ``(aggregate, cohorts)`` where ``aggregate`` has one row per
    calendar quarter (``date``, ``contributions``, ``distributions``,
    ``net_cash_flow``, ``cumulative_net_cash_flow``, ``nav``) spanning from
    the first commitment through the last cohort's fund life, and
    ``cohorts`` is a long-format DataFrame with each individual vintage-year
    commitment's own schedule (for drill-down / stacked-area plotting).
    """
    rng = np.random.default_rng(seed)
    cohort_frames = []

    for i, vintage_year in enumerate(range(start_year, start_year + n_commitment_years)):
        # split the annual target unevenly but plausibly across commitments
        splits = rng.dirichlet(np.ones(commitments_per_year) * 3.0)
        for j, frac in enumerate(splits):
            commitment = annual_commitment_target * frac
            schedule = ta_vintage_schedule(
                commitment=commitment,
                vintage_year=vintage_year,
                fund_life_years=assumptions.fund_life_years,
                contribution_rate_annual=assumptions.contribution_rate_annual,
                distribution_rate_annual=assumptions.distribution_rate_annual,
                growth_rate_annual=assumptions.growth_rate_annual,
            )
            schedule["cohort_id"] = f"{vintage_year}-{j+1}"
            schedule["vintage_year"] = vintage_year
            schedule["commitment"] = commitment
            cohort_frames.append(schedule)

    cohorts = pd.concat(cohort_frames, ignore_index=True)

    aggregate = (
        cohorts.groupby("date")[["contributions", "distributions", "net_cash_flow", "nav"]]
        .sum()
        .sort_index()
    )
    full_index = pd.date_range(start=aggregate.index.min(), end=aggregate.index.max(), freq="QE")
    aggregate = aggregate.reindex(full_index, fill_value=0.0)
    # NAV must be forward-filled at quarters where no cohort had an entry (shouldn't
    # happen given aligned quarterly grids, but is defensive against edge cohorts)
    aggregate["cumulative_net_cash_flow"] = aggregate["net_cash_flow"].cumsum()
    aggregate.index.name = "date"

    return aggregate.reset_index(), cohorts
