"""Public Market Equivalent (PME) benchmarking.

A fund-of-funds analyst never looks at a buyout fund's IRR in isolation —
the real question is "did this manager beat what I could have earned putting
the same dollars, on the same dates, into a public index instead?" PME
answers that question by replaying the fund's actual capital calls and
distributions against a public benchmark. Because this portfolio is
small-cap buyout, the natural benchmark is the Russell 2000 (small-cap
proxy) — a synthetic, seeded Russell-2000-like total return series is used
here so the whole project stays self-contained and reproducible.

Two complementary PME methodologies are implemented:

- **Kaplan-Schoar PME (KS-PME)**, Kaplan & Schoar (2005): the ratio of the
  future-value of distributions (plus residual value) to the future-value
  of contributions, both grown to a common valuation date using the index's
  total return. ``KS-PME > 1.0`` means the fund outperformed the index on a
  dollar-weighted multiple basis. It is simple, always well-defined, and is
  the PME statistic most commonly quoted in industry (e.g. by Preqin,
  Cambridge Associates, and in GP fundraising decks).

- **Long-Nickels PME (LN-PME / ICM)**, Long & Nickels (1996): builds a
  hypothetical "shadow" portfolio that receives the *same* capital calls and
  distributions, on the *same* dates, but invested in the index instead of
  the fund. The IRR of that shadow portfolio is the "PME IRR" — directly
  comparable to the fund's own XIRR. The spread (fund IRR minus PME IRR) is
  a clean, IRR-denominated measure of manager outperformance. Its known
  weakness: if a fund dramatically outperforms the index, the shadow
  portfolio's NAV can go negative (there isn't enough hypothetical index
  value left to fund later distributions), which makes the PME IRR
  undefined — a real methodological quirk of LN-PME, not a bug, and one
  worth knowing about before trusting it blindly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pe_analytics.cash_flows import CashFlow, Fund
from pe_analytics.irr_moic import xirr


@dataclass
class IndexSeries:
    """A daily public market index level series with date-based lookup."""

    levels: pd.Series  # index: pd.Timestamp (daily), values: index level

    def level_at(self, d: date) -> float:
        """Index level on date ``d``, linearly interpolated on log-level
        between the nearest available trading-day observations (and
        clamped to the series' start/end if ``d`` falls outside it)."""
        ts = pd.Timestamp(d)
        if ts <= self.levels.index[0]:
            return float(self.levels.iloc[0])
        if ts >= self.levels.index[-1]:
            return float(self.levels.iloc[-1])
        pos = self.levels.index.searchsorted(ts)
        d0, d1 = self.levels.index[pos - 1], self.levels.index[pos]
        v0, v1 = self.levels.iloc[pos - 1], self.levels.iloc[pos]
        frac = (ts - d0) / (d1 - d0)
        log_interp = math.log(v0) + frac * (math.log(v1) - math.log(v0))
        return float(math.exp(log_interp))


def generate_synthetic_index(
    start: date,
    end: date,
    annual_drift: float = 0.09,
    annual_vol: float = 0.21,
    seed: int = 22,
    name: str = "Synthetic Russell 2000 TR",
) -> IndexSeries:
    """Generate a seeded, reproducible small-cap-like daily total return index.

    Uses geometric Brownian motion with drift/vol calibrated to be broadly
    representative of long-run small-cap equity total returns (the Russell
    2000's realized long-run annualized total return and volatility have
    historically been in a similar range). This is illustrative only — it
    is not fit to actual historical Russell 2000 data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    dt = 1.0 / 252.0
    daily_drift = (annual_drift - 0.5 * annual_vol**2) * dt
    daily_vol = annual_vol * math.sqrt(dt)
    shocks = rng.normal(daily_drift, daily_vol, size=n)
    log_levels = np.cumsum(shocks)
    levels = 100.0 * np.exp(log_levels - log_levels[0])
    series = pd.Series(levels, index=dates, name=name)
    return IndexSeries(levels=series)


@dataclass
class PMEResult:
    fund_id: str
    fund_irr: float | None
    ks_pme: float | None
    ln_pme_irr: float | None
    irr_spread_bps: float | None  # fund IRR minus LN-PME IRR, in basis points

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "fund_id": self.fund_id,
            "fund_irr": self.fund_irr,
            "ks_pme": self.ks_pme,
            "ln_pme_irr": self.ln_pme_irr,
            "irr_spread_bps": self.irr_spread_bps,
        }


def kaplan_schoar_pme(
    cash_flows: list[CashFlow],
    residual_value: float,
    as_of_date: date,
    index: IndexSeries,
) -> float | None:
    """KS-PME = [FV(distributions) + residual value] / FV(contributions).

    Every cash flow is grown to ``as_of_date`` using the ratio of index
    levels; residual value is added unscaled since it is already expressed
    in as-of-date dollars (the fund's current reported NAV).
    """
    idx_T = index.level_at(as_of_date)
    fv_calls = 0.0
    fv_dists = 0.0
    for cf in cash_flows:
        idx_t = index.level_at(cf.cf_date)
        if idx_t <= 0:
            continue
        scale = idx_T / idx_t
        if cf.kind == "capital_call":
            fv_calls += -cf.amount * scale
        else:
            fv_dists += cf.amount * scale
    if fv_calls <= 0:
        return None
    return (fv_dists + residual_value) / fv_calls


def long_nickels_pme_irr(
    cash_flows: list[CashFlow],
    residual_value: float,
    as_of_date: date,
    index: IndexSeries,
) -> tuple[float | None, bool]:
    """Long-Nickels (ICM) PME IRR via a shadow index-tracking portfolio.

    Replays the fund's exact capital calls and distributions into a
    hypothetical portfolio invested in the index, growing it by the index's
    return between cash flow dates, then computes the IRR of that shadow
    portfolio's cash flows (same dates/amounts as the fund, but with the
    shadow terminal NAV in place of the fund's actual residual value).

    Returns ``(irr, nav_went_negative)``. When the shadow NAV goes negative
    (the fund's distributions outpaced what the index could have supported),
    the resulting IRR is not economically meaningful — this is flagged via
    the second return value rather than silently returned as a number.
    """
    flows = sorted(cash_flows, key=lambda cf: cf.cf_date)
    if not flows:
        return None, False

    shadow_nav = 0.0
    prev_date = flows[0].cf_date
    went_negative = False
    shadow_flows: list[tuple[date, float]] = []

    for cf in flows:
        idx_prev = index.level_at(prev_date)
        idx_now = index.level_at(cf.cf_date)
        if idx_prev > 0:
            shadow_nav *= idx_now / idx_prev
        prev_date = cf.cf_date

        if cf.kind == "capital_call":
            shadow_nav += -cf.amount
            shadow_flows.append((cf.cf_date, cf.amount))
        else:
            shadow_nav -= cf.amount
            shadow_flows.append((cf.cf_date, cf.amount))
            if shadow_nav < 0:
                went_negative = True

    idx_prev = index.level_at(prev_date)
    idx_now = index.level_at(as_of_date)
    if idx_prev > 0:
        shadow_nav *= idx_now / idx_prev

    shadow_flows.append((as_of_date, shadow_nav))
    irr = xirr(shadow_flows)
    return irr, went_negative


def pme_analysis(fund: Fund, as_of_date: date, index: IndexSeries, fund_irr: float | None) -> PMEResult:
    ks = kaplan_schoar_pme(fund.cash_flows, fund.residual_value, as_of_date, index)
    ln_irr, went_negative = long_nickels_pme_irr(fund.cash_flows, fund.residual_value, as_of_date, index)
    if went_negative:
        ln_irr = None  # not economically meaningful; see docstring above

    spread_bps = None
    if fund_irr is not None and ln_irr is not None:
        spread_bps = (fund_irr - ln_irr) * 10_000

    return PMEResult(
        fund_id=fund.fund_id,
        fund_irr=fund_irr,
        ks_pme=ks,
        ln_pme_irr=ln_irr,
        irr_spread_bps=spread_bps,
    )


def portfolio_kaplan_schoar_pme(funds: list[Fund], as_of_date: date, index: IndexSeries) -> float | None:
    """Pooled KS-PME across the whole portfolio (all funds' cash flows combined)."""
    all_cfs = [cf for f in funds for cf in f.cash_flows]
    total_residual = sum(f.residual_value for f in funds)
    return kaplan_schoar_pme(all_cfs, total_residual, as_of_date, index)
