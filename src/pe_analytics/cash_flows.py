"""Synthetic private equity fund cash flow generation.

Generates realistic, dated capital call and distribution streams for a
portfolio of buyout funds, co-investments, and secondary interests, in the
shape a fund-of-funds (FoF) limited partner actually receives them: quarterly
capital calls during an investment period, distributions ramping through a
harvest period, and a quarterly-reported NAV path that lands near zero once
a fund is fully realized.

All data here is synthetic and illustrative. It is generated with a seeded
random number generator so results are reproducible, but it does not
represent any real fund, GP, or investment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np

FundStrategy = Literal["primary", "co-investment", "secondary"]
CashFlowKind = Literal["capital_call", "distribution"]


@dataclass(frozen=True)
class CashFlow:
    """A single dated LP-level cash flow.

    By convention (matching how a fund-of-funds LP experiences cash flows):
    capital calls are negative (cash leaving the LP's account) and
    distributions are positive (cash returning to the LP).
    """

    cf_date: date
    amount: float
    kind: CashFlowKind

    def __post_init__(self) -> None:
        if self.kind == "capital_call" and self.amount > 0:
            raise ValueError("capital_call cash flows must be <= 0")
        if self.kind == "distribution" and self.amount < 0:
            raise ValueError("distribution cash flows must be >= 0")


@dataclass
class Fund:
    """A single underlying fund position held by the fund-of-funds.

    ``vintage_year`` is always the *underlying fund's* vintage (the year it
    started investing), even for a secondary interest bought several years
    later — this matches how LPs and GPs label vintage year in practice.
    For a secondary, ``acquisition_date`` records when the FoF actually
    entered the position.
    """

    fund_id: str
    name: str
    strategy: FundStrategy
    vintage_year: int
    sector_focus: str
    commitment: float
    cash_flows: list[CashFlow]
    nav_history: list[tuple[date, float]]
    fund_life_years: float
    acquisition_date: date | None = None
    target_moic: float = field(default=0.0, repr=False)

    @property
    def paid_in(self) -> float:
        """Total capital called from the LP (positive number)."""
        return -sum(cf.amount for cf in self.cash_flows if cf.kind == "capital_call")

    @property
    def distributed(self) -> float:
        """Total capital returned to the LP."""
        return sum(cf.amount for cf in self.cash_flows if cf.kind == "distribution")

    @property
    def residual_value(self) -> float:
        """Most recently reported NAV (0.0 if fully realized or not yet reported)."""
        if not self.nav_history:
            return 0.0
        latest = self.nav_history[-1][1]
        return latest if latest > 1e-6 else 0.0

    @property
    def is_realized(self) -> bool:
        return self.residual_value < 1e-6 and self.distributed > 0

    def sorted_cash_flows(self) -> list[CashFlow]:
        return sorted(self.cash_flows, key=lambda cf: cf.cf_date)


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

_SECTORS = [
    "Industrials & Manufacturing",
    "Business Services",
    "Healthcare Services",
    "Consumer & Retail",
    "Tech-Enabled Services",
    "Niche Industrials",
    "Specialty Distribution",
    "Education Services",
    "Food & Beverage",
    "Building Products",
]

_GP_NAMES = [
    "Meridian Capital Partners",
    "Cascade Buyout Fund",
    "Stonebridge Equity Partners",
    "Highcrest Capital",
    "Ironwood Partners",
    "Summit Ridge Capital",
    "Harborview Equity Partners",
    "Redwood Private Equity",
    "Pinnacle Growth Partners",
    "Northgate Capital Partners",
]


def _s_curve(frac: np.ndarray, steepness: float = 6.0) -> np.ndarray:
    """Logistic S-curve on [0, 1] -> [0, 1], centered at frac=0.5."""
    raw = 1.0 / (1.0 + np.exp(-steepness * (frac - 0.5)))
    lo, hi = 1.0 / (1.0 + np.exp(steepness * 0.5)), 1.0 / (1.0 + np.exp(-steepness * 0.5))
    return (raw - lo) / (hi - lo)


def _quarter_ends(start: date, n_quarters: int) -> list[date]:
    dates = []
    year, q = start.year, (start.month - 1) // 3
    for i in range(n_quarters):
        total_q = q + i
        yr = year + total_q // 4
        month = (total_q % 4) * 3 + 3
        # last day of `month`
        if month == 12:
            d = date(yr, 12, 31)
        else:
            d = date(yr, month + 1, 1) - timedelta(days=1)
        dates.append(d)
    return dates


def _jitter_date(d: date, rng: np.random.Generator, max_days: int = 25) -> date:
    offset = int(rng.integers(-max_days, max_days + 1))
    return d + timedelta(days=offset)


def _generate_primary_lifecycle(
    commitment: float,
    vintage_year: int,
    fund_life_years: float,
    target_moic: float,
    rng: np.random.Generator,
    as_of_date: date,
    investment_period_years: float = 5.0,
) -> tuple[list[CashFlow], list[tuple[date, float]]]:
    """Simulate a standard diversified primary fund's quarterly lifecycle.

    Capital calls are front-loaded over the investment period; distributions
    ramp up starting mid-life and continue through the harvest period. NAV is
    derived from a smooth TVPI(t) path (with an initial J-curve dip below
    1.0x from fee drag) so that ``distributions + NAV`` is internally
    consistent with the fund's target MOIC at the end of its life.
    """
    n_quarters = int(round(fund_life_years * 4))
    start = date(vintage_year, 1, 1)
    q_dates = _quarter_ends(start, n_quarters)
    t_years = np.array([(d - start).days / 365.25 for d in q_dates])
    frac = t_years / fund_life_years

    # --- capital call schedule: front-loaded over investment period ---
    call_period_q = int(round(investment_period_years * 4))
    call_weights = np.exp(-0.22 * np.arange(call_period_q))
    call_weights /= call_weights.sum()
    quarterly_calls = np.zeros(n_quarters)
    quarterly_calls[:call_period_q] = call_weights * commitment
    cum_paid_in = np.cumsum(quarterly_calls)

    # --- TVPI(t): J-curve dip then S-curve growth to target_moic ---
    growth = 1.0 + (target_moic - 1.0) * _s_curve(frac, steepness=5.0)
    dip = 0.07 * np.exp(-frac / 0.12)
    tvpi_path = growth * (1.0 - dip)
    tvpi_path = np.maximum(tvpi_path, 0.05)
    total_value = tvpi_path * cum_paid_in

    # --- distribution schedule: none during investment period, then a
    #     harvest-shaped ramp that exactly drains total_value by fund end ---
    dist_start_q = int(round(2.5 * 4))
    dist_weights = np.zeros(n_quarters)
    if dist_start_q < n_quarters:
        span = n_quarters - dist_start_q
        x = np.arange(span)
        shape = _s_curve(x / max(span - 1, 1), steepness=3.5) + 0.15
        dist_weights[dist_start_q:] = shape
    dist_weights /= dist_weights.sum()

    final_total_value = total_value[-1]
    quarterly_dists = dist_weights * final_total_value

    # Ensure distributions never exceed available total value at each point
    # (clip and re-derive NAV as the residual so the series stays consistent).
    cum_dist = np.cumsum(quarterly_dists)
    nav_path = total_value - cum_dist
    if (nav_path < -1e-6).any():
        nav_path = np.clip(nav_path, 0.0, None)

    cash_flows: list[CashFlow] = []
    nav_history: list[tuple[date, float]] = []
    for i, d in enumerate(q_dates):
        if d > as_of_date:
            break
        if quarterly_calls[i] > 1.0:
            cf_date = _jitter_date(d, rng)
            cash_flows.append(CashFlow(cf_date, -round(float(quarterly_calls[i]), 2), "capital_call"))
        if quarterly_dists[i] > 1.0:
            cf_date = _jitter_date(d, rng)
            cash_flows.append(CashFlow(cf_date, round(float(quarterly_dists[i]), 2), "distribution"))
        nav_history.append((d, round(float(nav_path[i]), 2)))

    return cash_flows, nav_history


def _generate_co_investment(
    commitment: float,
    vintage_year: int,
    target_moic: float,
    rng: np.random.Generator,
    as_of_date: date,
    hold_years: float = 4.0,
) -> tuple[list[CashFlow], list[tuple[date, float]], float]:
    """Simulate a single-deal co-investment: one funding event, one exit.

    Co-investments are called almost entirely up front (no diversified
    capital call schedule) and typically exit as a single event rather than
    a multi-year harvest, so both MOIC dispersion and timing risk are higher
    than a diversified primary fund.
    """
    start = date(vintage_year, rng.integers(1, 4) * 3, 1)
    funding_date = _jitter_date(start, rng, max_days=15)
    exit_date = funding_date + timedelta(days=int(hold_years * 365.25))

    cash_flows = [CashFlow(funding_date, -round(commitment, 2), "capital_call")]
    nav_history = [(funding_date, round(commitment, 2))]

    # a small follow-on call is common for co-investments (~10-15% of deal size)
    if rng.random() < 0.4:
        followon_date = _jitter_date(
            funding_date + timedelta(days=int(0.4 * hold_years * 365.25)), rng, max_days=15
        )
        followon_amt = commitment * rng.uniform(0.08, 0.18)
        if followon_date <= as_of_date:
            cash_flows.append(CashFlow(followon_date, -round(followon_amt, 2), "capital_call"))
        total_commitment = commitment + followon_amt
    else:
        total_commitment = commitment

    exit_value = total_commitment * target_moic

    quarter_marks = _quarter_ends(funding_date, int(hold_years * 4) + 1)
    for d in quarter_marks:
        if d >= exit_date or d > as_of_date:
            break
        t = (d - funding_date).days / max((exit_date - funding_date).days, 1)
        nav = total_commitment * (1.0 + (target_moic - 1.0) * _s_curve(np.array([t]))[0])
        nav_history.append((d, round(float(nav), 2)))

    if exit_date <= as_of_date:
        cash_flows.append(CashFlow(exit_date, round(exit_value, 2), "distribution"))
        nav_history.append((exit_date, 0.0))
    else:
        # still held as of the valuation date: mark remaining NAV, no exit yet
        t = (as_of_date - funding_date).days / max((exit_date - funding_date).days, 1)
        nav = total_commitment * (1.0 + (target_moic - 1.0) * _s_curve(np.array([min(t, 1.0)]))[0])
        nav_history.append((as_of_date, round(float(nav), 2)))

    return sorted(cash_flows, key=lambda cf: cf.cf_date), nav_history, total_commitment


def _generate_secondary(
    commitment: float,
    vintage_year: int,
    fund_life_years: float,
    target_moic: float,
    rng: np.random.Generator,
    as_of_date: date,
    entry_year_offset: float = 3.5,
    discount_to_nav: float = 0.08,
) -> tuple[list[CashFlow], list[tuple[date, float]], date]:
    """Simulate a secondary purchase of an LP interest partway through a fund's life.

    A full "shadow" primary lifecycle is simulated for the underlying fund
    from its true vintage year. The FoF then buys in at ``entry_year_offset``
    years into that life, paying the underlying fund's NAV at that point less
    a discount (secondaries commonly transact at a discount to reported
    NAV), and receives the underlying fund's cash flows from that date
    forward. This produces the muted early J-curve that is characteristic of
    secondaries: capital is deployed against an already-invested, partially
    de-risked portfolio.
    """
    shadow_commitment = commitment / max(target_moic, 1.0) * 1.4  # oversized shadow fund so our slice is a fraction
    shadow_cfs, shadow_nav, = _generate_primary_lifecycle(
        commitment=shadow_commitment,
        vintage_year=vintage_year,
        fund_life_years=fund_life_years,
        target_moic=target_moic,
        rng=rng,
        as_of_date=date(vintage_year + int(fund_life_years) + 1, 12, 31),
    )

    entry_date = date(vintage_year, 1, 1) + timedelta(days=int(entry_year_offset * 365.25))
    entry_date = _jitter_date(entry_date, rng, max_days=30)

    nav_at_entry = 0.0
    for d, nav in shadow_nav:
        if d <= entry_date:
            nav_at_entry = nav
        else:
            break
    nav_at_entry = max(nav_at_entry, shadow_commitment * 0.3)

    # scale factor: the FoF's slice of the shadow fund, priced off NAV at entry
    purchase_price = nav_at_entry * (1.0 - discount_to_nav)
    scale = commitment / nav_at_entry if nav_at_entry > 0 else 0.0

    cash_flows = [CashFlow(entry_date, -round(purchase_price, 2), "capital_call")]
    nav_history = [(entry_date, round(purchase_price, 2))]

    for cf in shadow_cfs:
        if cf.cf_date <= entry_date or cf.cf_date > as_of_date:
            continue
        scaled_amt = round(cf.amount * scale, 2)
        if abs(scaled_amt) < 1.0:
            continue
        cash_flows.append(CashFlow(cf.cf_date, scaled_amt, cf.kind))

    for d, nav in shadow_nav:
        if d <= entry_date or d > as_of_date:
            continue
        nav_history.append((d, round(nav * scale, 2)))

    return sorted(cash_flows, key=lambda cf: cf.cf_date), nav_history, entry_date


def generate_fund_portfolio(
    n_funds: int = 10,
    start_vintage: int = 2013,
    end_vintage: int = 2020,
    as_of_date: date = date(2026, 6, 30),
    commitment_range: tuple[float, float] = (8_000_000, 12_000_000),
    n_co_investments: int = 2,
    n_secondaries: int = 1,
    seed: int = 42,
) -> list[Fund]:
    """Generate a synthetic small-cap buyout fund-of-funds portfolio.

    Produces ``n_funds`` positions spread across vintage years
    ``start_vintage``-``end_vintage``, mostly diversified primary fund
    commitments with a handful of co-investments and a secondary purchase,
    mirroring how a small-cap buyout fund-of-funds like 747 Capital actually
    deploys capital across the three strategies.
    """
    rng = np.random.default_rng(seed)
    vintages = np.linspace(start_vintage, end_vintage, n_funds)
    vintages = np.round(vintages).astype(int)
    vintages = np.sort(vintages)

    n_primary = n_funds - n_co_investments - n_secondaries
    strategies = (
        ["primary"] * n_primary + ["co-investment"] * n_co_investments + ["secondary"] * n_secondaries
    )
    rng.shuffle(strategies)

    funds: list[Fund] = []
    for i in range(n_funds):
        vintage_year = int(vintages[i])
        strategy = strategies[i]
        commitment = float(rng.uniform(*commitment_range))
        sector = _SECTORS[i % len(_SECTORS)]
        gp_base = _GP_NAMES[i % len(_GP_NAMES)]
        fund_life = float(rng.uniform(9.0, 12.0))

        # Target MOIC dispersion: mostly solid (1.4x-2.2x) buyout outcomes,
        # with realistic tails — one laggard below 1.0x, one standout above 2.5x.
        moic_roll = rng.random()
        if moic_roll < 0.12:
            target_moic = float(rng.uniform(0.65, 0.95))
        elif moic_roll > 0.88:
            target_moic = float(rng.uniform(2.4, 3.2))
        else:
            target_moic = float(rng.uniform(1.35, 2.15))

        if strategy == "primary":
            roman = ["II", "III", "IV", "V"][i % 4]
            name = f"{gp_base} {roman}"
            cfs, nav_hist = _generate_primary_lifecycle(
                commitment, vintage_year, fund_life, target_moic, rng, as_of_date
            )
            fund = Fund(
                fund_id=f"F{i+1:02d}",
                name=name,
                strategy="primary",
                vintage_year=vintage_year,
                sector_focus=sector,
                commitment=commitment,
                cash_flows=cfs,
                nav_history=nav_hist,
                fund_life_years=fund_life,
                target_moic=target_moic,
            )
        elif strategy == "co-investment":
            deal_names = ["Project Atlas", "Project Beacon", "Project Compass", "Project Delta"]
            name = f"{deal_names[i % len(deal_names)]} Co-Invest"
            hold = float(rng.uniform(3.0, 5.5))
            cfs, nav_hist, total_commit = _generate_co_investment(
                commitment, vintage_year, target_moic, rng, as_of_date, hold_years=hold
            )
            fund = Fund(
                fund_id=f"F{i+1:02d}",
                name=name,
                strategy="co-investment",
                vintage_year=vintage_year,
                sector_focus=sector,
                commitment=total_commit,
                cash_flows=cfs,
                nav_history=nav_hist,
                fund_life_years=hold,
                target_moic=target_moic,
            )
        else:  # secondary
            roman = ["II", "III", "IV"][i % 3]
            name = f"{gp_base} {roman} (Secondary Interest)"
            entry_offset = float(rng.uniform(3.0, 4.5))
            cfs, nav_hist, acq_date = _generate_secondary(
                commitment, vintage_year, fund_life, target_moic, rng, as_of_date,
                entry_year_offset=entry_offset,
            )
            fund = Fund(
                fund_id=f"F{i+1:02d}",
                name=name,
                strategy="secondary",
                vintage_year=vintage_year,
                sector_focus=sector,
                commitment=commitment,
                cash_flows=cfs,
                nav_history=nav_hist,
                fund_life_years=fund_life,
                acquisition_date=acq_date,
                target_moic=target_moic,
            )

        funds.append(fund)

    return funds
