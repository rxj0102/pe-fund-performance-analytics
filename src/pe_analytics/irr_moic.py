"""Core PE performance metrics: XIRR, MOIC, DPI, RVPI, TVPI.

These are the metrics every fund-of-funds analyst pulls up first when
screening a manager or reviewing portfolio performance:

- **IRR** (via XIRR on actual dated cash flows): the money-weighted annual
  return, accounting for exactly when capital was called and returned. This
  is the industry-standard way to compute PE IRR — never approximate it by
  annualizing a simple return, since PE cash flows are irregular and
  unevenly timed.
- **MOIC** (Multiple on Invested Capital): total value returned per dollar
  invested, with no regard for timing. ``MOIC = (distributions + residual
  value) / paid-in capital``.
- **DPI** (Distributions to Paid-In): cash-on-cash return already realized.
  ``DPI = distributions / paid-in capital``.
- **RVPI** (Residual Value to Paid-In): unrealized value still held in the
  fund. ``RVPI = residual value / paid-in capital``.
- **TVPI** (Total Value to Paid-In): ``TVPI = DPI + RVPI``, economically
  identical to MOIC but expressed the way GPs report it quarterly.

IRR and TVPI are complementary, not redundant: TVPI tells you the multiple
a fund is on track to return; IRR tells you how efficiently it got (or is
getting) there. A fund can have a strong TVPI but a mediocre IRR if capital
sat called-but-unrealized for years — which matters enormously to a
fund-of-funds managing liquidity across a whole program.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from scipy.optimize import brentq

from pe_analytics.cash_flows import CashFlow, Fund


def xirr(cash_flows: list[tuple[date, float]], guess: float = 0.15) -> float | None:
    """Compute the money-weighted IRR on actual dated cash flows (XIRR).

    Solves for the annualized rate ``r`` such that the NPV of all cash flows,
    discounted from a common reference date using actual day counts (Actual/365),
    is zero:

        sum_i  amount_i / (1 + r) ** ((date_i - date_0) / 365)  ==  0

    This is the standard Excel XIRR convention and the correct way to compute
    IRR on irregularly timed private equity cash flows — a naive annualized
    return would silently misstate performance whenever calls and
    distributions aren't evenly spaced, which in PE they never are.

    Returns ``None`` if no root can be found (e.g. all cash flows are the
    same sign, so no discount rate can make NPV zero).
    """
    flows = sorted(cash_flows, key=lambda x: x[0])
    if len(flows) < 2:
        return None
    amounts = [amt for _, amt in flows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None  # IRR is undefined without both an outflow and an inflow

    t0 = flows[0][0]
    days = [(d - t0).days / 365.0 for d, _ in flows]

    def npv(rate: float) -> float:
        return sum(amt / (1.0 + rate) ** t for amt, t in zip(amounts, days))

    # Bracket the root. NPV(rate) is monotonically decreasing in rate for a
    # typical call-then-distribute cash flow pattern, so search a wide range
    # of annual rates before giving up.
    lo, hi = -0.9999, 10.0
    try:
        npv_lo, npv_hi = npv(lo), npv(hi)
    except (OverflowError, ZeroDivisionError):
        return None
    if npv_lo * npv_hi > 0:
        # try to widen/narrow the bracket by scanning
        found = False
        for hi_try in (5.0, 20.0, 50.0, 100.0):
            if npv(lo) * npv(hi_try) < 0:
                hi = hi_try
                found = True
                break
        if not found:
            return None
    try:
        rate = brentq(npv, lo, hi, xtol=1e-10, maxiter=500)
    except (ValueError, RuntimeError):
        return None
    return rate


@dataclass
class PerformanceMetrics:
    """Standard PE performance summary for a fund or portfolio, as of a date."""

    as_of_date: date
    paid_in: float
    distributed: float
    residual_value: float
    irr: float | None
    moic: float
    dpi: float
    rvpi: float
    tvpi: float

    def as_dict(self) -> dict[str, float | date | None]:
        return {
            "as_of_date": self.as_of_date,
            "paid_in": self.paid_in,
            "distributed": self.distributed,
            "residual_value": self.residual_value,
            "irr": self.irr,
            "moic": self.moic,
            "dpi": self.dpi,
            "rvpi": self.rvpi,
            "tvpi": self.tvpi,
        }


def compute_metrics(
    cash_flows: list[CashFlow],
    residual_value: float,
    as_of_date: date,
) -> PerformanceMetrics:
    """Compute IRR/MOIC/DPI/RVPI/TVPI from a fund's cash flows and current NAV.

    ``residual_value`` (the fund's current reported NAV) is added to the
    cash flow list as a final synthetic "distribution" dated ``as_of_date``
    when computing IRR — this is standard practice for reporting an
    interim, not-yet-fully-realized IRR: it treats unrealized value as if
    liquidated at the reporting date, which is why interim IRR is always a
    mark-to-model estimate, not a realized return.
    """
    paid_in = -sum(cf.amount for cf in cash_flows if cf.kind == "capital_call")
    distributed = sum(cf.amount for cf in cash_flows if cf.kind == "distribution")

    dated_flows = [(cf.cf_date, cf.amount) for cf in cash_flows]
    if residual_value > 1e-6:
        dated_flows.append((as_of_date, residual_value))
    irr = xirr(dated_flows)

    dpi = distributed / paid_in if paid_in > 0 else 0.0
    rvpi = residual_value / paid_in if paid_in > 0 else 0.0
    tvpi = dpi + rvpi
    moic = tvpi  # MOIC and TVPI are the same ratio, expressed for the same purpose

    return PerformanceMetrics(
        as_of_date=as_of_date,
        paid_in=paid_in,
        distributed=distributed,
        residual_value=residual_value,
        irr=irr,
        moic=moic,
        dpi=dpi,
        rvpi=rvpi,
        tvpi=tvpi,
    )


def fund_metrics(fund: Fund, as_of_date: date) -> PerformanceMetrics:
    """Compute performance metrics for a single Fund position."""
    return compute_metrics(fund.cash_flows, fund.residual_value, as_of_date)


def portfolio_metrics(funds: list[Fund], as_of_date: date) -> PerformanceMetrics:
    """Compute aggregate performance metrics across a portfolio of funds.

    Cash flows are pooled across all funds (as a fund-of-funds LP experiences
    them in aggregate) and residual values are summed, then IRR/MOIC/DPI/RVPI/
    TVPI are computed on the pooled series. This is the correct way to
    compute a portfolio-level IRR — it is *not* a simple or weighted average
    of individual fund IRRs, since that would ignore the actual timing of
    pooled cash flows.
    """
    all_cash_flows = [cf for fund in funds for cf in fund.cash_flows]
    total_residual = sum(fund.residual_value for fund in funds)
    return compute_metrics(all_cash_flows, total_residual, as_of_date)
