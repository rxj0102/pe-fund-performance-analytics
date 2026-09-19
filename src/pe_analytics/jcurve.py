"""J-curve modeling: cumulative net cash flow over a fund's life.

Every PE fund draws down capital before it returns any — fees and early
acquisition costs are paid out immediately, while portfolio companies take
years to grow and exit. Plotted as cumulative net cash flow (distributions
minus calls) against time since inception, this produces the characteristic
"J" shape: a deepening negative trough during the investment period,
followed by a recovery and eventual crossover into positive territory as
distributions accelerate.

This matters operationally, not just aesthetically, for a fund-of-funds: an
LP committing to several vintages in a row will see its blended cash flow
position sit underwater for the first several years of the program even
while returns are on track — the J-curve is exactly the liquidity dynamic
the pacing model in ``pacing_model.py`` has to plan around.
"""

from __future__ import annotations

import pandas as pd

from pe_analytics.cash_flows import Fund


def single_fund_jcurve(fund: Fund, points_per_year: int = 4) -> pd.DataFrame:
    """Cumulative net cash flow over a single fund's life, on a quarterly grid.

    Returns a DataFrame indexed by ``years_since_inception`` with columns:
      - ``date``: calendar date for that grid point
      - ``cum_net_cf``: cumulative (distributions - calls) in dollars
      - ``cum_net_cf_pct_commitment``: the same, normalized by commitment
    """
    flows = fund.sorted_cash_flows()
    if not flows:
        return pd.DataFrame(columns=["date", "years_since_inception", "cum_net_cf", "cum_net_cf_pct_commitment"])

    inception = flows[0].cf_date
    last_date = flows[-1].cf_date
    total_years = max((last_date - inception).days / 365.25, 0.25)
    n_points = int(total_years * points_per_year) + 2

    cf_series = pd.Series(
        {pd.Timestamp(cf.cf_date): cf.amount for cf in flows}
    )
    cf_series = cf_series.groupby(cf_series.index).sum()

    grid_dates = pd.date_range(start=inception, periods=n_points, freq=f"{12 // points_per_year}ME")
    all_dates = cf_series.index.union(grid_dates)
    cf_full = cf_series.reindex(all_dates, fill_value=0.0).sort_index()
    cum_cf = cf_full.cumsum()

    sampled = cum_cf.reindex(cum_cf.index.union(grid_dates)).sort_index().ffill().reindex(grid_dates)
    years_since = [(d - pd.Timestamp(inception)).days / 365.25 for d in grid_dates]

    return pd.DataFrame(
        {
            "date": grid_dates,
            "years_since_inception": years_since,
            "cum_net_cf": sampled.values,
            "cum_net_cf_pct_commitment": sampled.values / fund.commitment,
        }
    )


def portfolio_average_jcurve(
    funds: list[Fund], max_years: float = 12.0, points_per_year: int = 4
) -> pd.DataFrame:
    """Average (equal-weighted) normalized J-curve across a portfolio of funds.

    Each fund's cumulative net cash flow is normalized to a % of its own
    commitment and aligned on "years since its own inception" (not calendar
    time), so funds from different vintage years overlay on a common
    lifecycle axis. This is the standard way industry vintage-year J-curve
    charts are built (e.g. Cambridge Associates benchmark reports): it shows
    the *typical shape* of the J-curve across the program, independent of
    when each fund actually started.

    Returns a DataFrame with one row per grid point (``years_since_inception``)
    and one column per fund (``cum_net_cf_pct_commitment``) plus an
    ``average`` column giving the equal-weighted mean across funds that are
    still within their observed life at that point.
    """
    grid = [i / points_per_year for i in range(int(max_years * points_per_year) + 1)]
    data: dict[str, list[float | None]] = {}

    for fund in funds:
        curve = single_fund_jcurve(fund, points_per_year=points_per_year)
        if curve.empty:
            continue
        max_observed_year = curve["years_since_inception"].max()
        series = curve.set_index("years_since_inception")["cum_net_cf_pct_commitment"]
        col = []
        last_val = 0.0
        for y in grid:
            if y > max_observed_year + 1e-6:
                col.append(None)
                continue
            idx = series.index[series.index <= y]
            last_val = series.loc[idx[-1]] if len(idx) else 0.0
            col.append(float(last_val))
        data[fund.fund_id] = col

    df = pd.DataFrame(data, index=grid)
    df.index.name = "years_since_inception"
    df["average"] = df.mean(axis=1, skipna=True)
    return df
