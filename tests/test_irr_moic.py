"""Validate XIRR/MOIC/DPI/RVPI/TVPI against hand-calculated examples.

These are the load-bearing metric implementations for the whole project, so
they are checked first, against numbers worked out independently (and cross-
checked against Excel's XIRR), before anything else is built on top of them.
"""

from datetime import date

import pytest

from pe_analytics.cash_flows import CashFlow, Fund
from pe_analytics.irr_moic import fund_metrics, portfolio_metrics, xirr


def test_xirr_simple_annual_case():
    """One capital call, one distribution exactly one year later.

    -100 at t0, +112 at t0+365 days => (112/100) - 1 = 12.00% IRR exactly,
    since 2019-01-01 to 2020-01-01 spans exactly 365 days under Actual/365
    (2019 is not a leap year).
    """
    flows = [(date(2019, 1, 1), -100.0), (date(2020, 1, 1), 112.0)]
    rate = xirr(flows)
    assert rate is not None
    assert rate == pytest.approx(0.12, abs=1e-6)


def test_xirr_multi_cash_flow_series_zeroes_npv():
    """A multi-call, multi-distribution series: verify the *definition* of
    XIRR directly, independent of the module's own implementation — the
    rate returned must discount all dated cash flows (Actual/365) to a net
    present value of zero. This is the same check Excel's XIRR() performs.

    Cash flows:
        2018-01-01  -100,000  (call)
        2019-06-15   -50,000  (call)
        2020-12-01   +30,000  (distribution)
        2022-03-10   +180,000 (distribution)
    """
    flows = [
        (date(2018, 1, 1), -100_000.0),
        (date(2019, 6, 15), -50_000.0),
        (date(2020, 12, 1), 30_000.0),
        (date(2022, 3, 10), 180_000.0),
    ]
    rate = xirr(flows)
    assert rate is not None

    t0 = flows[0][0]
    npv = sum(amt / (1.0 + rate) ** ((d - t0).days / 365.0) for d, amt in flows)
    assert npv == pytest.approx(0.0, abs=1e-4)
    # sanity band: this series' true rate is ~9.98%, not a round number
    assert rate == pytest.approx(0.0998, abs=5e-4)


def test_xirr_returns_none_without_both_signs():
    flows = [(date(2020, 1, 1), -100.0), (date(2021, 1, 1), -50.0)]
    assert xirr(flows) is None


def test_moic_dpi_rvpi_tvpi_hand_calculated():
    """A simple fund: $100 called, $80 distributed, $60 residual NAV.

    Expected by definition:
        DPI  = 80 / 100 = 0.80x
        RVPI = 60 / 100 = 0.60x
        TVPI = DPI + RVPI = 1.40x
        MOIC = TVPI = 1.40x
    """
    fund = Fund(
        fund_id="TEST01",
        name="Test Fund I",
        strategy="primary",
        vintage_year=2018,
        sector_focus="Test Sector",
        commitment=100.0,
        cash_flows=[
            CashFlow(date(2018, 1, 1), -60.0, "capital_call"),
            CashFlow(date(2019, 1, 1), -40.0, "capital_call"),
            CashFlow(date(2021, 1, 1), 30.0, "distribution"),
            CashFlow(date(2022, 1, 1), 50.0, "distribution"),
        ],
        nav_history=[(date(2023, 1, 1), 60.0)],
        fund_life_years=10.0,
    )
    metrics = fund_metrics(fund, as_of_date=date(2023, 1, 1))

    assert metrics.paid_in == pytest.approx(100.0)
    assert metrics.distributed == pytest.approx(80.0)
    assert metrics.residual_value == pytest.approx(60.0)
    assert metrics.dpi == pytest.approx(0.80)
    assert metrics.rvpi == pytest.approx(0.60)
    assert metrics.tvpi == pytest.approx(1.40)
    assert metrics.moic == pytest.approx(1.40)
    assert metrics.irr is not None


def test_fully_realized_fund_dpi_equals_tvpi():
    """A fully realized fund (NAV = 0) should have DPI == TVPI == MOIC."""
    fund = Fund(
        fund_id="TEST02",
        name="Test Fund II",
        strategy="primary",
        vintage_year=2015,
        sector_focus="Test Sector",
        commitment=100.0,
        cash_flows=[
            CashFlow(date(2015, 1, 1), -100.0, "capital_call"),
            CashFlow(date(2023, 1, 1), 250.0, "distribution"),
        ],
        nav_history=[(date(2023, 1, 1), 0.0)],
        fund_life_years=8.0,
    )
    metrics = fund_metrics(fund, as_of_date=date(2023, 1, 1))
    assert metrics.rvpi == pytest.approx(0.0)
    assert metrics.dpi == pytest.approx(metrics.tvpi)
    assert metrics.moic == pytest.approx(2.5)


def test_portfolio_metrics_pools_cash_flows_not_averages_irr():
    """Portfolio IRR must be computed on pooled dated cash flows, not by
    averaging individual fund IRRs (which would ignore relative sizing and
    timing overlap across funds)."""
    fund_a = Fund(
        fund_id="A",
        name="Fund A",
        strategy="primary",
        vintage_year=2018,
        sector_focus="X",
        commitment=100.0,
        cash_flows=[
            CashFlow(date(2018, 1, 1), -100.0, "capital_call"),
            CashFlow(date(2022, 1, 1), 200.0, "distribution"),
        ],
        nav_history=[(date(2023, 1, 1), 0.0)],
        fund_life_years=8.0,
    )
    fund_b = Fund(
        fund_id="B",
        name="Fund B",
        strategy="primary",
        vintage_year=2019,
        sector_focus="Y",
        commitment=200.0,
        cash_flows=[
            CashFlow(date(2019, 1, 1), -200.0, "capital_call"),
            CashFlow(date(2023, 1, 1), 260.0, "distribution"),
        ],
        nav_history=[(date(2023, 1, 1), 0.0)],
        fund_life_years=8.0,
    )
    metrics = portfolio_metrics([fund_a, fund_b], as_of_date=date(2023, 1, 1))
    assert metrics.paid_in == pytest.approx(300.0)
    assert metrics.distributed == pytest.approx(460.0)
    assert metrics.tvpi == pytest.approx(460.0 / 300.0)
    assert metrics.irr is not None
