# PE Fund-of-Funds Performance Analytics

[![CI](https://github.com/rxj0102/pe-fund-performance-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/pe-fund-performance-analytics/actions/workflows/ci.yml)

A Python toolkit for analyzing private equity fund performance the way a **fund-of-funds (FoF) analyst** actually does it: dated-cash-flow IRR, MOIC/DPI/RVPI/TVPI, Public Market Equivalent (PME) benchmarking, J-curve modeling, commitment pacing, and a structured manager due-diligence scorecard — built around a synthetic small-cap buyout portfolio spanning primaries, co-investments, and secondaries.

**Full walkthrough:** [`notebooks/portfolio_analysis.ipynb`](notebooks/portfolio_analysis.ipynb) — an end-to-end analysis written with investment-committee-style commentary.

> **This project uses entirely synthetic, illustrative cash flow data.** No fund, GP, or manager named here is real, and no real fund data was used at any point. The methodology — how each metric is computed — is real and industry-standard; the numbers behind it are generated (seeded, reproducible) for demonstration purposes only.

---

## What a fund-of-funds does

A private equity fund-of-funds doesn't invest directly in companies — it invests in *other GPs' funds* (primary commitments), alongside them in individual deals (co-investments), and occasionally buys existing LP interests from other investors (secondaries), on behalf of its own limited partners. Its job is threefold: pick managers who can repeat their track record, benchmark whether that access is actually worth the illiquidity versus just buying the public market, and plan the pacing of new commitments so the fund-of-funds itself never runs short of cash when calls come due. This project builds the analytical toolkit for all three.

## Why track IRR, MOIC, DPI, RVPI, *and* TVPI — not just one

| Metric | What it measures | What it misses |
|---|---|---|
| **IRR** (XIRR on dated cash flows) | Time-sensitive, money-weighted annual return | Says nothing about total value created — a fund can have a great IRR on a small, fast deal and a mediocre one on a much larger value-creation story |
| **MOIC** | Total value returned per dollar invested | Completely ignores timing — a 2.5x in 4 years and a 2.5x in 12 years look identical |
| **DPI** | *Realized* cash-on-cash return already distributed | Doesn't capture value still building in the portfolio |
| **RVPI** | *Unrealized* value still marked on the books | Is a GP's mark, not a proven exit — the least trustworthy of the four for a young fund |
| **TVPI** | DPI + RVPI — the total value multiple, GP-reported quarterly | Same blind spot as MOIC: no sense of timing efficiency |

A fund-of-funds analyst looks at all five together because they answer different questions: TVPI tells you where a fund is headed, DPI/RVPI tell you how much of that is proven versus still a mark, and IRR tells you whether capital was used efficiently getting there. **PME** answers a different question again — not "was this a good return," but "was locking up capital in an illiquid fund actually worth it, versus the public market?" Two methodologies are implemented:

- **Kaplan-Schoar PME (KS-PME):** future-values every cash flow to a common date using a benchmark index, then divides FV(distributions + residual value) by FV(contributions). Simple, always defined, and the PME statistic most commonly quoted industry-wide.
- **Long-Nickels PME (LN-PME):** replays the fund's *exact* cash flows into a hypothetical index-tracking shadow portfolio and computes its IRR — directly comparable to the fund's own IRR. Known limitation: if a fund dramatically outperforms, the shadow portfolio's NAV can go negative, making its IRR undefined (this is flagged, not silently miscomputed).

Because this portfolio is small-cap buyout, benchmarking uses a synthetic **Russell 2000**-like index.

## What's in the repo

```
src/pe_analytics/
    cash_flows.py         # Fund/CashFlow model + synthetic portfolio generator
                           #   (primary, co-investment, and secondary cash flow profiles)
    irr_moic.py            # XIRR, MOIC, DPI, RVPI, TVPI — per fund and pooled portfolio-wide
    pme_benchmarking.py    # Synthetic index generator; Kaplan-Schoar & Long-Nickels PME
    jcurve.py               # Cumulative net cash flow over fund life — single fund & portfolio-averaged
    pacing_model.py         # Takahashi-Alexander-style commitment pacing / liquidity projection
    manager_scorecard.py   # Structured 1-5 manager due-diligence scorecard
notebooks/
    portfolio_analysis.ipynb  # Full end-to-end walkthrough with IC-style commentary and charts
tests/
    test_irr_moic.py        # XIRR/MOIC/DPI/RVPI/TVPI validated against hand-calculated examples
```

### The synthetic portfolio

10 positions, 2013–2020 vintage years: 7 diversified primary fund commitments, 2 co-investments (single-deal, front-loaded calls, concentrated risk), and 1 secondary (an LP interest purchased partway through an existing fund's life, priced off NAV at a discount — which mutes the early J-curve). Generation is fully seeded (`numpy.random.default_rng`), so every run is reproducible.

## How to run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt

# run the validation tests
pytest tests/ -v

# run the full analysis notebook end to end
jupyter nbconvert --to notebook --execute --inplace notebooks/portfolio_analysis.ipynb
# or open it interactively:
jupyter notebook notebooks/portfolio_analysis.ipynb
```

Or use the library directly:

```python
from datetime import date
from pe_analytics.cash_flows import generate_fund_portfolio
from pe_analytics.irr_moic import fund_metrics, portfolio_metrics

funds = generate_fund_portfolio()
as_of = date(2026, 6, 30)

for fund in funds:
    m = fund_metrics(fund, as_of)
    print(f"{fund.fund_id}: IRR={m.irr:.1%}  TVPI={m.tvpi:.2f}x  DPI={m.dpi:.2f}x  RVPI={m.rvpi:.2f}x")

print(portfolio_metrics(funds, as_of))
```

## Reproducibility

### Data availability

No external data is downloaded. Inputs are generated or parameterised inside the code.

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (6 tests) finishes in about 9 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## Disclaimer

All fund names, GP names, cash flows, NAVs, index levels, and manager due-diligence profiles in this repository are **synthetically generated for illustrative purposes**. Nothing here represents any real fund, general partner, limited partner, or investment, and no real, non-public fund data was used in building this project. Random number generation is seeded throughout for reproducibility.

## License

MIT — see [LICENSE](LICENSE).
