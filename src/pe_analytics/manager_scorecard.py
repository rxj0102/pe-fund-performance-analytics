"""Manager due-diligence scorecard.

Quantitative fund metrics (IRR, TVPI, PME) tell you what a manager *has*
returned. They say very little about whether that performance is
repeatable, or about a manager whose fund is still too young to have a
meaningful track record. A fund-of-funds underwriting a primary commitment
leans just as heavily on structured qualitative diligence — this module
implements a simple, transparent 1-5 scorecard across four dimensions that
FoF investment committees routinely examine:

1. **Team stability** — has the investment team that generated the track
   record stayed together, or has there been senior-partner turnover that
   puts continuity of the strategy at risk?
2. **Strategy consistency** — is the current fund investing in the same
   sectors, check sizes, and deal types as the funds that built the track
   record, or is there drift (chasing bigger deals, new sectors, new
   geographies) that makes the historical numbers less predictive?
3. **Track record dispersion vs. benchmark** — is outperformance broad-based
   across the portfolio, or is it concentrated in one or two outlier deals
   that mask a mediocre base rate? High dispersion lowers confidence that
   returns are repeatable skill rather than a few lucky outcomes.
4. **Fund size discipline** — is the GP raising a fund size consistent with
   the opportunity set that generated past returns, or stepping up
   aggressively in a way that risks style drift and diminishing returns to
   scale — a particularly acute risk in small-cap buyout, where the
   addressable deal universe is finite?

This scorecard is intentionally simple and transparent (rule-based, not a
black box) — the point for an investment committee is that every score
traces back to an explainable rationale, not that it is a sophisticated
predictive model.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ManagerProfile:
    """Qualitative and track-record inputs an analyst would gather during
    primary due diligence on a GP, ahead of a commitment decision."""

    manager_name: str
    fund_id: str

    # 1. Team stability
    years_team_together: float
    key_partner_turnover_pct: float  # % of senior investment team departed since prior fund
    founder_still_active: bool

    # 2. Strategy consistency
    prior_fund_sectors: list[str]
    current_fund_sectors: list[str]
    check_size_consistent: bool
    geography_consistent: bool

    # 3. Track record dispersion
    historical_deal_irrs: list[float] = field(default_factory=list)
    benchmark_irr: float = 0.10

    # 4. Fund size discipline
    prior_fund_size: float = 0.0
    current_fund_size: float = 0.0


@dataclass
class SubScore:
    score: float  # 1.0 - 5.0
    rationale: str


@dataclass
class ScorecardResult:
    fund_id: str
    manager_name: str
    team_stability: SubScore
    strategy_consistency: SubScore
    track_record_dispersion: SubScore
    fund_size_discipline: SubScore
    overall_score: float
    rating: str

    def as_dict(self) -> dict[str, str | float]:
        return {
            "fund_id": self.fund_id,
            "manager_name": self.manager_name,
            "team_stability": self.team_stability.score,
            "strategy_consistency": self.strategy_consistency.score,
            "track_record_dispersion": self.track_record_dispersion.score,
            "fund_size_discipline": self.fund_size_discipline.score,
            "overall_score": self.overall_score,
            "rating": self.rating,
        }


_WEIGHTS = {
    "team_stability": 0.25,
    "strategy_consistency": 0.25,
    "track_record_dispersion": 0.30,
    "fund_size_discipline": 0.20,
}


def _score_team_stability(p: ManagerProfile) -> SubScore:
    if p.key_partner_turnover_pct <= 0.10 and p.founder_still_active:
        return SubScore(5.0, "No material senior departures; founder/lead investor still active.")
    if p.key_partner_turnover_pct <= 0.25:
        return SubScore(
            3.5, f"Modest turnover ({p.key_partner_turnover_pct:.0%}) in the senior team since the prior fund."
        )
    if p.key_partner_turnover_pct <= 0.40 or not p.founder_still_active:
        return SubScore(
            2.0,
            f"Elevated turnover ({p.key_partner_turnover_pct:.0%}) or departure of the founder/lead "
            "investor — meaningful key-person risk to underwrite around.",
        )
    return SubScore(
        1.0,
        f"Turnover of {p.key_partner_turnover_pct:.0%} of the senior team is a material red flag; "
        "the team that generated the track record is largely gone.",
    )


def _score_strategy_consistency(p: ManagerProfile) -> SubScore:
    prior = set(p.prior_fund_sectors)
    current = set(p.current_fund_sectors)
    overlap = len(prior & current) / len(prior | current) if (prior | current) else 1.0

    flags = []
    if not p.check_size_consistent:
        flags.append("check sizes have shifted materially from prior funds")
    if not p.geography_consistent:
        flags.append("geographic focus has expanded/shifted")
    if overlap < 0.5:
        flags.append("sector focus has drifted from the funds that built the track record")

    if not flags and overlap >= 0.75:
        return SubScore(5.0, "Sector focus, check size, and geography are consistent with the track record.")
    if len(flags) <= 1:
        return SubScore(3.5, "Largely consistent strategy, with one area of drift: " + (flags[0] if flags else ""))
    if len(flags) == 2:
        return SubScore(2.0, "Multiple areas of strategy drift: " + "; ".join(flags) + ".")
    return SubScore(
        1.0,
        "Significant strategy drift across sector, check size, and geography — the track record is a "
        "weak predictor of what this fund will actually invest in: " + "; ".join(flags) + ".",
    )


def _score_track_record_dispersion(p: ManagerProfile) -> SubScore:
    irrs = p.historical_deal_irrs
    if len(irrs) < 3:
        return SubScore(2.5, "Track record too short to assess dispersion with confidence (fewer than 3 deals).")

    mean_irr = statistics.mean(irrs)
    stdev_irr = statistics.pstdev(irrs)
    pct_beat_benchmark = sum(1 for r in irrs if r > p.benchmark_irr) / len(irrs)
    # dispersion relative to the mean itself (coefficient of variation), guarding div-by-zero
    cov = stdev_irr / abs(mean_irr) if abs(mean_irr) > 1e-6 else float("inf")

    if pct_beat_benchmark >= 0.7 and cov < 0.6:
        return SubScore(
            5.0,
            f"{pct_beat_benchmark:.0%} of deals beat the benchmark with low dispersion (CoV {cov:.2f}) — "
            "broad-based outperformance, consistent with repeatable skill rather than a few outliers.",
        )
    if pct_beat_benchmark >= 0.5 and cov < 1.0:
        return SubScore(
            3.5,
            f"{pct_beat_benchmark:.0%} of deals beat the benchmark; moderate dispersion (CoV {cov:.2f}).",
        )
    if pct_beat_benchmark >= 0.4:
        return SubScore(
            2.0,
            f"Only {pct_beat_benchmark:.0%} of deals beat the benchmark and dispersion is high "
            f"(CoV {cov:.2f}) — returns look concentrated in a small number of outlier deals.",
        )
    return SubScore(
        1.0,
        f"Just {pct_beat_benchmark:.0%} of deals beat the benchmark — the base rate of success is weak "
        "regardless of headline fund-level IRR.",
    )


def _score_fund_size_discipline(p: ManagerProfile) -> SubScore:
    if p.prior_fund_size <= 0:
        return SubScore(3.0, "No prior fund to compare step-up against (first institutional fund).")
    step_up = (p.current_fund_size - p.prior_fund_size) / p.prior_fund_size
    if step_up <= 0.20:
        return SubScore(5.0, f"Disciplined fund sizing: {step_up:+.0%} step-up from the prior fund.")
    if step_up <= 0.50:
        return SubScore(3.5, f"Moderate step-up ({step_up:+.0%}) from the prior fund — worth monitoring.")
    if step_up <= 0.80:
        return SubScore(
            2.0,
            f"Aggressive step-up ({step_up:+.0%}) raises style-drift risk, particularly in a small-cap "
            "strategy with a finite addressable deal universe.",
        )
    return SubScore(
        1.0,
        f"Step-up of {step_up:+.0%} is a major red flag for a small-cap manager — fund size risks "
        "outgrowing the strategy that generated the track record.",
    )


def _rating_from_score(score: float) -> str:
    if score >= 4.3:
        return "High Conviction — Recommend Commit"
    if score >= 3.5:
        return "Conviction — Commit"
    if score >= 2.5:
        return "Watch List — Commit with Reservations / Further Diligence"
    return "Decline"


_SECTOR_POOL = [
    "Industrials & Manufacturing",
    "Business Services",
    "Healthcare Services",
    "Consumer & Retail",
    "Tech-Enabled Services",
    "Specialty Distribution",
    "Building Products",
]


def generate_synthetic_manager_profiles(
    funds: list, benchmark_irr: float = 0.09, seed: int = 13
) -> list[ManagerProfile]:
    """Generate illustrative qualitative diligence profiles for each fund's GP.

    Deliberately correlates the underlying "manager quality" signal (derived
    from each fund's synthetic ``target_moic``) with the qualitative inputs,
    but with meaningful noise layered on top — diligence inputs are directional
    evidence, not a perfect crystal ball, and the scorecard should not simply
    recover the fund's already-known quantitative outcome. This is what makes
    the resulting scores a genuine test of whether the qualitative framework
    adds signal beyond just looking at realized returns.
    """
    rng = np.random.default_rng(seed)
    profiles: list[ManagerProfile] = []

    for fund in funds:
        quality = float(np.clip((fund.target_moic - 0.7) / (3.2 - 0.7), 0.0, 1.0))
        manager_name = fund.name.split(" (")[0].rsplit(" ", 1)[0]

        turnover_pct = float(np.clip(rng.normal(0.30 - 0.22 * quality, 0.12), 0.0, 0.75))
        founder_active = bool(rng.random() < (0.55 + 0.35 * quality))

        current_sectors = list(rng.choice(_SECTOR_POOL, size=2, replace=False))
        if rng.random() < (0.35 + 0.45 * quality):
            prior_sectors = list(current_sectors)
        else:
            prior_sectors = list(rng.choice(_SECTOR_POOL, size=2, replace=False))
        check_size_consistent = bool(rng.random() < (0.5 + 0.4 * quality))
        geography_consistent = bool(rng.random() < (0.6 + 0.3 * quality))

        n_deals = int(rng.integers(5, 10))
        mean_deal_irr = benchmark_irr + (quality - 0.5) * 0.28
        dispersion_sd = 0.26 - 0.12 * quality
        deal_irrs = list(rng.normal(mean_deal_irr, dispersion_sd, size=n_deals))

        prior_fund_size = float(rng.uniform(150_000_000, 500_000_000))
        step_up = float(np.clip(rng.normal(0.15 + 0.25 * quality, 0.20), -0.10, 1.20))
        current_fund_size = prior_fund_size * (1.0 + step_up)

        profiles.append(
            ManagerProfile(
                manager_name=manager_name,
                fund_id=fund.fund_id,
                years_team_together=float(rng.uniform(4.0, 15.0)),
                key_partner_turnover_pct=turnover_pct,
                founder_still_active=founder_active,
                prior_fund_sectors=prior_sectors,
                current_fund_sectors=current_sectors,
                check_size_consistent=check_size_consistent,
                geography_consistent=geography_consistent,
                historical_deal_irrs=deal_irrs,
                benchmark_irr=benchmark_irr,
                prior_fund_size=prior_fund_size,
                current_fund_size=current_fund_size,
            )
        )

    return profiles


def score_manager(profile: ManagerProfile) -> ScorecardResult:
    team = _score_team_stability(profile)
    strategy = _score_strategy_consistency(profile)
    dispersion = _score_track_record_dispersion(profile)
    size = _score_fund_size_discipline(profile)

    overall = (
        team.score * _WEIGHTS["team_stability"]
        + strategy.score * _WEIGHTS["strategy_consistency"]
        + dispersion.score * _WEIGHTS["track_record_dispersion"]
        + size.score * _WEIGHTS["fund_size_discipline"]
    )

    return ScorecardResult(
        fund_id=profile.fund_id,
        manager_name=profile.manager_name,
        team_stability=team,
        strategy_consistency=strategy,
        track_record_dispersion=dispersion,
        fund_size_discipline=size,
        overall_score=round(overall, 2),
        rating=_rating_from_score(overall),
    )
