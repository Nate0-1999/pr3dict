from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.platforms.base import Market
from src.strategies.dependency_detector import (
    DependencyAssessment,
    DependencyDetector,
    DependencyRelation,
    DependencyVerifierPort,
)


def _market(mid: str, title: str) -> Market:
    return Market(
        id=mid,
        ticker=mid.upper(),
        title=title,
        description="",
        yes_price=Decimal("0.50"),
        no_price=Decimal("0.50"),
        volume=Decimal("1000"),
        liquidity=Decimal("1000"),
        close_time=datetime.now(timezone.utc) + timedelta(days=20),
        resolved=False,
        platform="polymarket",
    )


def test_detect_mutually_exclusive_candidates():
    detector = DependencyDetector()
    a = _market("a", "Will Donald Trump win the 2028 US presidential election?")
    b = _market("b", "Will Kamala Harris win the 2028 US presidential election?")
    assessments = detector.detect([a, b])
    assert len(assessments) == 1
    assert assessments[0].relation == DependencyRelation.MUTUALLY_EXCLUSIVE
    assert assessments[0].confidence >= 0.7


def test_detect_equivalent_on_same_normalized_title():
    detector = DependencyDetector()
    a = _market("a", "Will BTC be above 120k by Dec 31 2026?")
    b = _market("b", "Will BTC be above 120k by Dec 31, 2026?")
    assessments = detector.detect([a, b])
    assert len(assessments) == 1
    assert assessments[0].relation == DependencyRelation.EQUIVALENT


def test_detect_implies_for_nominee_to_win():
    detector = DependencyDetector()
    a = _market("a", "Will Donald Trump win the Republican nomination in 2028?")
    b = _market("b", "Will Donald Trump win the 2028 US presidential election?")
    assessments = detector.detect([a, b])
    assert len(assessments) == 1
    assert assessments[0].relation == DependencyRelation.IMPLIES


class _Verifier(DependencyVerifierPort):
    def verify(self, market_a: Market, market_b: Market, deterministic: DependencyAssessment):
        return DependencyAssessment(
            market_a_id=market_a.id,
            market_b_id=market_b.id,
            relation=DependencyRelation.INDEPENDENT,
            confidence=0.91,
            reason="verifier override",
            source="verifier",
        )


def test_verifier_can_override_deterministic_assessment():
    detector = DependencyDetector()
    a = _market("a", "Will Donald Trump win the 2028 US presidential election?")
    b = _market("b", "Will Kamala Harris win the 2028 US presidential election?")
    assessments = detector.detect([a, b], verifier=_Verifier())
    assert len(assessments) == 1
    assert assessments[0].relation == DependencyRelation.INDEPENDENT
    assert assessments[0].source == "verifier"
