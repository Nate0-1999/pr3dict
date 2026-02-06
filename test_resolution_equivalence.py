from datetime import datetime, timezone

import pytest

from src.strategies.dispute.resolution_equivalence import (
    EquivalenceDecision,
    ResolutionEquivalenceEngine,
    ResolutionMarket,
    heuristic_similarity,
    normalize_kalshi_market,
    normalize_polymarket_market,
    rank_candidates,
)


def _mk_market(platform: str, market_id: str, title: str, close_time: datetime) -> ResolutionMarket:
    return ResolutionMarket(
        platform=platform,
        market_id=market_id,
        title=title,
        description=title,
        resolution_source="official source",
        resolution_criteria=title,
        close_time=close_time,
    )


def test_normalizers_map_expected_fields():
    p = normalize_polymarket_market(
        {
            "conditionId": "poly-1",
            "question": "Will BTC close above 100k?",
            "description": "Resolves on exchange close",
            "resolutionSource": "CoinGecko",
            "endDate": "2026-12-31T00:00:00Z",
        }
    )
    assert p.platform == "polymarket"
    assert p.market_id == "poly-1"

    k = normalize_kalshi_market(
        {
            "ticker": "KXBTC26",
            "title": "BTC over 100k by year-end",
            "subtitle": "Based on market close",
            "settlement_source": "CoinGecko",
            "rules_primary": "Settle to yes if above 100k",
            "close_time": "2026-12-31T00:00:00Z",
        }
    )
    assert k.platform == "kalshi"
    assert k.market_id == "KXBTC26"


def test_heuristic_similarity_prefers_semantically_close_market():
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    base = _mk_market("polymarket", "p1", "Will BTC be above 100k on Dec 31 2026?", now)
    near = _mk_market("kalshi", "k1", "BTC above 100000 by end of 2026", now)
    far = _mk_market("kalshi", "k2", "Will the Lakers win next game?", now)

    assert heuristic_similarity(base, near) > heuristic_similarity(base, far)


def test_rank_candidates_returns_sorted_top_k():
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    base = _mk_market("polymarket", "p1", "Will BTC be above 100k on Dec 31 2026?", now)
    candidates = [
        _mk_market("kalshi", "k1", "BTC above 100000 by end of 2026", now),
        _mk_market("kalshi", "k2", "Will the Lakers win next game?", now),
        _mk_market("kalshi", "k3", "Will BTC exceed 100k this year?", now),
    ]
    ranked = rank_candidates(base, candidates, top_k=2, min_score=0.05)
    assert len(ranked) == 2
    assert ranked[0].heuristic_score >= ranked[1].heuristic_score


class _FakeJudge:
    async def judge(self, base: ResolutionMarket, candidate: ResolutionMarket) -> EquivalenceDecision:
        if "btc" in candidate.title.lower():
            return EquivalenceDecision(same_event=True, confidence=0.82, rationale="Same BTC threshold event")
        return EquivalenceDecision(same_event=False, confidence=0.2, rationale="Different event")


@pytest.mark.asyncio
async def test_engine_with_fake_judge_returns_best_match():
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    base = _mk_market("polymarket", "p1", "Will BTC be above 100k on Dec 31 2026?", now)
    candidates = [
        _mk_market("kalshi", "k1", "BTC above 100000 by end of 2026", now),
        _mk_market("kalshi", "k2", "Will the Lakers win next game?", now),
    ]
    engine = ResolutionEquivalenceEngine(judge=_FakeJudge())
    match = await engine.find_best_match(base, candidates, top_k=2, llm_match_threshold=0.7)

    assert match is not None
    assert match.candidate.market_id == "k1"
    assert match.llm_decision is not None
    assert match.llm_decision.same_event is True


@pytest.mark.asyncio
async def test_engine_without_judge_requires_high_heuristic():
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    base = _mk_market("polymarket", "p1", "Will BTC be above 100k on Dec 31 2026?", now)
    candidates = [_mk_market("kalshi", "k2", "Will the Lakers win next game?", now)]
    engine = ResolutionEquivalenceEngine(judge=None)
    match = await engine.find_best_match(base, candidates, llm_match_threshold=0.7)

    assert match is None
