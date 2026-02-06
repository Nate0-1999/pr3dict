"""Cross-platform resolution-event equivalence for dispute strategy."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Protocol

import httpx

DEFAULT_EQUIVALENCE_MODEL = "google/gemini-2.5-flash"


@dataclass
class ResolutionMarket:
    platform: str
    market_id: str
    title: str
    description: str
    resolution_source: str
    resolution_criteria: str
    close_time: Optional[datetime] = None

    @property
    def full_text(self) -> str:
        return " ".join(
            [
                self.title or "",
                self.description or "",
                self.resolution_source or "",
                self.resolution_criteria or "",
            ]
        ).strip()


@dataclass
class EquivalenceDecision:
    same_event: bool
    confidence: float
    rationale: str


@dataclass
class CandidateScore:
    candidate: ResolutionMarket
    heuristic_score: float


@dataclass
class EquivalenceMatch:
    candidate: ResolutionMarket
    heuristic_score: float
    llm_decision: Optional[EquivalenceDecision]
    combined_score: float


class EquivalenceJudge(Protocol):
    async def judge(self, base: ResolutionMarket, candidate: ResolutionMarket) -> EquivalenceDecision:
        pass


def normalize_polymarket_market(payload: dict) -> ResolutionMarket:
    end_date = payload.get("endDate")
    close_time = _parse_iso_datetime(end_date) if end_date else None
    return ResolutionMarket(
        platform="polymarket",
        market_id=str(payload.get("conditionId") or payload.get("id") or ""),
        title=str(payload.get("question") or ""),
        description=str(payload.get("description") or ""),
        resolution_source=str(payload.get("resolutionSource") or ""),
        resolution_criteria=str(payload.get("description") or ""),
        close_time=close_time,
    )


def normalize_kalshi_market(payload: dict) -> ResolutionMarket:
    close_time_raw = payload.get("close_time")
    close_time = _parse_iso_datetime(close_time_raw) if close_time_raw else None
    title = str(payload.get("title") or payload.get("yes_sub_title") or payload.get("ticker") or "")
    description = str(payload.get("subtitle") or payload.get("description") or "")
    rules = str(payload.get("rules_primary") or payload.get("rules") or payload.get("settlement_source") or "")
    return ResolutionMarket(
        platform="kalshi",
        market_id=str(payload.get("ticker") or payload.get("id") or ""),
        title=title,
        description=description,
        resolution_source=str(payload.get("settlement_source") or ""),
        resolution_criteria=rules,
        close_time=close_time,
    )


def rank_candidates(
    base: ResolutionMarket,
    candidates: Iterable[ResolutionMarket],
    top_k: int = 5,
    min_score: float = 0.15,
) -> list[CandidateScore]:
    scored: list[CandidateScore] = []
    for candidate in candidates:
        score = heuristic_similarity(base, candidate)
        if score >= min_score:
            scored.append(CandidateScore(candidate=candidate, heuristic_score=score))
    scored.sort(key=lambda s: s.heuristic_score, reverse=True)
    return scored[:top_k]


def heuristic_similarity(a: ResolutionMarket, b: ResolutionMarket) -> float:
    lexical = _jaccard_similarity(_tokenize(a.full_text), _tokenize(b.full_text))
    numeric = _jaccard_similarity(_extract_number_tokens(a.full_text), _extract_number_tokens(b.full_text))
    time = _close_time_similarity(a.close_time, b.close_time, max_days=30.0)
    # Weighted blend; lexical dominates, then timing, then numeric overlap.
    return max(0.0, min(1.0, (0.6 * lexical) + (0.25 * time) + (0.15 * numeric)))


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) >= 3}


def _extract_number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text.lower()))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    inter = len(left.intersection(right))
    union = len(left.union(right))
    return inter / union if union else 0.0


def _close_time_similarity(a: Optional[datetime], b: Optional[datetime], max_days: float) -> float:
    if a is None or b is None:
        return 0.5
    diff_days = abs((a - b).total_seconds()) / 86400.0
    if diff_days >= max_days:
        return 0.0
    return 1.0 - (diff_days / max_days)


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class OpenRouterEquivalenceJudge:
    """Optional fuzzy judge for event equivalence via OpenRouter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("RESOLUTION_MATCH_MODEL") or DEFAULT_EQUIVALENCE_MODEL
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterEquivalenceJudge")

    async def judge(self, base: ResolutionMarket, candidate: ResolutionMarket) -> EquivalenceDecision:
        system_prompt = (
            "Decide if two prediction markets resolve the same real-world event.\n"
            "Use strict JSON only: "
            '{"same_event": boolean, "confidence": number, "rationale": string}.'
        )
        user_prompt = (
            f"Market A ({base.platform} / {base.market_id})\n"
            f"Title: {base.title}\n"
            f"Description: {base.description}\n"
            f"Resolution source: {base.resolution_source}\n"
            f"Resolution criteria: {base.resolution_criteria}\n"
            f"Close time: {base.close_time}\n\n"
            f"Market B ({candidate.platform} / {candidate.market_id})\n"
            f"Title: {candidate.title}\n"
            f"Description: {candidate.description}\n"
            f"Resolution source: {candidate.resolution_source}\n"
            f"Resolution criteria: {candidate.resolution_criteria}\n"
            f"Close time: {candidate.close_time}\n"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

        parsed = _parse_json_response(content)
        return EquivalenceDecision(
            same_event=bool(parsed.get("same_event")),
            confidence=float(parsed.get("confidence", 0.0)),
            rationale=str(parsed.get("rationale", "")),
        )


def _parse_json_response(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


class ResolutionEquivalenceEngine:
    """Heuristic narrowing + optional LLM fuzzy equivalence for cross-platform markets."""

    def __init__(self, judge: Optional[EquivalenceJudge] = None):
        self.judge = judge

    async def find_best_match(
        self,
        base: ResolutionMarket,
        candidates: Iterable[ResolutionMarket],
        top_k: int = 5,
        min_heuristic_score: float = 0.15,
        llm_match_threshold: float = 0.7,
    ) -> Optional[EquivalenceMatch]:
        ranked = rank_candidates(base, candidates, top_k=top_k, min_score=min_heuristic_score)
        if not ranked:
            return None

        if self.judge is None:
            best = ranked[0]
            if best.heuristic_score < llm_match_threshold:
                return None
            return EquivalenceMatch(
                candidate=best.candidate,
                heuristic_score=best.heuristic_score,
                llm_decision=None,
                combined_score=best.heuristic_score,
            )

        best_match: Optional[EquivalenceMatch] = None
        for scored in ranked:
            llm_decision = await self.judge.judge(base, scored.candidate)
            if not llm_decision.same_event:
                continue
            combined = (0.4 * scored.heuristic_score) + (0.6 * llm_decision.confidence)
            if llm_decision.confidence < llm_match_threshold:
                continue
            if best_match is None or combined > best_match.combined_score:
                best_match = EquivalenceMatch(
                    candidate=scored.candidate,
                    heuristic_score=scored.heuristic_score,
                    llm_decision=llm_decision,
                    combined_score=combined,
                )
        return best_match
