"""
Combinatorial dependency detection for prediction markets.

Design:
- Stage 1 deterministic narrowing/assessment
- Stage 2 optional external verifier (e.g., LLM) for ambiguous pairs
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Protocol, Sequence, Tuple

from ..platforms.base import Market


class DependencyRelation(str, Enum):
    EQUIVALENT = "equivalent"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    IMPLIES = "implies"
    INDEPENDENT = "independent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DependencyAssessment:
    market_a_id: str
    market_b_id: str
    relation: DependencyRelation
    confidence: float
    reason: str
    source: str = "deterministic"


class DependencyVerifierPort(Protocol):
    """Optional second-stage verifier for ambiguous dependency pairs."""

    def verify(
        self,
        market_a: Market,
        market_b: Market,
        deterministic: DependencyAssessment,
    ) -> Optional[DependencyAssessment]:
        """Return refined assessment, or None to keep deterministic result."""


class DependencyDetector:
    """
    Detect dependencies between markets for combinatorial arb constraints.

    The detector is intentionally conservative:
    - deterministic stage only outputs high-confidence relations
    - ambiguous cases remain UNKNOWN until verified
    """

    _stopwords = {
        "will",
        "the",
        "a",
        "an",
        "be",
        "is",
        "are",
        "to",
        "of",
        "in",
        "on",
        "for",
        "by",
        "and",
        "or",
        "at",
        "with",
        "from",
    }

    _identity_markers = {"win", "wins", "elected", "president", "nominee", "nomination", "primary"}

    _year_re = re.compile(r"\b(20[2-4]\d)\b")
    _word_re = re.compile(r"[a-z0-9]+")

    def detect(
        self,
        markets: Sequence[Market],
        verifier: Optional[DependencyVerifierPort] = None,
    ) -> List[DependencyAssessment]:
        """Run full dependency detection flow on all candidate pairs."""
        assessments: List[DependencyAssessment] = []
        for a, b in self.generate_candidates(markets):
            deterministic = self.assess_pair(a, b)
            refined = verifier.verify(a, b, deterministic) if verifier is not None else None
            assessments.append(refined or deterministic)
        return assessments

    def generate_candidates(self, markets: Sequence[Market]) -> List[Tuple[Market, Market]]:
        """
        Deterministically narrow pair space.

        Candidate rules:
        - same platform and unresolved
        - close times within 45 days
        - matching year token when present
        - at least 2 shared event signature tokens
        """
        candidates: List[Tuple[Market, Market]] = []
        n = len(markets)
        for i in range(n):
            for j in range(i + 1, n):
                a = markets[i]
                b = markets[j]
                if a.platform != b.platform:
                    continue
                if a.resolved or b.resolved:
                    continue
                if abs((a.close_time - b.close_time).total_seconds()) > 45 * 24 * 3600:
                    continue

                year_a = self._extract_year(a.title)
                year_b = self._extract_year(b.title)
                if year_a is not None and year_b is not None and year_a != year_b:
                    continue

                sig_a = self._event_signature_tokens(a.title)
                sig_b = self._event_signature_tokens(b.title)
                if len(sig_a.intersection(sig_b)) < 2:
                    continue

                candidates.append((a, b))
        return candidates

    def assess_pair(self, market_a: Market, market_b: Market) -> DependencyAssessment:
        """Deterministic relation assessment for one market pair."""
        text_a = self._normalize(market_a.title)
        text_b = self._normalize(market_b.title)

        if text_a == text_b:
            return DependencyAssessment(
                market_a_id=market_a.id,
                market_b_id=market_b.id,
                relation=DependencyRelation.EQUIVALENT,
                confidence=0.99,
                reason="identical normalized title",
            )

        focus_a = self._focus_tokens(market_a.title)
        focus_b = self._focus_tokens(market_b.title)
        sig_a = self._event_signature_tokens(market_a.title)
        sig_b = self._event_signature_tokens(market_b.title)
        overlap = sig_a.intersection(sig_b)

        if focus_a and focus_b and focus_a == focus_b and len(overlap) >= 3:
            return DependencyAssessment(
                market_a_id=market_a.id,
                market_b_id=market_b.id,
                relation=DependencyRelation.EQUIVALENT,
                confidence=0.8,
                reason="same focus entity and event signature overlap",
            )

        if focus_a and focus_b and focus_a != focus_b and len(overlap) >= 3:
            return DependencyAssessment(
                market_a_id=market_a.id,
                market_b_id=market_b.id,
                relation=DependencyRelation.MUTUALLY_EXCLUSIVE,
                confidence=0.72,
                reason="different focus entities on same event signature",
            )

        a_nom = {"nominee", "nomination", "primary"}.intersection(sig_a)
        b_nom = {"nominee", "nomination", "primary"}.intersection(sig_b)
        a_win = {"win", "wins", "elected", "president"}.intersection(sig_a)
        b_win = {"win", "wins", "elected", "president"}.intersection(sig_b)

        if focus_a and focus_b and focus_a == focus_b:
            if a_nom and b_win:
                return DependencyAssessment(
                    market_a_id=market_a.id,
                    market_b_id=market_b.id,
                    relation=DependencyRelation.IMPLIES,
                    confidence=0.66,
                    reason="nomination/primary phrasing implies election-win path",
                )
            if b_nom and a_win:
                return DependencyAssessment(
                    market_a_id=market_a.id,
                    market_b_id=market_b.id,
                    relation=DependencyRelation.IMPLIES,
                    confidence=0.66,
                    reason="nomination/primary phrasing implies election-win path",
                )

        return DependencyAssessment(
            market_a_id=market_a.id,
            market_b_id=market_b.id,
            relation=DependencyRelation.UNKNOWN,
            confidence=0.35,
            reason="insufficient deterministic evidence",
        )

    def _extract_year(self, text: str) -> Optional[int]:
        m = self._year_re.search(text.lower())
        return int(m.group(1)) if m else None

    def _normalize(self, text: str) -> str:
        tokens = self._word_re.findall(text.lower())
        return " ".join(tokens)

    def _focus_tokens(self, text: str) -> Tuple[str, ...]:
        """
        Extract probable focus entity from leading clause.

        Example:
        - "Will Donald Trump win..." -> ("donald", "trump")
        - "Will Biden be..." -> ("biden",)
        """
        normalized = self._normalize(text)
        tokens = normalized.split()
        if tokens and tokens[0] == "will":
            tokens = tokens[1:]
        verbs = {"win", "wins", "be", "become", "get", "receive"}

        focus: List[str] = []
        for tok in tokens:
            if tok in verbs:
                break
            if tok in self._stopwords:
                continue
            if tok.isdigit():
                continue
            focus.append(tok)
            if len(focus) >= 3:
                break
        return tuple(focus)

    def _event_signature_tokens(self, text: str) -> set[str]:
        tokens = self._word_re.findall(text.lower())
        focus = set(self._focus_tokens(text))
        signature = set()
        for tok in tokens:
            if tok in self._stopwords:
                continue
            if tok in focus:
                continue
            if tok.isdigit():
                signature.add(tok)
                continue
            if tok in self._identity_markers:
                signature.add(tok)
                continue
            # Keep content tokens with at least 4 chars to reduce noise.
            if len(tok) >= 4:
                signature.add(tok)
        return signature
