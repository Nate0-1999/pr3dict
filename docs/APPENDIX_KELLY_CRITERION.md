# Appendix: Kelly Criterion

**Position Sizing From First Principles**

---

## The Problem Kelly Was Solving

In 1956, John Kelly was a physicist at Bell Labs working on information theory. He was interested in a practical question:

> **If you have an edge in a repeated bet, how much should you wager each time to maximize long-term wealth?**

The naive answer is "bet everything" — if you have an edge, more is better. But this leads to ruin: one loss and you're wiped out.

The opposite extreme — betting a tiny fixed amount — is safe but leaves money on the table.

Kelly wanted the optimal middle ground: **the bet size that maximizes the expected growth rate of your bankroll over many repeated bets.**

---

## The Insight

Kelly realized this was an information theory problem. He framed it as: you have a noisy channel (your edge) transmitting information (the correct bet). How do you maximize the information rate?

His key insight: **you should bet a fraction of your bankroll proportional to your edge.**

---

## The Formula

For a simple bet with probability `p` of winning and odds `b` (you get back `b` times your bet if you win):

```
f* = (bp - q) / b

where:
  f* = fraction of bankroll to bet
  p  = probability of winning
  q  = probability of losing (1 - p)
  b  = decimal odds - 1 (what you win per dollar risked)
```

**Example:**
- You have 60% chance to win (p = 0.6)
- Odds are even money (b = 1, you win $1 for every $1 risked)

```
f* = (1 × 0.6 - 0.4) / 1 = 0.2
```

You should bet 20% of your bankroll.

---

## Why It Works

Kelly sizing has a special property: it maximizes the **geometric mean** of returns (equivalently, the expected log of wealth). This means:

1. **You never go broke** — You're always betting a fraction, never everything
2. **You grow faster than any other strategy** — In the long run, Kelly beats all other fixed-fraction approaches
3. **Bet size scales with edge** — Bigger edge = bigger bet, no edge = no bet

---

## Why We Use Fractional Kelly

Full Kelly is aggressive. In practice, we use a fraction (like 25%) because:
- Our probability estimates have uncertainty
- Drawdowns with full Kelly can be 50%+ (psychologically brutal)
- Model errors compound; fractional Kelly provides a buffer

For dispute trading specifically, we also discount for:
- Confidence in our dispute prediction
- Probability of INVALID resolution (both sides lose)

---

## Further Reading

- Kelly, J.L. (1956). "A New Interpretation of Information Rate" — The original paper
- Thorp, E.O. "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" — Practical applications

---

*PR3DICT Documentation*
