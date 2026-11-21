"""Promotion / rollback rules. Deterministic, in code, not in a prompt.

A candidate version promotes only if it beats or matches the active
version on BOTH scores:

- ``pass_rate`` (overall correctness), and
- ``refusal_accuracy`` (must-refuse cases actually refused).

Matching both is enough (a tie promotes — the new wording may be
clearer even at equal scores). Losing on either is a regression and
rolls back. With no active version yet, the first evaluated candidate
promotes (there is nothing to regress against).
"""

from __future__ import annotations

from typing import Optional

from .evals import EvalResult

PROMOTE = "promote"
ROLLBACK = "rollback"


def decide(
    candidate: EvalResult, active: Optional[EvalResult]
) -> tuple[str, str]:
    """Return (decision, human-readable reason)."""
    if active is None:
        return PROMOTE, (
            f"no active version yet; {candidate.version_id} becomes the "
            f"baseline (pass_rate={candidate.pass_rate:.3f}, "
            f"refusal_accuracy={candidate.refusal_accuracy:.3f})"
        )

    regressed = []
    if candidate.pass_rate < active.pass_rate:
        regressed.append(
            f"pass_rate {candidate.pass_rate:.3f} < active {active.pass_rate:.3f}"
        )
    if candidate.refusal_accuracy < active.refusal_accuracy:
        regressed.append(
            f"refusal_accuracy {candidate.refusal_accuracy:.3f} < active "
            f"{active.refusal_accuracy:.3f}"
        )

    if regressed:
        return ROLLBACK, (
            f"regression vs {active.version_id}: " + "; ".join(regressed)
            + f"; failing cases: {candidate.failures or 'none'}"
        )

    return PROMOTE, (
        f"matches or beats {active.version_id} "
        f"(pass_rate {candidate.pass_rate:.3f} >= {active.pass_rate:.3f}, "
        f"refusal_accuracy {candidate.refusal_accuracy:.3f} >= "
        f"{active.refusal_accuracy:.3f})"
    )
