"""Promotion / rollback rules."""

from agent_prompt_versioning.evals import EvalResult
from agent_prompt_versioning.rules import PROMOTE, ROLLBACK, decide


def _result(vid, pass_rate, refusal_accuracy, failures=None):
    return EvalResult(
        version_id=vid,
        pass_rate=pass_rate,
        refusal_accuracy=refusal_accuracy,
        passed=round(pass_rate * 12),
        total=12,
        failures=failures or [],
    )


def test_first_candidate_promotes_with_no_active():
    decision, reason = decide(_result("v1", 0.5, 0.5), None)
    assert decision == PROMOTE
    assert "no active version" in reason


def test_improvement_promotes():
    decision, _ = decide(_result("v2", 1.0, 1.0), _result("v1", 0.9, 0.8))
    assert decision == PROMOTE


def test_tie_promotes():
    decision, _ = decide(_result("v2", 0.9, 0.8), _result("v1", 0.9, 0.8))
    assert decision == PROMOTE


def test_pass_rate_regression_rolls_back():
    decision, reason = decide(
        _result("v2", 0.75, 0.83, failures=["valid-refund-status"]),
        _result("v1", 0.92, 0.83),
    )
    assert decision == ROLLBACK
    assert "pass_rate" in reason
    assert "valid-refund-status" in reason


def test_refusal_accuracy_regression_rolls_back():
    # Overall pass rate ties, but the candidate stopped refusing
    # things it must refuse — still a regression.
    decision, reason = decide(
        _result("v2", 0.92, 0.5), _result("v1", 0.92, 0.83)
    )
    assert decision == ROLLBACK
    assert "refusal_accuracy" in reason


def test_mixed_tradeoff_rolls_back():
    # Better pass rate, worse refusal accuracy → rollback. Both must hold.
    decision, _ = decide(_result("v2", 1.0, 0.5), _result("v1", 0.9, 0.83))
    assert decision == ROLLBACK
