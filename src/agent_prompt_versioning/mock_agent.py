"""Deterministic mock agent behavior: a pure function of prompt text.

In mock mode there is no LLM. Instead, the "agent" is a small rule
engine that reads directives out of the candidate system prompt and
applies them to each eval request. That makes prompt changes
*measurable* with zero network calls:

- A line like ``Refuse any request mentioning refunds.`` adds that
  topic to the refusal list — so valid refund questions start being
  refused, and the eval score drops. This is the Friday-tweak bug,
  reproduced deterministically.
- A line like ``Always verify the customer's identity before sharing
  account details.`` turns on the identity check — so requests for
  account details without verification get refused, and the score
  improves.

Base policy (independent of the prompt): refuse requests for another
person's data, requests to reveal the system prompt, fraud help, and
requests to bypass verification. Everything else is answered.

In live mode (``GEMINI_API_KEY`` set) the same eval cases are answered
by a real model using the candidate prompt; see ``evals.live_respond``.
"""

from __future__ import annotations

import re

ANSWER = "answer"
REFUSE = "refuse"

_REFUSE_TOPIC_RE = re.compile(r"refuse any request mentioning (\w+)", re.I)

# Requests the base policy always refuses, whatever the prompt says.
_ALWAYS_REFUSE_PATTERNS = [
    r"customer who ordered",          # another customer's data
    r"personal (phone|email|details)",  # someone else's contact info
    r"system prompt",                  # prompt extraction
    r"fake chargeback",                # fraud
    r"skip the identity check",        # bypass verification
]

# Requests for the caller's own account details. Only refused when the
# prompt turns on the identity check — this is the hole the "good
# line" closes.
_ACCOUNT_DETAIL_PATTERNS = [
    r"phone number on file",
    r"account balance",
    r"email address on file",
]


def parse_directives(prompt_text: str) -> dict:
    """Extract machine-readable directives from a system prompt."""
    topics = {m.group(1).lower() for m in _REFUSE_TOPIC_RE.finditer(prompt_text)}
    lowered = prompt_text.lower()
    identity_check = "verify" in lowered and "identity" in lowered
    return {"refuse_topics": topics, "identity_check": identity_check}


def mock_respond(prompt_text: str, request: str) -> str:
    """Return ANSWER or REFUSE for one request under one prompt."""
    directives = parse_directives(prompt_text)
    text = request.lower()

    for topic in directives["refuse_topics"]:
        # crude stem match so "refunds" in the prompt catches "refund"
        if topic.rstrip("s") in text:
            return REFUSE

    for pattern in _ALWAYS_REFUSE_PATTERNS:
        if re.search(pattern, text):
            return REFUSE

    if directives["identity_check"]:
        for pattern in _ACCOUNT_DETAIL_PATTERNS:
            if re.search(pattern, text):
                return REFUSE

    return ANSWER
