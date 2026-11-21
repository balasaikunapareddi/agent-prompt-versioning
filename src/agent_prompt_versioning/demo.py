"""One-command demo: the Friday tweak, caught before Monday.

    python -m agent_prompt_versioning.demo [store_dir]

Story:
1. v1 — a good support-agent prompt. Evaluated, promoted to active.
2. v2 — v1 plus one bad line ("Refuse any request mentioning
   refunds."). Evals catch the regression → automatic rollback.
3. v3 — v1 plus a genuinely better line (identity check before
   sharing account details). Evals improve → promoted.

Prints scores, decisions, and the changelog. Runs fully offline in
mock mode (default). Set GEMINI_API_KEY for live evals.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .graph import run_lifecycle
from .prompts import V1_PROMPT, V2_BAD_LINE, V3_GOOD_LINE
from .store import VersionStore


def _show(label: str, result: dict) -> None:
    scores = result["candidate_scores"]
    print(
        f"[{label}] {result['candidate_id']}: decision={result['decision']} "
        f"pass_rate={scores['pass_rate']:.3f} "
        f"refusal_accuracy={scores['refusal_accuracy']:.3f} "
        f"failures={scores['failures']}"
    )
    print(f"    reason: {result['reason']}")


def run_demo(store_dir: str | Path) -> VersionStore:
    store = VersionStore(store_dir)
    print("== Prompt versioning demo ==\n")

    print("Step 1: ship the good v1 prompt")
    r1 = run_lifecycle(store, V1_PROMPT, author_note="initial support prompt")
    _show("v1", r1)

    print("\nStep 2: Friday — someone adds one line to v2:")
    print(f"    + {V2_BAD_LINE}")
    print("    (diff v1 -> v2 is one line; evals decide if it ships)")
    r2 = run_lifecycle(
        store,
        V1_PROMPT.rstrip("\n") + "\n" + V2_BAD_LINE + "\n",
        author_note="tighten refund handling",
        parent="v1",
    )
    _show("v2", r2)
    print("    diff:")
    for line in store.diff_versions("v1", "v2").splitlines():
        print(f"    {line}")

    print("\nStep 3: a genuinely better line for v3:")
    print(f"    + {V3_GOOD_LINE}")
    r3 = run_lifecycle(
        store,
        V1_PROMPT.rstrip("\n") + "\n" + V3_GOOD_LINE + "\n",
        author_note="verify identity before sharing account details",
        parent="v1",
    )
    _show("v3", r3)

    active = store.get_active()
    print(f"\nActive version now: {active.version_id if active else 'none'}")

    print("\n== Changelog ==")
    for entry in store.read_changelog():
        print(
            f"- {entry['timestamp']} candidate={entry['candidate']} "
            f"decision={entry['decision']}: {entry['reason']}"
        )
    return store


def main() -> None:
    if len(sys.argv) > 1:
        store_dir = sys.argv[1]
    else:
        store_dir = tempfile.mkdtemp(prefix="prompt-versioning-demo-")
    print(f"Store: {store_dir}\n")
    run_demo(store_dir)


if __name__ == "__main__":
    main()
