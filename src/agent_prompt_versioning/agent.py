"""The prompt-versioning agent.

Behavior lives in the system prompt (``prompts.AGENT_SYSTEM_PROMPT``);
mechanics live in code. The agent loop is the LangGraph lifecycle in
``graph.py``: draft → evaluate → compare → promote/rollback →
changelog. :class:`PromptVersioningAgent` is the one readable place
that wires the pieces together — version store, system prompt,
tools, checkpointer — and exposes the loop as ``run()``.

The module-level functions in ``graph.py`` (``build_graph``,
``run_lifecycle``) remain the engine and stay importable for
existing callers; this class delegates to them unchanged.
"""

from __future__ import annotations

from typing import Optional

from .evals import default_respond, score_prompt
from .graph import build_graph as _build_lifecycle_graph
from .graph import run_lifecycle as _run_lifecycle
from .prompts import AGENT_SYSTEM_PROMPT
from .store import VersionStore
from .tools import make_tools


class PromptVersioningAgent:
    """Prompt release manager: version, eval, promote, roll back.

    One agent instance is bound to one :class:`VersionStore`. Its
    system prompt (behavior only — "report scores exactly, never
    promote on vibes") comes from ``prompts.py``; every checkable
    rule (scoring, promotion, rollback) lives in code. The tool set
    is the typed surface from ``tools.py``, built against the same
    store.

    The agent loop is one lifecycle pass per candidate prompt:
    draft the version, evaluate it (and the active version) against
    the fixed eval set, compare scores under the rules in
    ``rules.py``, promote or roll back, and write the changelog
    entry. ``run()`` executes that loop via the LangGraph state
    machine in ``graph.py``; ``evaluate()`` exposes the evaluate
    step on its own.
    """

    def __init__(
        self,
        store: VersionStore,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        checkpointer=None,
    ):
        self.store = store
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.tools = make_tools(store)

    # -- agent loop ----------------------------------------------------

    def run(
        self,
        prompt_text: str,
        author_note: str = "",
        parent: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict:
        """Run one full lifecycle: draft → evaluate → compare → decide.

        Returns the final state (also persisted to ``<store>/runs/``).
        """
        return _run_lifecycle(
            self.store,
            prompt_text,
            author_note=author_note,
            parent=parent,
            run_id=run_id,
            checkpointer=self.checkpointer,
        )

    def evaluate(self, version_id: str) -> dict:
        """Score one stored version against the fixed eval set."""
        result = score_prompt(
            self.store.get_text(version_id),
            respond=default_respond(),
            version_id=version_id,
        )
        return result.to_dict()

    def build_graph(self):
        """Compile the lifecycle graph bound to this agent's store."""
        return _build_lifecycle_graph(
            self.store, checkpointer=self.checkpointer
        )
