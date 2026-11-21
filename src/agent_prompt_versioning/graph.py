"""The prompt lifecycle as a LangGraph state machine.

    draft ──► evaluate ──► compare ──┬─► promote ──┐
                                     └─► rollback ─┴─► changelog ──► END

Why a graph: the lifecycle has a real branch (promote vs rollback)
that depends on computed scores, and every step needs the state the
previous step produced. That is exactly a state machine — LangGraph
gives us typed state, conditional edges, and a checkpointer so each
run's state is persisted and resumable/inspectable.

State is persisted two ways per run:
- the LangGraph checkpointer (``MemorySaver`` by default), keyed by
  ``thread_id`` = run id, and
- a JSON snapshot under ``<store>/runs/<run_id>.json`` so the run is
  readable without the graph runtime.
"""

from __future__ import annotations

import uuid
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .evals import EvalResult, default_respond, score_prompt
from .rules import PROMOTE, decide
from .store import (
    STATUS_ACTIVE,
    STATUS_ROLLED_BACK,
    STATUS_SUPERSEDED,
    VersionStore,
)


class LifecycleState(TypedDict, total=False):
    # inputs
    prompt_text: str
    author_note: str
    parent: Optional[str]
    # produced by nodes
    candidate_id: str
    active_id: Optional[str]
    candidate_scores: dict
    active_scores: Optional[dict]
    decision: str
    reason: str
    changelog_entry: dict


def build_graph(store: VersionStore, checkpointer=None):
    """Compile the lifecycle graph bound to one store."""
    respond = default_respond()

    def draft(state: LifecycleState) -> LifecycleState:
        record = store.create_version(
            state["prompt_text"],
            author_note=state.get("author_note", ""),
            parent=state.get("parent"),
        )
        active = store.get_active()
        return {
            "candidate_id": record.version_id,
            "active_id": active.version_id if active else None,
        }

    def evaluate(state: LifecycleState) -> LifecycleState:
        candidate = score_prompt(
            store.get_text(state["candidate_id"]),
            respond=respond,
            version_id=state["candidate_id"],
        )
        out: LifecycleState = {"candidate_scores": candidate.to_dict()}
        if state.get("active_id"):
            active = score_prompt(
                store.get_text(state["active_id"]),
                respond=respond,
                version_id=state["active_id"],
            )
            out["active_scores"] = active.to_dict()
        else:
            out["active_scores"] = None
        return out

    def compare(state: LifecycleState) -> LifecycleState:
        candidate = EvalResult(**state["candidate_scores"])
        active = (
            EvalResult(**state["active_scores"])
            if state.get("active_scores")
            else None
        )
        decision, reason = decide(candidate, active)
        return {"decision": decision, "reason": reason}

    def promote(state: LifecycleState) -> LifecycleState:
        if state.get("active_id"):
            store.set_status(state["active_id"], STATUS_SUPERSEDED)
        store.set_status(state["candidate_id"], STATUS_ACTIVE)
        return {}

    def rollback(state: LifecycleState) -> LifecycleState:
        store.set_status(state["candidate_id"], STATUS_ROLLED_BACK)
        return {}

    def changelog(state: LifecycleState) -> LifecycleState:
        entry = store.append_changelog(
            {
                "candidate": state["candidate_id"],
                "previous_active": state.get("active_id"),
                "decision": state["decision"],
                "reason": state["reason"],
                "candidate_scores": state["candidate_scores"],
                "active_scores": state.get("active_scores"),
                "author_note": state.get("author_note", ""),
            }
        )
        return {"changelog_entry": entry}

    graph = StateGraph(LifecycleState)
    graph.add_node("draft", draft)
    graph.add_node("evaluate", evaluate)
    graph.add_node("compare", compare)
    graph.add_node("promote", promote)
    graph.add_node("rollback", rollback)
    graph.add_node("changelog", changelog)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "evaluate")
    graph.add_edge("evaluate", "compare")
    graph.add_conditional_edges(
        "compare",
        lambda s: s["decision"],
        {PROMOTE: "promote", "rollback": "rollback"},
    )
    graph.add_edge("promote", "changelog")
    graph.add_edge("rollback", "changelog")
    graph.add_edge("changelog", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_lifecycle(
    store: VersionStore,
    prompt_text: str,
    author_note: str = "",
    parent: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Run one full lifecycle: draft → evaluate → compare → decide.

    Returns the final state (also persisted to ``<store>/runs/``).
    """
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    app = build_graph(store)
    final = app.invoke(
        {
            "prompt_text": prompt_text,
            "author_note": author_note,
            "parent": parent,
        },
        config={"configurable": {"thread_id": run_id}},
    )
    snapshot = {"run_id": run_id, **final}
    store.save_run(run_id, snapshot)
    return snapshot
