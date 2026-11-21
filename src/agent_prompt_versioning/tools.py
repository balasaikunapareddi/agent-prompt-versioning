"""Typed tool definitions for the prompt-versioning agent.

Each tool has a pydantic input schema, so calls are validated before
they touch the store. The LangGraph lifecycle (``graph.py``) drives
the same underlying functions; these tools are the agent-facing
surface (and what a live LLM would call).
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .evals import default_respond, score_prompt
from .store import VersionStore


class CreateVersionInput(BaseModel):
    prompt_text: str = Field(description="Full text of the new system prompt")
    author_note: str = Field(
        default="", description="Why this version exists (one line)"
    )
    parent: Optional[str] = Field(
        default=None, description="Version id this one is based on"
    )


class DiffVersionsInput(BaseModel):
    from_id: str = Field(description="Base version id, e.g. 'v1'")
    to_id: str = Field(description="Version id to compare against, e.g. 'v2'")


class VersionIdInput(BaseModel):
    version_id: str = Field(description="Version id, e.g. 'v2'")


def make_tools(store: VersionStore) -> list:
    """Build the tool set bound to one store instance."""

    @tool(args_schema=CreateVersionInput)
    def create_version(
        prompt_text: str, author_note: str = "", parent: Optional[str] = None
    ) -> dict:
        """Create a new draft prompt version. Never overwrites."""
        record = store.create_version(prompt_text, author_note, parent)
        return record.to_dict()

    @tool(args_schema=DiffVersionsInput)
    def diff_versions(from_id: str, to_id: str) -> str:
        """Unified diff between two prompt versions."""
        return store.diff_versions(from_id, to_id)

    @tool
    def list_versions() -> list[dict]:
        """List all prompt versions with metadata and status."""
        return [r.to_dict() for r in store.list_versions()]

    @tool(args_schema=VersionIdInput)
    def run_evals(version_id: str) -> dict:
        """Run the fixed eval set against one version and score it."""
        result = score_prompt(
            store.get_text(version_id),
            respond=default_respond(),
            version_id=version_id,
        )
        return result.to_dict()

    return [create_version, diff_versions, list_versions, run_evals]
