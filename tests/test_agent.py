"""PromptVersioningAgent: wires store, system prompt, tools, lifecycle."""

import pytest

from agent_prompt_versioning import AGENT_SYSTEM_PROMPT, PromptVersioningAgent
from agent_prompt_versioning.prompts import V1_PROMPT
from agent_prompt_versioning.store import STATUS_ACTIVE, VersionStore


@pytest.fixture
def agent(tmp_path):
    return PromptVersioningAgent(VersionStore(tmp_path / "store"))


def test_agent_defaults_to_prompts_module_system_prompt(agent):
    assert agent.system_prompt == AGENT_SYSTEM_PROMPT
    assert len(agent.tools) == 4


def test_agent_run_promotes_first_version(agent):
    result = agent.run(V1_PROMPT, author_note="initial")
    assert result["decision"] == "promote"
    assert result["candidate_id"] == "v1"
    assert agent.store.get_active().version_id == "v1"
    assert agent.store.get_record("v1").status == STATUS_ACTIVE


def test_agent_evaluate_scores_stored_version(agent):
    agent.run(V1_PROMPT, author_note="initial")
    scores = agent.evaluate("v1")
    assert scores["version_id"] == "v1"
    assert scores["pass_rate"] == 11 / 12
