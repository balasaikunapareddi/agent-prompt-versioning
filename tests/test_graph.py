"""LangGraph lifecycle: draft → evaluate → compare → promote/rollback."""

import json

import pytest

from agent_prompt_versioning.graph import run_lifecycle
from agent_prompt_versioning.prompts import V1_PROMPT, V2_BAD_LINE, V3_GOOD_LINE
from agent_prompt_versioning.store import (
    STATUS_ACTIVE,
    STATUS_ROLLED_BACK,
    STATUS_SUPERSEDED,
    VersionStore,
)

V2_PROMPT = V1_PROMPT.rstrip("\n") + "\n" + V2_BAD_LINE + "\n"
V3_PROMPT = V1_PROMPT.rstrip("\n") + "\n" + V3_GOOD_LINE + "\n"


@pytest.fixture
def store(tmp_path):
    return VersionStore(tmp_path / "store")


def test_first_version_promotes_to_active(store):
    result = run_lifecycle(store, V1_PROMPT, author_note="initial")
    assert result["decision"] == "promote"
    assert result["candidate_id"] == "v1"
    assert store.get_active().version_id == "v1"
    assert store.get_record("v1").status == STATUS_ACTIVE


def test_bad_line_rolls_back_and_v1_stays_active(store):
    run_lifecycle(store, V1_PROMPT, author_note="initial")
    result = run_lifecycle(store, V2_PROMPT, author_note="bad line", parent="v1")
    assert result["decision"] == "rollback"
    assert store.get_record("v2").status == STATUS_ROLLED_BACK
    assert store.get_active().version_id == "v1"


def test_good_line_promotes_and_supersedes_v1(store):
    run_lifecycle(store, V1_PROMPT, author_note="initial")
    result = run_lifecycle(store, V3_PROMPT, author_note="good line", parent="v1")
    assert result["decision"] == "promote"
    assert store.get_record("v2").status == STATUS_ACTIVE
    assert store.get_record("v1").status == STATUS_SUPERSEDED
    assert store.get_active().version_id == "v2"


def test_changelog_entry_written_with_reason(store):
    run_lifecycle(store, V1_PROMPT, author_note="initial")
    run_lifecycle(store, V2_PROMPT, author_note="bad line", parent="v1")
    entries = store.read_changelog()
    assert len(entries) == 2
    rollback_entry = entries[1]
    assert rollback_entry["candidate"] == "v2"
    assert rollback_entry["decision"] == "rollback"
    assert "regression" in rollback_entry["reason"]
    assert rollback_entry["candidate_scores"]["failures"]


def test_run_state_persisted_to_disk(store):
    result = run_lifecycle(store, V1_PROMPT, author_note="initial", run_id="run-test")
    path = store.runs_dir / "run-test.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "run-test"
    assert saved["decision"] == result["decision"]
    assert saved["candidate_scores"]["pass_rate"] == 11 / 12


def test_scores_recorded_for_both_sides(store):
    run_lifecycle(store, V1_PROMPT, author_note="initial")
    result = run_lifecycle(store, V3_PROMPT, author_note="good", parent="v1")
    assert result["candidate_scores"]["pass_rate"] == 1.0
    assert result["active_scores"]["pass_rate"] == 11 / 12
