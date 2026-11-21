"""Version store: creation, metadata, listing, status, diff."""

import pytest

from agent_prompt_versioning.store import (
    STATUS_ACTIVE,
    STATUS_DRAFT,
    STATUS_ROLLED_BACK,
    VersionStore,
)


@pytest.fixture
def store(tmp_path):
    return VersionStore(tmp_path / "store")


def test_create_version_writes_text_and_metadata(store):
    record = store.create_version("You are helpful.", author_note="first")
    assert record.version_id == "v1"
    assert record.status == STATUS_DRAFT
    assert record.parent is None
    assert record.author_note == "first"
    assert record.created_at  # ISO timestamp present
    assert store.get_text("v1") == "You are helpful."
    # both files exist on disk
    assert (store.versions_dir / "v1.txt").exists()
    assert (store.versions_dir / "v1.json").exists()


def test_version_ids_are_sequential_and_never_overwrite(store):
    store.create_version("one")
    store.create_version("two")
    store.create_version("three")
    assert [r.version_id for r in store.list_versions()] == ["v1", "v2", "v3"]
    assert store.get_text("v1") == "one"


def test_parent_must_exist(store):
    with pytest.raises(ValueError):
        store.create_version("orphan", parent="v99")


def test_parent_is_recorded(store):
    store.create_version("base")
    child = store.create_version("child", parent="v1")
    assert child.parent == "v1"


def test_get_active_none_then_one(store):
    assert store.get_active() is None
    store.create_version("a")
    store.create_version("b")
    store.set_status("v2", STATUS_ACTIVE)
    assert store.get_active().version_id == "v2"


def test_set_status_persists(store):
    store.create_version("a")
    store.set_status("v1", STATUS_ROLLED_BACK)
    assert store.get_record("v1").status == STATUS_ROLLED_BACK


def test_unknown_version_raises(store):
    with pytest.raises(KeyError):
        store.get_text("v42")


def test_diff_shows_added_line(store):
    store.create_version("line one\nline two\n")
    store.create_version("line one\nline two\nline three\n")
    diff = store.diff_versions("v1", "v2")
    assert "+line three" in diff
    assert "--- v1.txt" in diff
    assert "+++ v2.txt" in diff


def test_diff_identical_versions_is_empty(store):
    store.create_version("same\n")
    store.create_version("same\n")
    assert store.diff_versions("v1", "v2") == ""


def test_changelog_roundtrip(store):
    store.append_changelog({"candidate": "v1", "decision": "promote"})
    entries = store.read_changelog()
    assert len(entries) == 1
    assert entries[0]["candidate"] == "v1"
    assert entries[0]["timestamp"]
