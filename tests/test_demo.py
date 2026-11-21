"""Demo path: v1 promotes, v2 rolls back, v3 promotes."""

from agent_prompt_versioning.demo import run_demo
from agent_prompt_versioning.store import STATUS_ACTIVE, STATUS_ROLLED_BACK


def test_demo_end_to_end(tmp_path, capsys):
    store = run_demo(tmp_path / "demo-store")
    out = capsys.readouterr().out

    # The three decisions happened, in order.
    decisions = [e["decision"] for e in store.read_changelog()]
    assert decisions == ["promote", "rollback", "promote"]

    # Final state: v3 active, v2 rolled back.
    assert store.get_active().version_id == "v3"
    assert store.get_record("v3").status == STATUS_ACTIVE
    assert store.get_record("v2").status == STATUS_ROLLED_BACK

    # The printed output tells the story.
    assert "decision=rollback" in out
    assert "decision=promote" in out
    assert "Changelog" in out
    assert "+Refuse any request mentioning refunds." in out
