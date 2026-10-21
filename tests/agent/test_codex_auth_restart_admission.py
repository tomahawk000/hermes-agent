"""Restart continuations cannot revive a durably closed Codex auth failure."""
from unittest.mock import Mock

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_cross_process_turn_lease import _agent_with_db
from tests.agent.test_codex_auth_tip_admission import rotate


def _closed_failure(db, sid, holder=None, *, effects=False):
    db.append_message(sid, "user", "failed fixture action", turn_lease_holder=holder,
                      display_metadata={"gateway_input_owner": "failed-owner", "codex_auth_pending": False, "codex_auth_failure": True})
    if effects:
        db.append_message(sid, "assistant", turn_lease_holder=holder, tool_calls=[{
            "id": "fixture-call", "type": "function", "function": {"name": "fixture_effect", "arguments": "{}"}}])
        db.append_message(sid, "tool", "recorded fixture effect", tool_call_id="fixture-call", turn_lease_holder=holder)
    db.append_message(sid, "assistant", "auth failure boundary", turn_lease_holder=holder,
                      display_metadata={"codex_auth_failure": True})


@pytest.mark.parametrize("timing", ["immediate", "waiting"])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("effects", [False, True])
def test_restart_after_closed_auth_failure_is_refused_under_the_lease(tmp_path, monkeypatch, timing, rotated, effects):
    db = SessionDB(tmp_path / "state.db")
    peer = SessionDB(db.db_path)
    db.create_session("root", source="test")
    db.append_message("root", "user", "completed fixture request")
    db.append_message("root", "assistant", "completed fixture answer")
    seed = db.get_messages_as_conversation("root", repair_alternation=True)
    first = "fixture-first-owner"
    assert peer.try_acquire_session_turn_lease("root", first, ttl_seconds=30)
    target = "tip" if rotated else "root"
    committed = []
    def finish():
        if rotated:
            rotate(peer, "root", target)
        _closed_failure(peer, target, first, effects=effects)
        committed.append(peer.get_messages(target, include_inactive=True))
        peer.release_session_turn_lease("root", first)
    acquire = db.acquire_session_turn_lease
    def acquire_after_wait(*args, on_wait=None, **kwargs):
        def waiting(elapsed):
            if not committed:
                finish()
            if on_wait:
                on_wait(elapsed)
        return acquire(*args, on_wait=waiting, **kwargs)
    monkeypatch.setattr(db, "acquire_session_turn_lease", acquire_after_wait)
    loop = Mock(return_value={"completed": True, "failed": False, "messages": seed, "final_response": "mock result"})
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    agent = _agent_with_db(db, session_id="root", platform="telegram")
    try:
        if timing == "immediate":
            finish()
        result = AIAgent.run_conversation(agent, "fixture restart continuation", conversation_history=seed,
                                         persist_user_display_metadata={"restart_resume": True})
        assert result.get("auth_resume_blocked") is True
        assert result["failed"] and not result["completed"]
        loop.assert_not_called()
        assert agent._active_session_turn_lease_holder is None
        assert peer.get_messages(target, include_inactive=True) == committed[0]
        assert peer.try_acquire_session_turn_lease(target, "recovery-owner", ttl_seconds=30)
        peer.release_session_turn_lease(target, "recovery-owner")
    finally:
        peer.release_session_turn_lease("root", first)
        peer.close(); db.close()


@pytest.mark.parametrize("state,automatic,blocked", [
    ("auth_failed", False, False), ("new_human_completed", True, False),
    ("fresh", True, True), ("fresh", False, False),
])
def test_restart_gate_preserves_new_human_and_fresh_turn_contracts(tmp_path, monkeypatch, state, automatic, blocked):
    db = SessionDB(tmp_path / "state.db")
    if state != "fresh":
        db.create_session("root", source="test")
        _closed_failure(db, "root")
        if state == "new_human_completed":
            db.append_message("root", "user", "new human request")
            db.append_message("root", "assistant", "new completed answer")
    seed = db.get_messages_as_conversation("root", repair_alternation=True)
    loop = Mock(return_value={"completed": True, "failed": False, "messages": seed, "final_response": "mock result"})
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    agent = _agent_with_db(db, session_id="root", platform="telegram")
    try:
        before = db.get_messages("root", include_inactive=True)
        result = AIAgent.run_conversation(agent, "new fixture message", conversation_history=seed,
                                         persist_user_display_metadata={"restart_resume": True} if automatic else None)
        assert bool(result.get("auth_resume_blocked")) is blocked
        assert loop.call_count == int(not blocked)
        assert db.get_messages("root", include_inactive=True) == before
        assert agent._active_session_turn_lease_holder is None
    finally:
        db.close()


@pytest.mark.parametrize("mode", ["in_place", "rotating"])
@pytest.mark.parametrize("new_human", [False, True])
def test_summary_only_compaction_cannot_revive_a_failed_request(tmp_path, monkeypatch, mode, new_human):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    _closed_failure(db, "root", effects=True)
    if new_human:
        db.append_message("root", "user", "new request after failure")
        db.append_message("root", "assistant", "new completed answer")
    summary = [{"role": "user", "content": "synthetic compacted summary", "_compressed_summary": True}]
    if mode == "in_place":
        db.archive_and_compact("root", summary)
    else:
        db.publish_compression_child(parent_session_id="root", child_session_id="tip", source="test",
                                     messages=summary, require_compression_lease=False)
    loop = Mock(return_value={"completed": True, "failed": False, "messages": [], "final_response": "mock result"})
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    agent = _agent_with_db(db, session_id="root", platform="telegram")
    try:
        result = AIAgent.run_conversation(agent, "fixture continuation", conversation_history=[],
                                         persist_user_display_metadata={"restart_resume": True})
        assert bool(result.get("auth_resume_blocked")) is not new_human
        assert loop.call_count == int(new_human)
        assert agent._active_session_turn_lease_holder is None
    finally:
        db.close()
