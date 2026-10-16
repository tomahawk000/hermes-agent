"""Offline admission fence checks with independent SQLite connections."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_facade_lease import DurableTurnLease, admit_durable_turn_lease
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_transcript import TranscriptReadError
from hermes_state import SessionDB


@pytest.mark.parametrize("effects", [False, True])
@pytest.mark.parametrize("timing", ["waiting", "immediate"])
def test_admission_rejects_pending_input_committed_after_initial_read(tmp_path, monkeypatch, effects, timing):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="offline", user_id="u")
    entry = store.get_or_create_session(source)
    sid = entry.session_id
    db = store._db_for_session_id(sid)
    stale_history = store.load_transcript(sid)
    assert stale_history == []
    peer = SessionDB(db.db_path)
    holder = "offline-first-owner"
    assert peer.try_acquire_session_turn_lease(sid, holder, ttl_seconds=30)
    committed = []

    def commit_failed_input():
        peer.append_message(sid, "user", "offline action", turn_lease_holder=holder,
                            display_metadata={"gateway_input_owner": "failed-owner", "codex_auth_pending": True})
        if effects:
            peer.append_message(sid, "assistant", turn_lease_holder=holder, tool_calls=[{
                "id": "mock-effect", "type": "function", "function": {"name": "offline_effect", "arguments": "{}"}}])
            peer.append_message(sid, "tool", "mock effect already completed", turn_lease_holder=holder,
                                tool_call_id="mock-effect")
        committed.append(peer.get_messages(sid, include_inactive=True))
        with pytest.raises(TranscriptReadError):
            store.load_transcript(sid)
        peer.release_session_turn_lease(sid, holder)

    def status(text):
        if timing == "waiting" and not committed and "waiting for it" in text:
            commit_failed_input()

    build_threads = Mock()
    monkeypatch.setattr(DurableTurnLease, "build_threads", build_threads)
    agent = SimpleNamespace(_session_db=db, session_id=sid, _emit_status=status)
    admission = None
    try:
        if timing == "immediate":
            # The same gap exists when the first owner releases before acquisition
            # starts: no waiting callback or history replacement will occur.
            commit_failed_input()
        with pytest.raises(ValueError, match="no durable completion or failure boundary"):
            admission = admit_durable_turn_lease(
                agent, session_id=sid, relay_turn_id="later-turn",
                task_context={"platform": "telegram", "session_id": sid}, conversation_history=stale_history)
        build_threads.assert_not_called()
        assert len(committed) == 1
        assert peer.get_messages(sid, include_inactive=True) == committed[0]
        assert agent._active_session_turn_lease_holder is None
        # Rejected admission releases its acquired lease, without erasing evidence.
        assert peer.try_acquire_session_turn_lease(sid, "offline-recovery-owner", ttl_seconds=30)
        peer.release_session_turn_lease(sid, "offline-recovery-owner")
    finally:
        if admission and admission.lease:
            admission.lease.release()
        peer.release_session_turn_lease(sid, holder)
        peer.close()
        db.close()


@pytest.mark.parametrize("waited", [False, True])
def test_resolved_input_allows_admission_and_immediate_history_identity_is_preserved(tmp_path, monkeypatch, waited):
    db = SessionDB(tmp_path / "state.db")
    sid = "offline-session"
    db.create_session(sid, source="test")
    db.append_message(sid, "user", "offline old request",
                      display_metadata={"gateway_input_owner": "old-owner", "codex_auth_pending": False})
    db.append_message(sid, "assistant", "mock completed answer")
    peer = SessionDB(db.db_path)
    holder = "offline-first-owner"
    if waited:
        assert peer.try_acquire_session_turn_lease(sid, holder, ttl_seconds=30)
    seed = [{"role": "user", "content": "caller-owned prepared context"}]
    def status(text):
        if "waiting for it" in text:
            peer.release_session_turn_lease(sid, holder)
    monkeypatch.setattr(DurableTurnLease, "build_threads", lambda self: None)
    agent = SimpleNamespace(_session_db=db, session_id=sid, _emit_status=status)
    admission = None
    before = db.get_messages(sid, include_inactive=True)
    try:
        admission = admit_durable_turn_lease(
            agent, session_id=sid, relay_turn_id="later-turn", task_context={"platform": "telegram", "session_id": sid},
            conversation_history=seed)
        assert admission.lease is not None
        if waited:
            assert [m["content"] for m in admission.conversation_history] == ["offline old request", "mock completed answer"]
        else:
            assert admission.conversation_history is seed
        assert db.get_messages(sid, include_inactive=True) == before
    finally:
        if admission and admission.lease:
            admission.lease.release()
        peer.close()
        db.close()


@pytest.mark.parametrize("failure", ["read_failure", "malformed_metadata"])
def test_immediate_auth_state_read_failure_releases_lease_without_admission(tmp_path, monkeypatch, failure):
    db = SessionDB(tmp_path / "state.db")
    sid = "offline-session"
    db.create_session(sid, source="test")
    db.append_message(sid, "user", "offline request", display_metadata={"codex_auth_pending": True})
    peer = SessionDB(db.db_path)
    if failure == "read_failure":
        monkeypatch.setattr(db, "read_turn_admission", Mock(side_effect=RuntimeError("synthetic metadata read failure")))
        expected = RuntimeError
    else:
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET display_metadata=? WHERE session_id=?", ("{", sid)))
        expected = ValueError
    build_threads = Mock()
    monkeypatch.setattr(DurableTurnLease, "build_threads", build_threads)
    agent = SimpleNamespace(_session_db=db, session_id=sid, _emit_status=lambda text: None)
    before = peer.get_messages(sid, include_inactive=True)
    try:
        with pytest.raises(expected):
            admit_durable_turn_lease(agent, session_id=sid, relay_turn_id="later-turn",
                                     task_context={"platform": "telegram", "session_id": sid}, conversation_history=[])
        build_threads.assert_not_called()
        assert agent._active_session_turn_lease_holder is None
        assert peer.get_messages(sid, include_inactive=True) == before
        assert peer.try_acquire_session_turn_lease(sid, "offline-recovery-owner", ttl_seconds=30)
        peer.release_session_turn_lease(sid, "offline-recovery-owner")
    finally:
        peer.close()
        db.close()
