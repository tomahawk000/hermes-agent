"""Canonical continuation, ownership and history contracts for offline admission."""
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_facade_lease import DurableTurnLease, admit_durable_turn_lease
from hermes_state import SessionDB


@pytest.fixture
def state(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    peer = SessionDB(db.db_path)
    db.create_session("root", source="test")
    db.append_message("root", "user", "previous request")
    db.append_message("root", "assistant", "previous completed answer")
    monkeypatch.setattr(DurableTurnLease, "build_threads", Mock())
    yield db, peer
    peer.close()
    db.close()


def rotate(peer, parent, child):
    holder = "offline-compression-owner"
    assert peer.try_acquire_compression_lock(parent, holder, ttl_seconds=30)
    try:
        peer.publish_compression_child(parent_session_id=parent, child_session_id=child, source="test",
            messages=[{"role": "user", "content": "summary for " + child},
                      {"role": "assistant", "content": "summary acknowledged"}], compression_lock_holder=holder)
    finally:
        peer.release_compression_lock(parent, holder)


def pending(peer, sid, holder=None, *, effects=False):
    peer.append_message(sid, "user", "unresolved offline action", turn_lease_holder=holder,
                        display_metadata={"gateway_input_owner": "failed-owner", "codex_auth_pending": True})
    if effects:
        peer.append_message(sid, "assistant", turn_lease_holder=holder, tool_calls=[{
            "id": "mock", "type": "function", "function": {"name": "offline_effect", "arguments": "{}"}}])
        peer.append_message(sid, "tool", "mock effect completed", tool_call_id="mock", turn_lease_holder=holder)


def admit(db, peer, start, seed, timing, commit):
    first = "offline-first-owner"
    assert peer.try_acquire_session_turn_lease(start, first, ttl_seconds=30)
    committed = []
    def finish_first():
        commit(first)
        committed.append(True)
        peer.release_session_turn_lease(start, first)
    def status(text):
        if timing == "waiting" and not committed and "waiting for it" in text:
            finish_first()
    agent = SimpleNamespace(_session_db=db, session_id=start, _emit_status=status)
    context = {"platform": "telegram", "session_id": start}
    if timing == "immediate":
        finish_first()
    try:
        admission = admit_durable_turn_lease(agent, session_id=start, relay_turn_id="later-turn",
                                             task_context=context, conversation_history=seed)
    finally:
        peer.release_session_turn_lease(start, first)
    assert committed == [True]
    return agent, context, admission


@pytest.mark.parametrize("start", ["root", "middle", "tip"])
@pytest.mark.parametrize("timing", ["immediate", "waiting"])
@pytest.mark.parametrize("effects", [False, True])
def test_pending_current_tip_rejected_from_root_intermediate_or_tip(state, start, timing, effects):
    db, peer = state
    if start in {"middle", "tip"}:
        rotate(peer, "root", "middle")
    if start == "tip":
        rotate(peer, "middle", "tip")
    seed = db.get_messages_as_conversation(start, repair_alternation=True)
    evidence = []
    def commit(holder):
        if start == "root":
            rotate(peer, "root", "middle")
        if start != "tip":
            rotate(peer, "middle", "tip")
        pending(peer, "tip", holder, effects=effects)
        evidence.append(peer.get_messages("tip", include_inactive=True))
    with pytest.raises(ValueError, match="no durable completion or failure boundary"):
        admit(db, peer, start, seed, timing, commit)
    DurableTurnLease.build_threads.assert_not_called()
    assert peer.get_messages("tip", include_inactive=True) == evidence[0]
    assert peer.try_acquire_session_turn_lease("tip", "recovery-owner", ttl_seconds=30)
    peer.release_session_turn_lease("tip", "recovery-owner")


@pytest.mark.parametrize("start", ["root", "middle"])
@pytest.mark.parametrize("timing", ["immediate", "waiting"])
def test_changed_resolved_tip_reloads_history_and_retains_lineage_ownership(state, start, timing):
    db, peer = state
    if start == "middle":
        rotate(peer, "root", "middle")
    seed = db.get_messages_as_conversation(start, repair_alternation=True)
    def commit(holder):
        if start == "root":
            rotate(peer, "root", "middle")
        rotate(peer, "middle", "tip")
    agent, context, admission = admit(db, peer, start, seed, timing, commit)
    try:
        assert agent.session_id == context["session_id"] == "tip"
        assert admission.conversation_history is not seed
        assert [m["content"] for m in admission.conversation_history] == ["summary for tip", "summary acknowledged"]
        assert admission.lease.session_id == start
        assert db.refresh_session_turn_lease("tip", admission.lease.holder, ttl_seconds=30)
        assert not peer.try_acquire_session_turn_lease("root", "foreign-owner", ttl_seconds=30)
        assert not peer.try_acquire_session_turn_lease("tip", "foreign-owner", ttl_seconds=30)
    finally:
        admission.lease.release()
    assert agent._active_session_turn_lease_holder is None
    assert peer.try_acquire_session_turn_lease("root", "recovery-owner", ttl_seconds=30)
    peer.release_session_turn_lease("root", "recovery-owner")


@pytest.mark.parametrize("kind", ["branch", "delegate", "reset", "tool"])
@pytest.mark.parametrize("timing", ["immediate", "waiting"])
def test_unrelated_fork_pending_state_does_not_hijack_compression_admission(state, kind, timing):
    db, peer = state
    seed = db.get_messages_as_conversation("root", repair_alternation=True)
    def commit(holder):
        rotate(peer, "root", "tip")
        key = {"branch": "_branched_from", "delegate": "_delegate_from", "reset": "_reset_from"}.get(kind)
        peer.create_session("private-fork", source="tool" if kind == "tool" else "test", parent_session_id="root",
                            model_config={key: "root"} if key else {})
        pending(peer, "private-fork")
    agent, context, admission = admit(db, peer, "root", seed, timing, commit)
    try:
        assert agent.session_id == context["session_id"] == "tip"
        assert all("unresolved" not in str(m.get("content")) for m in admission.conversation_history)
        assert peer.get_messages("private-fork")[0]["display_metadata"]["codex_auth_pending"] is True
    finally:
        admission.lease.release()


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from", "_reset_from"])
@pytest.mark.parametrize("timing", ["immediate", "waiting"])
def test_inherited_fork_marker_does_not_hide_pending_compression_child(state, marker, timing):
    db, peer = state
    peer.create_session("ancestor", source="test")
    peer._execute_write(lambda conn: conn.execute("UPDATE sessions SET parent_session_id=?,model_config=? WHERE id='root'",
                                                  ("ancestor", '{"' + marker + '":"ancestor"}')))
    seed = db.get_messages_as_conversation("root", repair_alternation=True)
    def commit(holder):
        rotate(peer, "root", "tip")
        pending(peer, "tip", holder)
    with pytest.raises(ValueError, match="no durable completion or failure boundary"):
        admit(db, peer, "root", seed, timing, commit)
    assert peer.try_acquire_session_turn_lease("tip", "recovery-owner", ttl_seconds=30)
    peer.release_session_turn_lease("tip", "recovery-owner")


@pytest.mark.parametrize("failure", ["read_failure", "missing_child", "cycle", "different_conversation", "lost_owner", "malformed_route"])
def test_unknown_or_unowned_continuation_fails_closed_and_preserves_owner(state, monkeypatch, failure):
    db, peer = state
    seed = db.get_messages_as_conversation("root", repair_alternation=True)
    reader = db.read_turn_admission
    if failure == "missing_child":
        peer.end_session("root", "compression")
    elif failure == "cycle":
        peer.end_session("root", "compression")
        peer.create_session("middle", source="test", parent_session_id="root")
        peer.end_session("middle", "compression")
        peer._execute_write(lambda conn: conn.execute("UPDATE sessions SET parent_session_id='middle' WHERE id='root'"))
    elif failure in {"different_conversation", "malformed_route"}:
        peer.create_session("child", source="test", parent_session_id="root")
        peer.append_message("child", "user", "offline child input")
        if failure == "malformed_route":
            peer._execute_write(lambda conn: conn.execute("UPDATE sessions SET model_config='{' WHERE id='child'"))
    elif failure == "read_failure":
        def failed_read(*args, **kwargs):
            db._conn.set_authorizer(lambda action, table, *rest: sqlite3.SQLITE_DENY
                                    if action == sqlite3.SQLITE_READ and table == "sessions" else sqlite3.SQLITE_OK)
            try:
                return reader(*args, **kwargs)
            finally:
                db._conn.set_authorizer(None)
        monkeypatch.setattr(db, "read_turn_admission", failed_read)
    else:
        def lost_owner(*args, **kwargs):
            peer._execute_write(lambda conn: conn.execute("UPDATE session_turn_leases SET expires_at=0"))
            assert peer.try_acquire_session_turn_lease("root", "foreign-owner", ttl_seconds=30)
            return reader(*args, **kwargs)
        monkeypatch.setattr(db, "read_turn_admission", lost_owner)
    before = peer.get_messages("root", include_inactive=True)
    agent = SimpleNamespace(_session_db=db, session_id="root", _emit_status=lambda text: None)
    context = {"platform": "telegram", "session_id": "root"}
    try:
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            admit_durable_turn_lease(agent, session_id="root", relay_turn_id="later-turn", task_context=context,
                                     conversation_history=seed)
        DurableTurnLease.build_threads.assert_not_called()
        assert agent.session_id == context["session_id"] == "root"
        assert agent._active_session_turn_lease_holder is None
        assert peer.get_messages("root", include_inactive=True) == before
        if failure == "lost_owner":
            assert peer.refresh_session_turn_lease("root", "foreign-owner", ttl_seconds=30)
            assert not peer.try_acquire_session_turn_lease("root", "recovery-owner", ttl_seconds=30)
        else:
            assert peer.try_acquire_session_turn_lease("root", "recovery-owner", ttl_seconds=30)
            peer.release_session_turn_lease("root", "recovery-owner")
    finally:
        peer.release_session_turn_lease("root", "foreign-owner")


@pytest.mark.parametrize("deleted", [False, True])
@pytest.mark.parametrize("timing", ["immediate", "waiting"])
def test_native_fresh_row_admission_keeps_seed_and_marks_first_flush_needed(state, deleted, timing):
    db, peer = state
    if deleted:
        peer.create_session("fresh", source="test")
        peer.delete_session("fresh")
    seed = [{"role": "user", "content": "new caller-owned seed"}]
    agent, context, admission = admit(db, peer, "fresh", seed, timing, lambda holder: None)
    try:
        assert admission.conversation_history is seed
        assert agent.session_id == context["session_id"] == "fresh"
        assert agent._session_db_created is False
        assert db.get_session("fresh") is None
        assert not peer.try_acquire_session_turn_lease("fresh", "foreign-owner", ttl_seconds=30)
        db.create_session("fresh", source="test")
        db.append_message("fresh", "user", "new caller-owned seed", turn_lease_holder=admission.lease.holder)
        assert peer.get_messages("fresh")[0]["content"] == "new caller-owned seed"
    finally:
        admission.lease.release()
    assert peer.try_acquire_session_turn_lease("fresh", "recovery-owner", ttl_seconds=30)
    peer.release_session_turn_lease("fresh", "recovery-owner")
