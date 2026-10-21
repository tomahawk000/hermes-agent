"""Offline failed-turn persistence and transactional retry admission contracts."""
import asyncio
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from agent.session_persistence import _db_flush_row
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.session_transcript import TranscriptReadError


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._session_db = object()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._handle_message = AsyncMock(return_value="mock retry result")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="offline", user_id="u")
    entry = store.get_or_create_session(source)
    db = store._db_for_session_id(entry.session_id)
    yield store, runner, source, entry, db
    db.close()


@pytest.mark.parametrize("outcome", ["success", "expired", "fallback_success"])
def test_real_agent_flush_fences_input_before_mock_api_and_gateway_resolves_boundary(setup, monkeypatch, tmp_path, outcome):
    """The fence is durable before execution and ordinary success releases it."""
    import httpx
    import openai
    import run_agent
    from tests.agent.test_run_agent_codex_responses import _build_agent, _codex_message_response
    from unittest.mock import MagicMock

    store, runner, source, entry, db = setup
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    agent = _build_agent(monkeypatch)
    agent.provider = agent.requested_provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent.session_id = entry.session_id
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_session = run_agent.AIAgent._persist_session.__get__(agent)
    calls = []
    if outcome == "fallback_success":
        fallback = MagicMock()
        fallback.base_url, fallback.api_key = "https://openrouter.ai/api/v1", "synthetic-fallback"
        monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client",
                            MagicMock(return_value=(fallback, "synthetic-model")))
        agent._fallback_chain = [{"provider": "openrouter", "model": "synthetic-model"}]

    def request(_kwargs):
        calls.append(True)
        rows = db.get_messages(entry.session_id)
        assert len(rows) == 1 and rows[0]["role"] == "user"
        assert rows[0]["display_metadata"]["codex_auth_pending"] is True
        assert rows[0]["display_metadata"]["gateway_input_owner"] == "owner"
        if outcome == "fallback_success" and agent.provider == "openrouter":
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="mock completed answer",
                tool_calls=None, reasoning_content=None), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8), model="synthetic-model")
        if outcome == "success":
            return _codex_message_response("mock completed answer")
        response = httpx.Response(401, request=httpx.Request("POST", "https://example.invalid"))
        raise openai.AuthenticationError("Provided authentication token is expired",
                                         response=response, body={"error": {"code": "access_token_expired"}})

    monkeypatch.setattr(agent, "_interruptible_api_call", request)

    def refresh(**_kwargs):
        agent._codex_refresh_outcome = {"resolver_invoked": True, "force_requested": True, "outcome": "refresh_failed", "code": None, "relogin_required": None}
        return False

    monkeypatch.setattr(agent, "_try_refresh_codex_client_credentials", refresh)
    result = agent.run_conversation("offline action", persist_user_display_metadata={"gateway_input_owner": "owner"},
                                    persist_user_platform_id="offline-input")
    assert len(calls) == (2 if outcome == "fallback_success" else 1)
    assert result["completed"] is (outcome != "expired")
    # The terminal input is settled before the facade releases its lease, not
    # by a later gateway presentation write.
    assert result["messages"][0]["display_metadata"]["codex_auth_pending"] is False
    assert result["request_persistence"]["settled"] is True
    assert store.load_transcript(entry.session_id)
    messages = result["messages"]
    asyncio.run(runner._hmwa_persist_turn_transcript(
        event=MessageEvent(text="offline action", source=source, message_id="offline-input"),
        source=source, session_entry=entry, session_key=entry.session_key, agent_result=result,
        agent_messages=messages, prepared=runner._PreparedTurn([], "", "offline action", None, None, None, entry.session_id, "owner"),
        response=result["final_response"], agent_failed_early=outcome == "expired",
        hidden_reasoning_incomplete=False, is_context_overflow_failure=False))
    rows = db.get_messages(entry.session_id)
    assert len([row for row in rows if row["role"] == "user"]) == 1
    assert rows[0]["display_metadata"]["codex_auth_pending"] is False
    assert store.load_transcript(entry.session_id)
    retry_reply = asyncio.run(runner._handle_retry_command(MessageEvent(text="/retry", source=source)))
    if outcome != "expired":
        assert retry_reply == "mock retry result"
        runner._handle_message.assert_awaited_once()
    else:
        tail = next(row for row in reversed(rows) if row["role"] in {"user", "assistant", "tool"})
        assert tail["display_metadata"]["codex_auth_failure"] is True
        runner._handle_message.assert_not_awaited()


@pytest.mark.parametrize("layout", ["standalone", "merged", "repeated", "force_leading"])
@pytest.mark.parametrize("outcome", ["success", "expired"])
def test_compaction_preserves_owned_input_fence_and_resolves_all_active_copies(setup, layout, outcome):
    from agent.context_compressor import ContextCompressor, COMPRESSED_SUMMARY_METADATA_KEY

    store, runner, source, entry, db = setup
    user = {"role": "user", "content": "offline action",
            "display_metadata": {"gateway_input_owner": "owner", "codex_auth_pending": True}}
    db.append_messages_batch(entry.session_id, [user])
    compressor = ContextCompressor(model="test", quiet_mode=True, config_context_length=8000)
    summary = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold context\n{_SUMMARY_END_MARKER}"
    carrier = {"role": "user" if layout in {"merged", "repeated"} else "assistant", "content": summary,
               COMPRESSED_SUMMARY_METADATA_KEY: True}
    if layout == "force_leading":
        carrier = {**user, "display_metadata": dict(user["display_metadata"])}
        compressor._merge_summary_into_tail_row(carrier, summary, "user", True)
        compacted = [carrier]
    else:
        compacted = compressor._reappend_inflight_user_task([carrier], user)
        if layout == "repeated":
            compacted = compressor._reappend_inflight_user_task(
                [{"role": "assistant", "content": summary, COMPRESSED_SUMMARY_METADATA_KEY: True}], compacted[0])
    # Keep the original protected head too: completion must resolve both copies
    # owned by this input, without touching another owner's pending request.
    compacted = [user, *compacted]
    db.archive_and_compact(entry.session_id, compacted)
    pending = [r for r in db.get_messages(entry.session_id) if r["role"] == "user"]
    assert len(pending) == 2
    assert all(r["display_metadata"]["codex_auth_pending"] is True for r in pending)
    with pytest.raises(TranscriptReadError):
        SessionStore(store.sessions_dir, GatewayConfig()).load_transcript(entry.session_id)
    db.append_message(entry.session_id, "assistant", "mock terminal answer")
    if outcome == "expired":
        assert store.close_failed_auth_turn(entry.session_id, None, "offline failure", input_owner="owner")
    else:
        assert store.complete_codex_auth_turn(entry.session_id, "owner")
    rows = db.get_messages(entry.session_id)
    assert all(r["display_metadata"]["codex_auth_pending"] is False for r in rows if r["role"] == "user")
    assert store.load_transcript(entry.session_id)
    asyncio.run(runner._handle_retry_command(MessageEvent(text="/retry", source=source)))
    if outcome == "success":
        runner._handle_message.assert_awaited_once()
    else:
        runner._handle_message.assert_not_awaited()


@pytest.mark.parametrize("fault", ["read_failure", "no_platform_id", "write_failure", "active_lease", "legacy_no_id"])
@pytest.mark.parametrize("effects", [False, True])
def test_uncertain_failure_keeps_boundary_or_blocks_history_and_retry(setup, monkeypatch, fault, effects):
    store, runner, source, entry, db = setup
    sid = entry.session_id
    platform_id = "" if fault in {"no_platform_id", "legacy_no_id"} else "offline-input"
    user = {"role": "user", "content": "offline action"}
    if fault != "legacy_no_id":
        user["display_metadata"] = {"gateway_input_owner": "owner"}
    row = _db_flush_row(SimpleNamespace(provider="openai-codex"), user, True)
    row["platform_message_id"] = platform_id or None
    db.append_messages_batch(sid, [row])
    messages = [user]
    if effects:
        calls = [{"id": "mock-call", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]
        db.append_message(sid, "assistant", tool_calls=calls)
        db.append_message(sid, "tool", "mock effect", tool_call_id="mock-call")
        messages += [{"role": "assistant", "tool_calls": calls},
                     {"role": "tool", "content": "mock effect", "tool_call_id": "mock-call"}]
    if fault == "read_failure":
        read = db._read_one
        def fail_evidence(sql, *args, **kwargs):
            if "WITH target AS" in sql:
                raise RuntimeError("synthetic evidence failure")
            return read(sql, *args, **kwargs)
        monkeypatch.setattr(db, "_read_one", fail_evidence)
    if fault == "write_failure":
        monkeypatch.setattr(store, "close_failed_auth_turn", lambda *a, **k: False)
    holder = f"pid={os.getpid()}:turn=offline"
    if fault == "active_lease":
        assert db.try_acquire_session_turn_lease(sid, holder, ttl_seconds=60)
    failed = {"failed": True, "completed": False, "agent_persisted": True,
              "auth_failure": {"provider": "openai-codex"}, "messages": messages,
              "history_offset": 0, "current_turn_user_idx": 0, "last_prompt_tokens": 0}
    async def check():
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text=user["content"], source=source, message_id=platform_id),
            source=source, session_entry=entry, session_key=entry.session_key, agent_result=failed,
            agent_messages=messages, prepared=runner._PreparedTurn([user], "", user["content"], None, None, None, sid, "owner"),
            response="offline auth failure", agent_failed_early=True, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False)
        assert failed["request_persistence"]["retry_blocked"] is True
        closed = fault not in {"write_failure", "active_lease"}
        assert failed["request_persistence"]["boundary_saved"] is closed
        if fault == "active_lease":
            db.release_session_turn_lease(sid, holder)
        before = db.get_messages(sid, include_inactive=True)
        if closed:
            assert before[-1]["role"] == "assistant"
            assert before[-1]["display_metadata"]["codex_auth_failure"] is True
        else:
            with pytest.raises(TranscriptReadError):
                store.load_transcript(sid)
        await runner._handle_retry_command(MessageEvent(text="/retry", source=source))
        runner._handle_message.assert_not_awaited()
        assert db.get_messages(sid, include_inactive=True) == before
        # A restart loses the in-memory block, but the pending input/closed marker
        # still prevents replay on a fresh store with the same real SQLite database.
        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        runner.session_store = restarted
        try:
            await runner._handle_retry_command(MessageEvent(text="/retry", source=source))
            runner._handle_message.assert_not_awaited()
            assert db.get_messages(sid, include_inactive=True) == before
        finally:
            restarted._db_for_session_id(sid).close()
    asyncio.run(check())


@pytest.mark.parametrize("carrier", [False, True])
@pytest.mark.parametrize("race", ["marker", "newer_identical_input"])
def test_retry_checks_live_history_after_other_writer_commits(setup, monkeypatch, carrier, race):
    store, runner, source, entry, db = setup
    sid = entry.session_id
    content = "offline action"
    if carrier:
        content = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold context\n{_SUMMARY_END_MARKER}\n\noffline action"
    row_id = db.append_message(sid, "user", content, platform_message_id="offline-input")
    db.append_message(sid, "assistant", "mock partial result")
    snapshot_read = threading.Event()
    committed = threading.Event()
    load = store.load_transcript
    def racing_load(session_id):
        snapshot = load(session_id)
        snapshot_read.set()
        assert committed.wait(10)
        return snapshot
    monkeypatch.setattr(store, "load_transcript", racing_load)
    def other_writer():
        assert snapshot_read.wait(10)
        peer = SessionStore(store.sessions_dir, GatewayConfig())
        peer_db = peer._db_for_session_id(sid)
        try:
            if race == "marker":
                assert peer.close_failed_auth_turn(sid, row_id, "offline failure")
            else:
                peer_db.append_message(sid, "user", content, platform_message_id="newer-input")
                peer_db.append_message(sid, "assistant", "newer result")
            committed.set()
        finally:
            peer_db.close()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(other_writer)
        reply = asyncio.run(runner._handle_retry_command(MessageEvent(text="/retry", source=source)))
        future.result(timeout=10)
    runner._handle_message.assert_not_awaited()
    assert reply.startswith(("Retry failed", "Cannot retry"))
    rows = db.get_messages(sid, include_inactive=True)
    assert rows[0]["id"] == row_id and rows[0]["content"] == content and rows[0]["active"] == 1
    if race == "marker":
        tail = next(row for row in reversed(rows) if row["role"] in {"user", "assistant", "tool"})
        assert tail["display_metadata"]["codex_auth_failure"] is True
    else:
        assert [m["content"] for m in rows if m["role"] == "user"] == [content, content]


@pytest.mark.parametrize("outcome", ["success", "failure", "completion_write_failure", "unfinished", "newer_input", "malformed_metadata"])
def test_pending_input_requires_durable_matching_completion(setup, monkeypatch, outcome):
    store, runner, source, entry, db = setup
    sid = entry.session_id
    user = {"role": "user", "content": "offline action", "display_metadata": {"gateway_input_owner": "owner", "other": True}}
    row = _db_flush_row(SimpleNamespace(provider="openai-codex"), user, True)
    db.append_messages_batch(sid, [row])
    db.append_message(sid, "assistant", "mock result")
    execute_write = db._execute_write
    if outcome == "completion_write_failure":
        monkeypatch.setattr(db, "_execute_write", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic write failure")))
    elif outcome == "newer_input":
        db.append_message(sid, "user", "newer action", display_metadata={"gateway_input_owner": "newer", "codex_auth_pending": True})
        db.append_message(sid, "assistant", "newer result")
    elif outcome == "failure":
        assert store.close_failed_auth_turn(sid, row["_row_id"], "offline failure")
    elif outcome == "malformed_metadata":
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET display_metadata=? WHERE id=?", ("{", row["_row_id"])))
    if outcome in {"success", "completion_write_failure", "newer_input", "failure"}:
        assert store.complete_codex_auth_turn(sid, "owner") is (outcome == "success")
    if outcome == "completion_write_failure":
        monkeypatch.setattr(db, "_execute_write", execute_write)
    before = db.get_messages(sid, include_inactive=True)
    reply = asyncio.run(runner._handle_retry_command(MessageEvent(text="/retry", source=source)))
    if outcome == "success":
        assert reply == "mock retry result"
        runner._handle_message.assert_awaited_once()
        assert before[0]["display_metadata"] == {"gateway_input_owner": "owner", "other": True, "codex_auth_pending": False,
                                                   "gateway_input_outcome": "completed"}
    else:
        runner._handle_message.assert_not_awaited()
        assert db.get_messages(sid, include_inactive=True) == before


@pytest.mark.parametrize("reader", ["restart", "other_process"])
def test_unclosed_pending_input_blocks_ordinary_history_on_any_reader(setup, reader):
    store, _runner, _source, entry, db = setup
    sid = entry.session_id
    user = {"role": "user", "content": "mock effect request", "display_metadata": {"gateway_input_owner": "owner"}}
    db.append_messages_batch(sid, [_db_flush_row(SimpleNamespace(provider="openai-codex"), user, True)])
    db.append_message(sid, "assistant", tool_calls=[{"id": "mock", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
    db.append_message(sid, "tool", "mock effect already occurred", tool_call_id="mock")
    # A later user row must not let alternation repair hide the earlier pending input.
    db.append_message(sid, "user", "unrelated next request")
    before = db.get_messages(sid, include_inactive=True)
    if reader == "restart":
        fresh = SessionStore(store.sessions_dir, GatewayConfig())
        try:
            assert sid not in fresh._uncertain_auth_failures
            with pytest.raises(TranscriptReadError):
                fresh.load_transcript(sid)
        finally:
            fresh._db_for_session_id(sid).close()
    else:
        script = """from pathlib import Path
import sys
import hermes_state
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_transcript import TranscriptReadError
hermes_state.DEFAULT_DB_PATH = Path(sys.argv[1])
store = SessionStore(Path(sys.argv[2]), GatewayConfig())
try:
    store.load_transcript(sys.argv[3])
except TranscriptReadError:
    print('blocked')
else:
    raise AssertionError('pending input was exposed to ordinary history replay')
finally:
    store._db_for_session_id(sys.argv[3]).close()
"""
        result = subprocess.run([sys.executable, "-c", script, str(db.db_path), str(store.sessions_dir), sid],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "blocked"
    assert db.get_messages(sid, include_inactive=True) == before


@pytest.mark.parametrize("conflict", ["owner", "empty_owner", "false_owner", "platform", "known_identity_without_match", "row", "timestamp"])
def test_identical_legacy_content_does_not_override_input_identity(setup, conflict):
    store, _runner, _source, entry, db = setup
    sid = entry.session_id
    meta = {"gateway_input_owner": "other-owner"} if conflict in {"owner", "known_identity_without_match"} else None
    if conflict in {"empty_owner", "false_owner"}:
        meta = {"gateway_input_owner": "" if conflict == "empty_owner" else False}
    row_id = db.append_message(sid, "user", "identical text", display_metadata=meta,
                               platform_message_id="other-input" if conflict == "platform" else None)
    expected = {"role": "user", "content": "identical text"}
    if conflict == "row":
        expected["_row_id"] = row_id - 1
    elif conflict == "timestamp":
        expected["timestamp"] = 1.0
    before = db.get_messages(sid, include_inactive=True)
    store.note_uncertain_auth_failure(sid)
    assert not store.close_failed_auth_turn(
        sid, None, "offline failure", input_owner=None if conflict == "known_identity_without_match" else "owner",
        platform_message_id="original-input" if conflict == "platform" else "",
        expected_messages=[expected])
    assert db.get_messages(sid, include_inactive=True) == before
    with pytest.raises(TranscriptReadError):
        store.load_transcript(sid)
