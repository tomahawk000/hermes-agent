"""Independent offline regressions reopening PR1; no provider or transport execution."""
import asyncio
import os
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from agent.session_persistence import _db_flush_row
from agent.prompt_builder import steer_user_row
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_cross_process_turn_lease import _agent_with_db


def setup_store(tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="offline-review", user_id="fixture")
    entry = store.get_or_create_session(source)
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._session_db = store._db_for_session_id(entry.session_id)
    runner._refresh_agent_cache_message_count = AsyncMock()
    return store, source, entry, runner


def pending_input(db, sid, holder=None):
    user = {"role": "user", "content": "fixture request", "message_id": "fixture-input",
            "display_metadata": {"gateway_input_owner": "fixture-owner"}}
    row = _db_flush_row(NS(provider="openai-codex"), user, True)
    assert row["display_metadata"]["codex_auth_pending"] is True
    row_id = db.append_message(sid, turn_lease_holder=holder, **row)
    return user, row_id


def worker_result(monkeypatch, source, sid, raw, *, agent=None):
    """Run real run_sync and real approval wrapper; replace only setup/provider execution."""
    if agent is None:
        agent = NS(model="fixture-model", run_conversation=lambda *a, **k: raw)
    ctx = NS(source=source, session_key="offline-key", session_id=sid, user_config={},
             message="fixture request", agent_holder=[agent], tools_holder=[[]],
             title_user_message=None, persist_user_display_kind=None,
             persist_user_display_metadata={"gateway_input_owner": "fixture-owner"},
             moa_config=None, inbound_message_id="fixture-input", mute_notification_reply=False)
    runner = NS(_provider_routing={},
                _resolve_session_agent_runtime=lambda **kw: ("fixture-model", {"provider": "openai-codex"}),
                _resolve_session_reasoning_config=lambda **kw: None,
                _resolve_session_service_tier=lambda **kw: None,
                _resolve_turn_agent_config=lambda *a: {})
    worker = TurnRunner(runner, ctx)
    for name, replacement in {
        "_combined_ephemeral_prompt": lambda: "",
        "_setup_stream_consumer": lambda *a: (None, None, None, False),
        "_resolve_turn_agent": lambda *a: (agent, False),
        "_wire_turn_agent_callbacks": lambda *a: None,
        "_load_turn_history": lambda *a: ([], None, set()),
        "_prepare_turn_message": lambda *a: (None, None),
        "_native_image_run_message": lambda: "fixture request",
        "_finish_stream_consumer": lambda *a: None,
        "_sync_session_after_run": lambda *a: (False, sid, 0),
        "_append_auto_media_tags": lambda response, *a: response,
    }.items():
        monkeypatch.setattr(worker, name, replacement)
    return worker.run_sync()


def persist(runner, source, entry, raw, *, failed):
    prepared = runner._PreparedTurn([{"role": "system", "content": "fixture"}], "", "fixture request",
                                    None, None, None, entry.session_id, "fixture-owner")
    return runner._hmwa_persist_turn_transcript(
        event=MessageEvent(text="fixture request", source=source, message_id="fixture-input"),
        source=source, session_entry=entry, session_key=entry.session_key,
        agent_result=raw, agent_messages=raw["messages"], prepared=prepared,
        response=raw.get("final_response", ""), agent_failed_early=failed,
        hidden_reasoning_incomplete=False, is_context_overflow_failure=False)


def metadata(db, sid, row_id):
    return next(r for r in db.get_messages(sid) if r["id"] == row_id)["display_metadata"]


def rejected_load(store, sid):
    try:
        store.load_transcript(sid)
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)
    return None


@pytest.mark.parametrize("terminal", ["auth_failure", "auth_resume_blocked"])
@pytest.mark.parametrize("body", ["", "fixture terminal notice"])
def test_real_worker_preserves_auth_closure_contract(monkeypatch, terminal, body):
    raw = {"messages": [], "final_response": body, "failed": True, "completed": False,
           "error": "fixture failure", "current_turn_user_idx": 0,
           terminal: {"provider": "openai-codex"} if terminal == "auth_failure" else True}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="offline-review", user_id="fixture")
    actual = worker_result(monkeypatch, source, "fixture-session", raw)
    print({"raw_keys": sorted(raw), "mapped_keys": sorted(actual), "terminal": terminal, "body": body})
    assert actual.get(terminal) == raw[terminal]
    assert actual.get("current_turn_user_idx") == 0


def test_real_worker_auth_failure_reaches_durable_gateway_closure(tmp_path, monkeypatch):
    store, source, entry, runner = setup_store(tmp_path, monkeypatch)
    db, sid = runner._session_db, entry.session_id
    try:
        user, row_id = pending_input(db, sid)
        raw = {"messages": [user], "final_response": "fixture expired Codex token", "failed": True,
               "completed": False, "auth_failure": {"provider": "openai-codex"}, "current_turn_user_idx": 0}
        actual = worker_result(monkeypatch, source, sid, raw)
        asyncio.run(persist(runner, source, entry, actual, failed=True))
        observed = metadata(db, sid, row_id)
        print({"mapped_auth_failure": actual.get("auth_failure"), "user_metadata": observed,
               "blocked_load": rejected_load(store, sid), "notice": runner._hmwa_failed_turn_notice(actual)})
        assert observed.get("codex_auth_pending") is False
        assert observed.get("codex_auth_failure") is True
    finally:
        db.close()


@pytest.mark.parametrize("status", [429, 503])
def test_exhausted_non_auth_failure_settles_owned_input_across_restart(tmp_path, monkeypatch, status):
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import max_retries_exhausted_result
    from agent.conversation_loop import _close_durable_failed_turn
    from tests.agent.test_failed_turn_chat_copy import _Agent, _Http
    store, source, entry, runner = setup_store(tmp_path, monkeypatch)
    db, sid = runner._session_db, entry.session_id
    try:
        user, row_id = pending_input(db, sid)
        error = _Http(status, f"fixture HTTP {status}: unavailable")
        classified = classify_api_error(error, provider="openai-codex", model="fixture-model")
        raw = max_retries_exhausted_result(
            _Agent(), error, classified, max_retries=3, is_rate_limited=status == 429,
            error_msg=str(error).lower(), api_kwargs=None, api_messages=[], messages=[user],
            conversation_history=[], api_call_count=4, approx_tokens=10,
            provider="openai-codex", base_url="https://example.invalid/codex", model="fixture-model")
        raw["current_turn_user_idx"] = 0
        # Exercise the core boundary closer, with a fixture-only flush to real SQLite.
        core = NS(_session_db=db, session_id=sid, _persist_disabled=False)
        core._flush_messages_to_session_db = lambda rows: [
            db.append_message(sid, "assistant", r["content"], display_kind=r.get("display_kind"))
            for r in rows[1:]]
        _close_durable_failed_turn(core, raw)
        actual = worker_result(monkeypatch, source, sid, raw)
        asyncio.run(persist(runner, source, entry, actual, failed=True))
        observed = metadata(db, sid, row_id)
        restarted = SessionStore(tmp_path / "sessions", GatewayConfig())
        blocked_after_restart = rejected_load(restarted, sid)
        print({"status": status, "metadata": observed, "roles": [r["role"] for r in db.get_messages(sid)],
               "blocked_before_restart": rejected_load(store, sid), "blocked_after_restart": blocked_after_restart})
        assert observed.get("codex_auth_pending") is False
        assert blocked_after_restart is None
    finally:
        db.close()


@pytest.mark.parametrize("outcome", ["success", "auth_failure"])
@pytest.mark.parametrize("next_row", ["steer", "new_independent_input", "none"])
def test_owned_input_settles_through_same_turn_steer_but_not_new_request(tmp_path, monkeypatch, outcome, next_row):
    store, source, entry, runner = setup_store(tmp_path, monkeypatch)
    db, sid = runner._session_db, entry.session_id
    try:
        user, row_id = pending_input(db, sid)
        messages = [user]
        if next_row != "none":
            row = steer_user_row("fixture same-turn correction") if next_row == "steer" else {
                "role": "user", "content": "fixture new independent input",
                "platform_message_id": "fixture-new-input", "display_metadata": {"gateway_input_owner": "new-owner"}}
            db.append_message(sid, **row)
            messages.append(row)
        db.append_message(sid, "assistant", "fixture terminal result")
        if outcome == "success":
            settled = store.complete_codex_auth_turn(sid, "fixture-owner")
        else:
            settled = store.close_failed_auth_turn(sid, row_id, "fixture failed boundary", input_owner="fixture-owner",
                                                  platform_message_id="fixture-input", expected_messages=messages)
        observed = metadata(db, sid, row_id)
        print({"outcome": outcome, "next_row": next_row, "settled": settled, "metadata": observed,
               "blocked_load": rejected_load(store, sid)})
        assert settled is (next_row != "new_independent_input")
        assert observed.get("codex_auth_pending") is (next_row == "new_independent_input")
    finally:
        db.close()


@pytest.mark.parametrize("followup", ["cross_connection", "queued"])
def test_completed_input_is_settled_before_real_followup_admission(tmp_path, monkeypatch, followup):
    store, source, entry, runner = setup_store(tmp_path, monkeypatch)
    db, sid = runner._session_db, entry.session_id
    first = _agent_with_db(db, session_id=sid, platform="telegram")
    first.provider = "openai-codex"
    row_id = []
    def terminal_loop(agent, *args, **kwargs):
        user, rid = pending_input(db, sid, agent._active_session_turn_lease_holder)
        row_id.append(rid)
        db.append_message(sid, "assistant", "fixture completed answer", turn_lease_holder=agent._active_session_turn_lease_holder)
        return {"messages": [user, {"role": "assistant", "content": "fixture completed answer"}],
                "final_response": "fixture completed answer", "completed": True, "failed": False}
    loop = Mock(side_effect=terminal_loop)
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    peer = SessionDB(db.db_path)
    try:
        raw = AIAgent.run_conversation(first, "fixture request", conversation_history=[])
        assert first._active_session_turn_lease_holder is None
        assert loop.call_count == 1
        error = None
        if followup == "cross_connection":
            holder = f"pid={os.getpid()}:turn=fixture-peer"
            assert peer.try_acquire_session_turn_lease(sid, holder, ttl_seconds=30)
            try:
                peer.read_turn_admission(sid, holder)
            except Exception as exc:
                error = str(exc)
            finally:
                peer.release_session_turn_lease(sid, holder)
        else:
            # Real queue drain/recursive follow-up entry; recursive execution uses the native facade.
            second = _agent_with_db(peer, session_id=sid, platform="telegram")
            runner._delivery_adapter_for = lambda source: None
            runner._run_agent_deliver_first_response = AsyncMock()
            async def nested(**kwargs):
                return AIAgent.run_conversation(second, kwargs["message"], conversation_history=kwargs["history"])
            runner._run_agent = nested
            ctx = NS(source=source, session_id=sid, session_key=entry.session_key, run_generation=1,
                     _interrupt_depth=0, history=[], _status_thread_metadata=None, channel_prompt=None,
                     context_prompt="", result_holder=[raw])
            try:
                asyncio.run(runner._run_agent_queued_followup(ctx, None, "fixture queued input", None, raw, raw, None))
            except Exception as exc:
                error = str(exc)
        # The late gateway step can clear the marker, demonstrating the ordering gap itself.
        before = metadata(db, sid, row_id[0])
        asyncio.run(persist(runner, source, entry, raw, failed=False))
        after = metadata(db, sid, row_id[0])
        print({"followup": followup, "admission_error": error, "before_late_gateway": before,
               "after_late_gateway": after, "core_loop_calls": loop.call_count})
        assert after.get("codex_auth_pending") is False
        assert error is None, "Completed predecessor was still pending when its successor acquired the lease"
    finally:
        peer.close()
        db.close()
