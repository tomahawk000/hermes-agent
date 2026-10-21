"""Saved Codex failures are truthful, pinned and never replayed by generic /retry."""
import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


@pytest.mark.parametrize("case", ["saved", "unsaved", "unreadable", "tools", "unknown", "newer", "interrupted", "attachment"])
def test_auth_failure_receipt_and_retry_preserve_input(tmp_path, monkeypatch, case):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._session_db = object()
        runner._refresh_agent_cache_message_count = AsyncMock()
        runner._handle_message = AsyncMock()
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="offline", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        user = {"role": "user", "content": "offline request"}
        if case == "attachment":
            user["content"] = [{"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}]
        if case != "unsaved":
            row_id = db.append_message(sid, "user", user["content"], platform_message_id="offline-input")
        messages = [user]
        if case in {"tools", "interrupted"}:
            tool_call = {"role": "assistant", "tool_calls": [{"id": "offline-tool", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}
            tool_result = {"role": "tool", "content": "offline result", "tool_call_id": "offline-tool"}
            db.append_message(sid, "assistant", tool_calls=tool_call["tool_calls"])
            db.append_message(sid, "tool", "offline result", tool_call_id="offline-tool")
            messages.extend([tool_call, tool_result])
        if case == "newer":
            db.append_message(sid, "user", "newer offline input", platform_message_id="newer-input")
        if case == "unreadable":
            monkeypatch.setattr(db, "_read_one", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline read failure")))
        failed = {"failed": True, "completed": False, "interrupted": case == "interrupted", "agent_persisted": True,
                  "auth_failure": {"provider": "openai-codex"}, "messages": [] if case == "unknown" else messages,
                  "history_offset": 0, "current_turn_user_idx": 0, "last_prompt_tokens": 0}
        for _ in range(2):
            await runner._hmwa_persist_turn_transcript(
                event=MessageEvent(text="offline request", source=source, message_id="offline-input"),
                source=source, session_entry=entry, session_key=entry.session_key, agent_result=failed,
                agent_messages=failed["messages"], prepared=runner._PreparedTurn([user], "", "offline request", None, None, None, sid, "owner"),
                response="Codex access token expired", agent_failed_early=True, hidden_reasoning_incomplete=False,
                is_context_overflow_failure=False,
            )
        notice = runner._hmwa_failed_turn_notice(failed)
        receipt = failed["request_persistence"]
        assert receipt["saved"] is (case not in {"unsaved", "unreadable"})
        assert ("Your request is saved." in notice) is receipt["saved"]
        assert ("No tools are recorded" in notice) is (case in {"saved", "attachment"})
        assert "send it again" not in notice and "before resending" not in notice
        if case not in {"unsaved", "unreadable", "newer"}:
            before = db.get_messages(sid, include_inactive=True)
            assert receipt["row_id"] == row_id and receipt["retry_blocked"] is True
            assert sum(m["role"] == "assistant" and bool((m.get("display_metadata") or {}).get("codex_auth_failure")) for m in before) == 1
            assert (next(m for m in before if m["id"] == row_id)["display_metadata"] or {}).get("codex_auth_failure") is True
            reply = await runner._handle_retry_command(MessageEvent(text="/retry", source=source))
            assert "Cannot retry this Codex" in reply
            assert db.get_messages(sid, include_inactive=True) == before
            runner._handle_message.assert_not_awaited()
        else:
            assert receipt["retry_blocked"] is True
            assert receipt["boundary_saved"] is (case == "unreadable")
            before = db.get_messages(sid, include_inactive=True)
            await runner._handle_retry_command(MessageEvent(text="/retry", source=source))
            assert db.get_messages(sid, include_inactive=True) == before
            runner._handle_message.assert_not_awaited()
        db.close()
    asyncio.run(check())


@pytest.mark.parametrize("race", ["newer_user", "duplicate_boundary", "existing_assistant", "lease"])
def test_auth_boundary_compare_and_swap_never_closes_newer_input(tmp_path, monkeypatch, race):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="offline", user_id="u")
    sid = store.get_or_create_session(source).session_id
    db = store._db_for_session_id(sid)
    row_id = db.append_message(sid, "user", "offline request", platform_message_id="offline-input")
    evidence = store.failed_request_evidence(sid, "offline-input")
    assert evidence["saved"] and evidence["current"]
    if race == "newer_user":
        db.append_message(sid, "user", "newer request", platform_message_id="newer-input")
    elif race == "existing_assistant":
        db.append_message(sid, "assistant", "existing boundary", display_metadata={"other": True})
    elif race == "lease":
        assert db.try_acquire_session_turn_lease(sid, f"pid={os.getpid()}:turn=offline", ttl_seconds=60)
    before = db.get_messages(sid)
    expected = race not in {"newer_user", "lease"}
    assert store.close_failed_auth_turn(sid, row_id, "offline failure boundary") is expected
    if expected:
        after = db.get_messages(sid)
        assert store.close_failed_auth_turn(sid, row_id, "offline failure boundary")
        assert db.get_messages(sid) == after
        assert [m["role"] for m in after] == ["user", "assistant"]
        if race == "existing_assistant":
            assert after[-1]["content"] == "existing boundary" and after[-1]["display_metadata"]["other"]
    else:
        assert db.get_messages(sid) == before
    db.close()
