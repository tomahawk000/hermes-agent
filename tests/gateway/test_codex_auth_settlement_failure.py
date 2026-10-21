"""A failed holder-qualified close cannot be repaired later without its lease."""
import asyncio

import pytest

from agent.session_persistence import _db_flush_row
from gateway.session_transcript import TranscriptReadError
from run_agent import AIAgent
from tests.agent.test_cross_process_turn_lease import _agent_with_db
from tests.gateway.test_codex_auth_terminal_integration import setup_store, worker_result, persist


@pytest.mark.parametrize('case', ['write_failure', 'unfinished_call'])
def test_unproven_primary_completion_retains_fence_through_worker_and_gateway(tmp_path, monkeypatch, case):
    store, source, entry, runner = setup_store(tmp_path, monkeypatch)
    db, sid = runner._session_db, entry.session_id
    agent = _agent_with_db(db, session_id=sid, platform='telegram')
    agent.provider = 'openai-codex'

    def loop(agent, message, *_args, **_kwargs):
        holder = agent._active_session_turn_lease_holder
        user = {'role':'user', 'content':message, 'message_id':'fixture-input',
                'display_metadata':{'gateway_input_owner':'fixture-owner'}}
        db.append_messages_batch(sid, [_db_flush_row(agent, user, True)], turn_lease_holder=holder)
        if case == 'unfinished_call':
            db.append_message(sid, 'assistant', turn_lease_holder=holder, tool_calls=[{
                'id':'fixture-call', 'type':'function', 'function':{'name':'fixture_effect','arguments':'{}'}}])
        db.append_message(sid, 'assistant', 'fixture response', turn_lease_holder=holder)
        return {'messages':db.get_messages_as_conversation(sid, include_row_ids=True),
                'completed':True, 'failed':False, 'current_turn_user_idx':0,
                'final_response':'fixture response'}

    monkeypatch.setattr('agent.conversation_loop.run_conversation', loop)
    real_settle = db.settle_gateway_input
    if case == 'write_failure':
        def refuse(*_args, **_kwargs):
            raise RuntimeError('fixture transaction failure')
        monkeypatch.setattr(db, 'settle_gateway_input', refuse)
    try:
        raw = AIAgent.run_conversation(agent, 'fixture request', conversation_history=[],
            persist_user_display_metadata={'gateway_input_owner':'fixture-owner'})
        assert raw['failed'] and not raw['completed']
        assert raw['request_persistence']['settlement_attempted']
        assert not raw['request_persistence']['settled']
        monkeypatch.setattr(db, 'settle_gateway_input', real_settle)
        before = db.get_messages(sid, include_inactive=True)
        mapped = worker_result(monkeypatch, source, sid, raw)
        asyncio.run(persist(runner, source, entry, mapped, failed=True))
        assert db.get_messages(sid, include_inactive=True) == before
        assert before[0]['display_metadata']['codex_auth_pending'] is True
        with pytest.raises(TranscriptReadError) as refusal:
            store.load_transcript(sid)
        assert isinstance(refusal.value.__cause__, ValueError)
        assert 'no durable completion or failure boundary' in str(refusal.value.__cause__)
    finally:
        db.close()
