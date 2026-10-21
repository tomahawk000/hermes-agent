"""Native steer provenance and recursive worker handoff of independent owned requests."""
import asyncio
import copy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.session_persistence import _db_flush_row, settle_gateway_turn_before_release
from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
from gateway.platforms.event import MessageEvent
from gateway.run_turn import _gateway_input_owner
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_cross_process_turn_lease import _agent_with_db
from tests.gateway.test_codex_auth_terminal_integration import setup_store, pending_input, worker_result, persist


@pytest.mark.parametrize('path',['pre_api','post_tool'])
@pytest.mark.parametrize('terminal',['complete','auth','auth_open'])
@pytest.mark.parametrize('layout',['plain','in_place','rotating'])
def test_consumed_steer_keeps_owned_execution_and_never_forges_delivery_identity(tmp_path,monkeypatch,path,terminal,layout):
    store, _source, entry, runner = setup_store(tmp_path,monkeypatch)
    db,sid=runner._session_db,entry.session_id
    holder='fixture-owner-holder'
    try:
        assert db.try_acquire_session_turn_lease(sid,holder,ttl_seconds=30)
        pending_input(db,sid,holder)
        db.append_message(sid,'assistant',turn_lease_holder=holder,tool_calls=[{
            'id':'fixture-tool','type':'function','function':{'name':'fixture_effect','arguments':'{}'}}])
        db.append_message(sid,'tool','fixture tool result',tool_call_id='fixture-tool',turn_lease_holder=holder)
        messages=db.get_messages_as_conversation(sid,include_row_ids=True)
        agent=NS(_gateway_input_owner='fixture-owner',_drain_pending_steer=lambda:'fixture correction')
        if path=='pre_api':
            _inject_steer_after_newest_tool_result(agent,messages,'fixture correction')
        else:
            apply_pending_steer_to_tool_results(agent,messages,1)
        steer=messages[-1]
        assert steer['display_kind']=='steer'
        assert steer['display_metadata']=={'gateway_input_owner':'fixture-owner'}
        assert 'platform_message_id' not in steer and 'message_id' not in steer
        assert 'codex_auth_pending' not in steer['display_metadata']
        db.append_messages_batch(sid,[steer],turn_lease_holder=holder)
        answer={'role':'assistant','content':'fixture final response'}
        if terminal != 'auth_open':
            db.append_messages_batch(sid,[answer],turn_lease_holder=holder)
            messages.append(answer)
        target=sid
        if layout=='in_place':
            db.archive_and_compact(sid,messages,tail_count=len(messages))
            messages=db.get_messages_as_conversation(sid,include_row_ids=True)
        elif layout=='rotating':
            target='fixture-tip'
            assert db.try_acquire_compression_lock(sid,'fixture-compressor',ttl_seconds=30)
            try:
                db.publish_compression_child(parent_session_id=sid,child_session_id=target,source='test',
                    messages=messages,compression_lock_holder='fixture-compressor')
            finally:
                db.release_compression_lock(sid,'fixture-compressor')
        agent._session_db=db
        agent.session_id=target
        agent._active_session_turn_lease_holder=holder
        agent._persist_disabled=False
        result={'messages':messages,'completed':terminal=='complete','failed':terminal!='complete'}
        if terminal!='complete':
            result['auth_failure']={'provider':'openai-codex','kind':'access_token_expired'}
        settle_gateway_turn_before_release(agent,result,{'gateway_input_owner':'fixture-owner'})
        assert result['request_persistence']['settled']
        assert result['request_persistence']['tools_recorded']
        rows=db.get_messages(target)
        users=[r for r in rows if r['role']=='user']
        assert len(users)==2
        assert all(r['display_metadata']['codex_auth_pending'] is False for r in users)
        assert all(m['display_metadata']['codex_auth_pending'] is False for m in messages if m['role']=='user')
        assert users[-1]['display_kind']=='steer' and users[-1]['platform_message_id'] is None
        assert users[0]['platform_message_id']=='fixture-input'
        assert users[0]['message_uid']!=users[-1]['message_uid']
        if terminal == 'auth_open':
            assert messages[-1]['role'] == 'assistant' and messages[-1]['_row_id'] == rows[-1]['id']
            assert messages[-1]['message_uid'] == rows[-1]['message_uid']
        assert store.failed_request_evidence(target,'fixture-input')['tools_recorded']
    finally:
        db.release_session_turn_lease(sid,holder)
        db.close()


@pytest.mark.parametrize('eventful',[False,True])
@pytest.mark.parametrize('terminal',['complete','auth'])
def test_real_recursive_queue_preserves_distinct_input_and_terminal_worker_receipt(tmp_path,monkeypatch,eventful,terminal):
    store,source,entry,runner=setup_store(tmp_path,monkeypatch)
    db,sid=runner._session_db,entry.session_id
    peer=SessionDB(db.db_path)
    calls=[]
    def loop(agent,message,_system,history,*args,**kwargs):
        owner=kwargs['persist_user_display_metadata']['gateway_input_owner']
        calls.append(owner)
        database=agent._session_db
        user={'role':'user','content':message,'display_metadata':{'gateway_input_owner':owner}}
        row=_db_flush_row(agent,user,True)
        if kwargs.get('persist_user_platform_id'):
            row['platform_message_id']=kwargs['persist_user_platform_id']
        database.append_messages_batch(sid,[row],turn_lease_holder=agent._active_session_turn_lease_holder)
        database.append_message(sid,'assistant','fixture terminal response',turn_lease_holder=agent._active_session_turn_lease_holder)
        result={'messages':database.get_messages_as_conversation(sid,include_row_ids=True),
                'completed':len(calls)==1 or terminal=='complete','failed':len(calls)>1 and terminal=='auth',
                'final_response':'fixture terminal response','current_turn_user_idx':len(history or [])}
        if result['failed']:
            result['auth_failure']={'provider':'openai-codex','kind':'access_token_expired'}
        return result
    monkeypatch.setattr('agent.conversation_loop.run_conversation',loop)
    first=_agent_with_db(db,session_id=sid,platform='telegram');first.provider='openai-codex'
    second=_agent_with_db(peer,session_id=sid,platform='telegram');second.provider='openai-codex'
    try:
        raw=AIAgent.run_conversation(first,'fixture first',conversation_history=[],
            persist_user_display_metadata={'gateway_input_owner':'first-owner'},persist_user_platform_id='first-input')
        mapped=worker_result(monkeypatch,source,sid,raw)
        runner._delivery_adapter_for=lambda source:None
        runner._run_agent_deliver_first_response=AsyncMock()
        if eventful:
            runner._prepare_profile_scoped_inbound_message_text=AsyncMock(return_value='fixture queued')
            runner._session_key_for_source=lambda source:entry.session_key
            runner._reply_anchor_for_event=lambda event:None
            runner._pinned_channel_inputs=lambda key,prompt,source,**kwargs:(prompt,source)
            runner._persist_prompt_pins=AsyncMock()
            runner._intake_adapter_for=lambda source:None
        async def nested(**kwargs):
            assert db.get_messages(sid)[0]['display_metadata']['codex_auth_pending'] is False
            result=AIAgent.run_conversation(second,kwargs['message'],conversation_history=kwargs['history'],
                persist_user_display_metadata=kwargs['persist_user_display_metadata'],
                persist_user_platform_id=kwargs.get('inbound_message_id'))
            return worker_result(monkeypatch,source,sid,result)
        runner._run_agent=nested
        ctx=NS(source=source,session_id=sid,session_key=entry.session_key,run_generation=1,
               _interrupt_depth=0,history=[],_status_thread_metadata=None,channel_prompt=None,context_prompt='',result_holder=[raw])
        event=MessageEvent(text='fixture queued',source=source,message_id='queued-input') if eventful else None
        result=asyncio.run(runner._run_agent_queued_followup(ctx,None,'fixture queued',event,mapped,raw,None))
        assert len(calls)==2 and calls[0]!=calls[1]
        receipt=result['request_persistence']
        assert receipt['settled'] and receipt['input_owner']==calls[1]
        assert receipt['outcome']==('completed' if terminal=='complete' else 'auth_failure')
        if eventful:
            assert calls[1]==_gateway_input_owner(source,'queued-input')
            changed=copy.copy(source);changed.profile='other-fixture-profile'
            assert _gateway_input_owner(changed,'queued-input')!=calls[1]
        # The outer handler still has the opening event/owner. It must not
        # overwrite the already-settled terminal invocation's receipt or root.
        asyncio.run(persist(runner,source,entry,result,failed=terminal=='auth'))
        users=[r for r in db.get_messages(sid) if r['role']=='user']
        assert len(users)==2
        assert users[0]['display_metadata']['gateway_input_outcome']=='completed'
        assert not users[0]['display_metadata'].get('codex_auth_failure')
        assert users[1]['display_metadata']['gateway_input_owner']==calls[1]
        assert users[1]['display_metadata']['gateway_input_outcome']==receipt['outcome']
        assert all(r['display_metadata']['codex_auth_pending'] is False for r in users)
        assert store.load_transcript(sid)
    finally:
        peer.close()
        db.close()
