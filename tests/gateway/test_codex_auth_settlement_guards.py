"""Owned terminal writes keep uncertain effects, fencing and replay permission distinct."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent
from hermes_state import SessionDB
from hermes_state_retry import CodexAuthResumeBlocked
from tests.gateway.test_codex_auth_terminal_integration import setup_store, pending_input


@pytest.mark.parametrize('case,settled,outcome', [
    ('complete', True, 'completed'), ('transient', True, 'retryable_failure'),
    ('effects', True, 'review_required'), ('auth', True, 'auth_failure'),
    ('interrupt', False, None), ('unfinished_call', False, None),
    ('missing_terminal', False, None), ('newer', False, None),
    ('foreign_steer', False, None), ('lookalike', False, None),
    ('expired_holder', False, None), ('wrong_holder', False, None),
    ('rollback', False, None), ('malformed', False, None),
])
def test_terminal_settlement_requires_its_lease_input_and_effect_evidence(tmp_path, monkeypatch, case, settled, outcome):
    store, source, entry, runner = setup_store(tmp_path, monkeypatch)
    db, sid = runner._session_db, entry.session_id
    holder = 'fixture-admitted-holder'
    assert db.try_acquire_session_turn_lease(sid, holder, ttl_seconds=30)
    try:
        _, uid = pending_input(db, sid, holder)
        if case in {'effects', 'unfinished_call'}:
            db.append_message(sid, 'assistant', turn_lease_holder=holder, tool_calls=[{
                'id':'fixture-call', 'type':'function', 'function':{'name':'fixture_effect', 'arguments':'{}'}}])
            if case == 'effects':
                db.append_message(sid, 'tool', 'fixture effect recorded', tool_call_id='fixture-call', turn_lease_holder=holder)
        if case in {'newer','foreign_steer','lookalike'}:
            db.append_message(sid, 'user', 'fixture later input', turn_lease_holder=holder,
                display_kind='steer' if case == 'foreign_steer' else None,
                display_metadata={'gateway_input_owner':'foreign'} if case != 'lookalike' else None)
        if case not in {'missing_terminal','rollback'}:
            db.append_message(sid, 'assistant', 'fixture terminal boundary', turn_lease_holder=holder)
        if case == 'malformed':
            db._execute_write(lambda conn: conn.execute('UPDATE messages SET display_metadata=? WHERE id=?', ('{',uid)))
        if case == 'expired_holder':
            db._execute_write(lambda conn: conn.execute('UPDATE session_turn_leases SET expires_at=0 WHERE holder=?', (holder,)))
        before = db.get_messages(sid, include_inactive=True)
        if case == 'rollback':
            def abort(*args, **kwargs):
                raise RuntimeError('fixture rollback after row updates and boundary insert')
            monkeypatch.setattr(db, '_bump_session_counters', abort)
        kwargs = dict(input_owner='fixture-owner', user_row_id=uid, turn_lease_holder='wrong' if case == 'wrong_holder' else holder,
                      completed=case in {'complete','unfinished_call','interrupt'},
                      failed=case not in {'complete','unfinished_call','interrupt'},
                      auth_failure=case in {'auth','rollback'}, interrupted=case == 'interrupt')
        if case in {'expired_holder','wrong_holder','rollback','malformed'}:
            with pytest.raises((ValueError,RuntimeError)):
                db.settle_gateway_input(sid, **kwargs)
        else:
            receipt = db.settle_gateway_input(sid, **kwargs)
            assert receipt['settled'] is settled
            if settled:
                assert receipt['outcome'] == outcome
                assert receipt['retry_blocked'] is (case in {'auth','effects'})
        after = db.get_messages(sid, include_inactive=True)
        if not settled:
            assert after == before
        else:
            assert after[0]['display_metadata']['codex_auth_pending'] is False
            assert after[0]['display_metadata']['gateway_input_outcome'] == outcome
            assert store.load_transcript(sid)
        db.release_session_turn_lease(sid, holder)
        if settled:
            assert db.try_acquire_session_turn_lease(sid, 'fixture-next-holder', ttl_seconds=30)
            try:
                if outcome == 'completed':
                    assert db.read_turn_admission(sid, 'fixture-next-holder', reject_auth_failure=True)[2]
                else:
                    with pytest.raises(CodexAuthResumeBlocked):
                        db.read_turn_admission(sid, 'fixture-next-holder', reject_auth_failure=True)
            finally:
                db.release_session_turn_lease(sid, 'fixture-next-holder')
            runner._handle_message = AsyncMock(return_value='fixture explicit retry')
            response = asyncio.run(runner._handle_retry_command(MessageEvent(text='/retry',source=source)))
            if outcome in {'completed','retryable_failure'}:
                assert response == 'fixture explicit retry'
                runner._handle_message.assert_awaited_once()
            else:
                runner._handle_message.assert_not_awaited()
                assert db.get_messages(sid, include_inactive=True) == after
    finally:
        db.release_session_turn_lease(sid, holder)
        db.close()


@pytest.mark.parametrize('layout', ['plain','in_place','rotating'])
@pytest.mark.parametrize('outcome', ['completed','retryable_failure','review_required','auth_failure'])
def test_terminal_replay_policy_survives_compaction_without_blocking_new_requests(tmp_path, monkeypatch, layout, outcome):
    _store, _source, entry, runner = setup_store(tmp_path, monkeypatch)
    db, sid = runner._session_db, entry.session_id
    holder = 'fixture-first-holder'
    try:
        assert db.try_acquire_session_turn_lease(sid, holder, ttl_seconds=30)
        pending_input(db, sid, holder)
        if outcome == 'review_required':
            db.append_message(sid, 'assistant', turn_lease_holder=holder, tool_calls=[{
                'id':'fixture-call','type':'function','function':{'name':'fixture_effect','arguments':'{}'}}])
            db.append_message(sid, 'tool', 'recorded result', tool_call_id='fixture-call', turn_lease_holder=holder)
        db.append_message(sid, 'assistant', 'fixture terminal result', turn_lease_holder=holder)
        receipt = db.settle_gateway_input(sid, input_owner='fixture-owner',turn_lease_holder=holder,
            completed=outcome=='completed',failed=outcome!='completed',auth_failure=outcome=='auth_failure')
        assert receipt['outcome'] == outcome and receipt['settled']
        db.release_session_turn_lease(sid,holder)
        target = sid
        summary = [{'role':'user','content':'fixture summary only','_compressed_summary':True}]
        if layout == 'in_place':
            db.archive_and_compact(sid,summary)
        elif layout == 'rotating':
            target='fixture-tip'
            assert db.try_acquire_compression_lock(sid,'fixture-compressor',ttl_seconds=30)
            try:
                db.publish_compression_child(parent_session_id=sid,child_session_id=target,source='test',
                    messages=summary,compression_lock_holder='fixture-compressor')
            finally:
                db.release_compression_lock(sid,'fixture-compressor')
        peer=SessionDB(db.db_path)
        try:
            assert peer.try_acquire_session_turn_lease(sid,'fixture-restart',ttl_seconds=30)
            try:
                assert peer.read_turn_admission(sid,'fixture-restart')[0] == target
                if outcome == 'completed':
                    assert peer.read_turn_admission(sid,'fixture-restart',reject_auth_failure=True)[0] == target
                else:
                    with pytest.raises(CodexAuthResumeBlocked):
                        peer.read_turn_admission(sid,'fixture-restart',reject_auth_failure=True)
            finally:
                peer.release_session_turn_lease(sid,'fixture-restart')
            if outcome in {'review_required','auth_failure'}:
                from hermes_state_retry import validate_retry_history
                snapshot = peer.get_messages_as_conversation(target, repair_alternation=True)
                with pytest.raises(ValueError):
                    peer._execute_write(lambda conn: validate_retry_history(peer,conn,target,snapshot))
            # An explicit newer request supersedes the previous terminal disposition.
            peer.append_message(target,'user','fixture new independent request')
            peer.append_message(target,'assistant','fixture new completed answer')
            assert peer.try_acquire_session_turn_lease(target,'fixture-new-human',ttl_seconds=30)
            try:
                assert peer.read_turn_admission(target,'fixture-new-human',reject_auth_failure=True)[2]
            finally:
                peer.release_session_turn_lease(target,'fixture-new-human')
        finally:
            peer.close()
    finally:
        db.release_session_turn_lease(sid,holder)
        db.close()
