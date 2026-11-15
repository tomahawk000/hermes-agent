"""Transient-fetch retry for uv network operations (hermes-agent #45/#46).

A macOS CI leg failed an install/update E2E run because `uv sync` hit a
one-off DNS error fetching a wheel. The retry must cover exactly the fetch
class and never retry resolver conflicts or build failures, which repeat
identically and would only multiply wall-clock cost.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from pm.environment import PythonEnvironment, is_transient_fetch_failure
from pm.package import InstallError
from pm.environment import ResolutionConflict, BuildFailure


_DNS_ERROR_OUTPUT = (
    "error: Failed to fetch: `https://files.pythonhosted.org/packages/x/y.whl`\n"
    "  ├─▶ client error (Connect)\n"
    "  ├─▶ dns error\n"
    "  ╰─▶ failed to lookup address information: nodename nor servname provided\n"
)

_RESOLVER_OUTPUT = (
    "error: Because only the following versions are available and your pyproject depends on\n"
    "them, we can conclude that there is no solution found when resolving dependencies.\n"
)

_BUILD_OUTPUT = "error: the build backend returned an error while building pkg\n"

_OTHER_FAILURE_OUTPUT = "error: unexpected exit while preparing the environment\n"


def _environment(tmp_path) -> PythonEnvironment:
    return PythonEnvironment(
        uv=tmp_path / "uv",
        python=Path(sys.executable),
        destination=tmp_path / "dest",
        cache=tmp_path / "cache",
        env={},
    )


def _failed(output: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=output)


def _succeeded() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


@pytest.fixture
def no_delay(monkeypatch):
    monkeypatch.setenv("PM_UV_FETCH_RETRY_DELAY", "0")


def _patch_run(monkeypatch, outcomes):
    """Queue CompletedProcess outcomes; return the recorded call count."""
    calls = {"count": 0}

    def fake_run(self, args, *, cwd, timeout):
        calls["count"] += 1
        index = min(calls["count"] - 1, len(outcomes) - 1)
        return outcomes[index]

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    return calls


def test_transient_fetch_markers_detected():
    assert is_transient_fetch_failure(_DNS_ERROR_OUTPUT)
    assert is_transient_fetch_failure("error sending request for url (https://x) ... connection reset")
    assert not is_transient_fetch_failure(_RESOLVER_OUTPUT)
    assert not is_transient_fetch_failure(_BUILD_OUTPUT)
    assert not is_transient_fetch_failure(_OTHER_FAILURE_OUTPUT)


def test_resolver_and_build_markers_win_over_fetch_markers():
    """Mixed output (a conflict page can mention a URL) must never retry."""
    mixed = _RESOLVER_OUTPUT + _DNS_ERROR_OUTPUT
    assert not is_transient_fetch_failure(mixed)
    mixed_build = _BUILD_OUTPUT + _DNS_ERROR_OUTPUT
    assert not is_transient_fetch_failure(mixed_build)


def test_sync_retries_transient_fetch_then_succeeds(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    calls = _patch_run(monkeypatch, [_failed(_DNS_ERROR_OUTPUT), _succeeded()])

    environment.sync(tmp_path)

    assert calls["count"] == 2


def test_sync_never_retries_resolver_conflict(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    calls = _patch_run(monkeypatch, [_failed(_RESOLVER_OUTPUT)])

    with pytest.raises(ResolutionConflict):
        environment.sync(tmp_path)

    assert calls["count"] == 1


def test_sync_never_retries_build_failure(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    calls = _patch_run(monkeypatch, [_failed(_BUILD_OUTPUT)])

    with pytest.raises(BuildFailure):
        environment.sync(tmp_path)

    assert calls["count"] == 1


def test_sync_never_retries_unclassified_failure(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    calls = _patch_run(monkeypatch, [_failed(_OTHER_FAILURE_OUTPUT)])

    with pytest.raises(InstallError):
        environment.sync(tmp_path)

    assert calls["count"] == 1


def test_sync_gives_up_after_bounded_retries(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    monkeypatch.setenv("PM_UV_FETCH_RETRIES", "2")
    calls = _patch_run(monkeypatch, [_failed(_DNS_ERROR_OUTPUT)])

    with pytest.raises(InstallError):
        environment.sync(tmp_path)

    assert calls["count"] == 3


def test_retry_count_env_override_disables_retries(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    monkeypatch.setenv("PM_UV_FETCH_RETRIES", "0")
    calls = _patch_run(monkeypatch, [_failed(_DNS_ERROR_OUTPUT)])

    with pytest.raises(InstallError):
        environment.sync(tmp_path)

    assert calls["count"] == 1


def test_lock_retries_transient_fetch(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    calls = _patch_run(monkeypatch, [_failed(_DNS_ERROR_OUTPUT), _succeeded()])

    environment.lock(tmp_path)

    assert calls["count"] == 2


def test_pip_install_retries_transient_fetch(tmp_path, monkeypatch, no_delay):
    environment = _environment(tmp_path)
    calls = _patch_run(monkeypatch, [_failed(_DNS_ERROR_OUTPUT), _succeeded()])

    environment._install_requirements_file(tmp_path / "requirements.txt")

    assert calls["count"] == 2
