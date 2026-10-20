"""Git 2.56 UCRT layout recovery without changing the host platform."""

from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import update_cmd_git


@pytest.fixture
def git_roots(tmp_path, monkeypatch):
    installed = tmp_path / "program-files"
    roots = {
        r"C:\Program Files": installed,
        r"C:\Program Files (x86)": tmp_path / "program-files-x86",
    }
    shared = tmp_path / "managed"
    profile = shared / "profiles" / "example"

    def fixture_path(value):
        return roots.get(str(value), Path(value))

    # Map install locations to real disposable directories. The host platform
    # and filesystem behavior remain real; only the command probe is mocked.
    monkeypatch.setattr(update_cmd_git, "Path", fixture_path)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: shared)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile)
    return installed / "Git", shared / "git", profile / "git"


def git_core(root, prefix):
    executable = root / prefix / "libexec" / "git-core" / "git.exe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.touch()
    return executable


@pytest.mark.parametrize("prefix", ["mingw64", "ucrt64"])
@pytest.mark.parametrize("layout", [0, 1, 2], ids=["installed", "shared", "profile"])
def test_locator_accepts_healthy_git_in_each_layout(git_roots, monkeypatch, prefix, layout):
    expected = git_core(git_roots[layout], prefix)
    probes = []

    def healthy(argv):
        probes.append(argv)
        return False

    monkeypatch.setattr(update_cmd_git, "_probe_fork_bomb", healthy)
    assert update_cmd_git._locate_real_git() == expected
    assert probes == [[str(expected)]]


@pytest.mark.parametrize(
    "shared_probe,profile_available,expected_layout",
    [(False, True, 1), (True, True, 2), (None, True, 2), (True, False, None), (None, False, None)],
    ids=["shared-first", "trampoline-fallback", "failed-probe-fallback", "trampoline-only", "failed-probe-only"],
)
def test_locator_prefers_shared_root_only_after_a_healthy_probe(
    git_roots, monkeypatch, shared_probe, profile_available, expected_layout,
):
    shared = git_core(git_roots[1], "ucrt64")
    profile = git_core(git_roots[2], "mingw64") if profile_available else None
    monkeypatch.setattr(
        update_cmd_git, "_probe_fork_bomb", lambda argv: shared_probe if argv == [str(shared)] else False,
    )
    expected = {1: shared, 2: profile, None: None}[expected_layout]
    assert update_cmd_git._locate_real_git() == expected
