from pathlib import Path

from hiveloom.paths import hiveloom_home
from hiveloom.trust import record_trust, trust_store_path


def test_explicit_home_owns_trust_store(monkeypatch, tmp_path):
    home = tmp_path / "writable"
    monkeypatch.setenv("HOME", "/nonexistent-read-only-home")
    monkeypatch.setenv("HIVELOOM_HOME", str(home))
    record_trust(tmp_path / "harness")
    assert trust_store_path() == home / "trust.json"
    assert trust_store_path().is_file()


def test_empty_override_uses_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HIVELOOM_HOME", "")
    assert hiveloom_home() == tmp_path / ".hiveloom"


def test_relative_override_returns_absolute_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIVELOOM_HOME", "scratch")
    assert hiveloom_home() == Path.cwd() / "scratch"
