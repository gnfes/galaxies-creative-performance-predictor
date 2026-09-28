import importlib
from pathlib import Path

import pytest

from creative_predictor import config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("CP_TEST_KEY", "abc123")
    assert config.require_env("CP_TEST_KEY") == "abc123"


def test_require_env_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("CP_TEST_KEY", "  abc123 ")
    assert config.require_env("CP_TEST_KEY") == "abc123"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_require_env_rejects_missing_or_blank(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CP_TEST_KEY", raising=False)
    else:
        monkeypatch.setenv("CP_TEST_KEY", value)
    with pytest.raises(RuntimeError, match="CP_TEST_KEY"):
        config.require_env("CP_TEST_KEY")


def test_default_root_is_repo_root_when_running_from_source():
    assert config._default_root() == REPO_ROOT


def test_default_root_falls_back_to_cwd_when_installed(tmp_path, monkeypatch):
    installed = tmp_path / "lib/python3.12/site-packages/creative_predictor/config.py"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    assert config._default_root(installed) == workdir


def test_root_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATIVE_PREDICTOR_ROOT", str(tmp_path))
    try:
        reloaded = importlib.reload(config)
        assert reloaded.ROOT_DIR == tmp_path.resolve()
        assert reloaded.RAW_DIR == tmp_path.resolve() / "data" / "raw"
    finally:
        monkeypatch.delenv("CREATIVE_PREDICTOR_ROOT")
        importlib.reload(config)
