import os

import pandas as pd

from safepath import tracking
from safepath.config import load_config


def test_disabled_without_key_is_a_no_op(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)  # no .env here
    monkeypatch.setattr(tracking, "load_dotenv", lambda path=tmp_path / ".env": None)
    cfg = load_config()
    assert not tracking.enabled(cfg)
    with tracking.run(cfg, "x", "test", {"a": 1}) as wb:
        assert not wb.active
        wb.log({"m": 1.0})
        wb.table("t", pd.DataFrame({"a": [1]}))
        wb.image("i", tmp_path / "missing.png")
        wb.summary({"s": 2})


def test_no_wandb_switch_and_config_flag(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    cfg = load_config()
    monkeypatch.setenv(tracking.DISABLE_ENV, "1")
    assert not tracking.enabled(cfg)
    monkeypatch.delenv(tracking.DISABLE_ENV)
    cfg["tracking"]["enabled"] = False
    assert not tracking.enabled(cfg)


def test_load_dotenv_keeps_existing_values(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nWANDB_API_KEY='abc123'\nOTHER=1\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("OTHER", "keep")
    tracking.load_dotenv(env)
    assert os.environ["WANDB_API_KEY"] == "abc123"
    assert os.environ["OTHER"] == "keep"


def test_git_commit_is_a_hash():
    commit = tracking.git_commit()
    assert commit == "unknown" or len(commit.removesuffix("-dirty")) == 40
