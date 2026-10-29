import pytest

from safepath.cli import build_parser, main
from safepath.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["download"]["page_size"] == 2000
    assert cfg["paths"]["raw_dir"].is_absolute()


def test_cli_lists_all_steps():
    help_text = build_parser().format_help()
    for step in ["download", "graph", "snap", "features", "train", "evaluate", "app", "all"]:
        assert step in help_text


def test_unknown_step_exits():
    with pytest.raises(SystemExit):
        main(["nope"])
