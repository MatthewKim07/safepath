import pytest

from safepath import cli, osm_context
from safepath.config import load_config
from safepath.fetch import FetchError, require_rows, with_retries


def test_with_retries_raises_after_last_attempt():
    calls = []

    def boom():
        calls.append(1)
        raise ConnectionError("overpass said no")

    with pytest.raises(FetchError, match="after 3 attempts: overpass said no") as info:
        with_retries(boom, "test fetch", attempts=3, backoff_seconds=0)
    assert len(calls) == 3
    assert isinstance(info.value.__cause__, ConnectionError)


def test_with_retries_recovers_from_transient_failures():
    outcomes = iter([TimeoutError("slow"), "ok"])

    def flaky():
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result

    assert with_retries(flaky, "test fetch", attempts=3, backoff_seconds=0) == "ok"


def test_require_rows_rejects_empty():
    with pytest.raises(FetchError, match="returned no rows"):
        require_rows([], "empty layer")


def test_layer_fetch_failure_raises_and_writes_no_cache(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["paths"]["processed_dir"] = tmp_path
    cfg["paths"]["raw_dir"] = tmp_path
    cfg["osm"] = {"fetch_attempts": 2, "backoff_seconds": 0}

    def fail(*args, **kwargs):
        raise RuntimeError("HTTP 429 Too Many Requests")

    monkeypatch.setattr(osm_context.ox, "features_from_place", fail)
    with pytest.raises(FetchError, match="layer 'tram' failed after 2 attempts"):
        osm_context.load_or_fetch_layer(cfg, "tram")
    assert not list(tmp_path.glob("osm_*.parquet"))


def test_all_stops_at_first_failing_step(monkeypatch):
    ran = []

    def ok(name):
        return lambda cfg: ran.append(name)

    def fail(cfg):
        ran.append("graph")
        raise FetchError("graph download failed after 3 attempts")

    steps = {name: (help_text, ok(name)) for name, (help_text, _) in cli.STEPS.items()}
    steps["graph"] = ("graph", fail)
    monkeypatch.setattr(cli, "STEPS", steps)
    assert cli.main(["all"]) == 1
    assert ran == cli.PIPELINE[: cli.PIPELINE.index("graph") + 1]
