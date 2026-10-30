import pandas as pd
import pytest

from safepath.config import load_config
from safepath.data import volumes
from safepath.fetch import FetchError


def test_window_hours():
    assert volumes.window_hours("8R") == 8
    assert volumes.window_hours("8S") == 8
    assert volumes.window_hours("14") == 14


def _tmc_csv(path, **overrides):
    row = {
        "count_id": 1,
        "count_date": "2024-05-14",
        "location_name": "A St / B Ave",
        "longitude": -79.4,
        "latitude": 43.7,
        "centreline_type": 2,
        "centreline_id": 99,
        "count_duration": "8R",
        "total_vehicle": 100,
        "total_heavy_pct": 0.05,
        "n_appr_vehicle": 40,
        "e_appr_vehicle": 30,
        "s_appr_vehicle": 20,
        "w_appr_vehicle": 10,
    }
    row.update(overrides)
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_parse_tmc(tmp_path):
    out = volumes.parse_tmc(_tmc_csv(tmp_path / "tmc.csv"))
    assert out.loc[0, "entering"] == 100
    assert out.loc[0, "window_hours"] == 8
    assert out.loc[0, "n_entering"] == 40


def test_parse_tmc_rejects_inconsistent_totals(tmp_path):
    with pytest.raises(ValueError, match="sum of approach volumes"):
        volumes.parse_tmc(_tmc_csv(tmp_path / "tmc.csv", total_vehicle=101))


def test_parse_tmc_rejects_unknown_duration(tmp_path):
    with pytest.raises(ValueError, match="unknown TMC duration"):
        volumes.parse_tmc(_tmc_csv(tmp_path / "tmc.csv", count_duration="12"))


def test_svc_15min_sums_speed_bins(tmp_path):
    pd.DataFrame(
        {
            "count_id": [5, 5],
            "time_start": ["2024-05-14 00:00:00", "2024-05-14 00:00:00"],
            "direction": ["EB", "WB"],
            "vol_1_19kph": [1, 2],
            "vol_20_25kph": [3, 4],
        }
    ).to_csv(tmp_path / "speed.csv", index=False)
    out = volumes.load_svc_15min([tmp_path / "speed.csv"])
    assert out["volume"].tolist() == [4, 6]


def test_find_package_requires_exact_title(monkeypatch):
    cfg = load_config()
    cfg["volumes"]["fetch_attempts"] = 1
    found = {"results": [{"title": "Traffic Counts - Cycling", "name": "x", "id": "1"}]}
    monkeypatch.setattr(volumes, "_get_json", lambda *a, **k: found)
    with pytest.raises(FetchError, match="expected one package titled"):
        volumes.find_package(cfg, cfg["volumes"]["packages"]["tmc"])


def test_download_size_mismatch_raises_and_leaves_no_file(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["volumes"].update(fetch_attempts=2, backoff_seconds=0)

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"truncated"

    monkeypatch.setattr(volumes.requests, "get", lambda *a, **k: Resp())
    res = {"name": "tmc_summary_data.csv", "url": "https://example/x.csv", "size": 1000}
    dest = tmp_path / "x.csv"
    with pytest.raises(FetchError, match="after 2 attempts"):
        volumes.download_resource(cfg, res, dest)
    assert not dest.exists()


def test_attach_raw_windows_flags_missing_bins():
    tmc = pd.DataFrame({"count_id": [1, 2, 3], "duration": ["14", "14", "8R"]})
    full = volumes.window_bins("14")
    bins = pd.DataFrame(
        {
            "count_id": [1] * len(full) + [2] * (len(full) - 4),
            "minute": full + full[4:],
        }
    )
    out = volumes.attach_raw_windows(tmc, bins)
    assert out["raw_bins"].tolist()[:2] == [56, 52]
    assert out["window_ok"].tolist()[:2] == [True, False]
    assert pd.isna(out.loc[2, "window_ok"])
