import pandas as pd

from safepath.data.ottawa import _parse_dates, clean_records, is_freeway_location, location_parts

BBOX = {"min_lon": -76.4, "max_lon": -75.2, "min_lat": 44.9, "max_lat": 45.6}


def test_location_parts():
    assert location_parts("BANK ST @ WALKLEY RD (0011423)") == ("at", ["BANK ST", "WALKLEY RD"])
    assert location_parts("MERIVALE RD btwn CALDWELL AVE & CENTRAL PARK DR (__3Z094A)") == (
        "btwn",
        ["MERIVALE RD"],
    )
    kind, roads = location_parts("CATHERINE ST/HWY 417 O'CONN IC119BR76 @ O'CONNOR ST")
    assert kind == "at" and roads[0] == "CATHERINE ST" and len(roads) == 3


def test_freeway_rule_keeps_ramp_terminals():
    assert is_freeway_location("HIGHWAY 417 btwn HWY417 IC124 RAMP65 & HWY417 IC124 RAMP67")
    assert is_freeway_location("HWY416 IC66 RAMP62 btwn HIGHWAY 416 & FALLOWFIELD RD")
    assert is_freeway_location("HWY 417 NICHOLAS IC118R36 @ HWY 417 NICHOLAS IC118R43")
    # ramp terminal on a city street: a surface intersection
    assert not is_freeway_location("BRONSON AVE @ HWY 417 BRONSON IC121AR51 (0002693)")
    # city road segment whose ends are ramps
    assert not is_freeway_location("RIVERSIDE DR SB btwn HWY417 IC117 RAMP53 & HUNT CLUB RD")
    # city expressway stays, like the Gardiner / DVP in Toronto
    assert not is_freeway_location("REGIONAL ROAD 174 EB btwn OR174 IC102 RAMP51 & X")
    assert not is_freeway_location("")


def test_parse_dates_handles_every_layer_format():
    out = _parse_dates(pd.Series(["2016-Jul-13", "15-Jul-15", "1552968000000", "2023-10-25"]))
    assert [d.date().isoformat() for d in out] == [
        "2016-07-13", "2015-07-15", "2019-03-19", "2023-10-25"
    ]  # fmt: skip


def test_clean_drops_non_reportable_and_bad_coords_and_flags_freeway():
    raw = pd.DataFrame(
        {
            "source_id": ["17--1", "17--2", "17--3", "17--4"],
            "occ_date": pd.to_datetime(["2017-01-01"] * 4),
            "lon": [-75.7, -75.7, 0.0, -75.6],
            "lat": [45.4, 45.4, 0.0, 45.3],
            "classification": [
                "02 - Non-fatal injury",
                "04 - Non-reportable",
                "03 - P.D. only",
                "01 - Fatal injury",
            ],  # fmt: skip
            "location_text": [
                "A ST @ B ST",
                "A ST @ B ST",
                "C ST @ D ST",
                "HIGHWAY 417 btwn X & Y",
            ],  # fmt: skip
        }
    )
    out, stats = clean_records(raw, BBOX)
    assert list(out["event_id"]) == ["2017-1", "2017-4"]
    assert list(out["injury_or_fatal"]) == [True, True]
    assert list(out["is_freeway"]) == [False, True]
    assert stats["dropped_non_reportable_by_year"] == {2017: 1}
    assert out["is_fail_to_remain"].isna().all()
