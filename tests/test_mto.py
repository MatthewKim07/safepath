import numpy as np
import pandas as pd
import pytest

from safepath.data import mto


def test_parse_pdf_lines_carries_section_header():
    lines = [
        "Year Highway Location Description Dist",
        "1988 401 KEELE ST IC-362 NORTH YORK 3.1 UC 301,000 316,000 340,100 282,900 33,100 389 1.2 73 0.2",  # noqa: E501
        "1989 401 UC 305,000 320,000 344,000 286,000 33,500 400 1.2 70 0.2",
        "2020 401 N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A",
        "1988 409 HWY 401 ETOBICOKE 0.4 UC 83,000 87,200 93,800 78,000 6,650 31 2.9 6 0.6",
        "Page 892",
    ]  # fmt: skip
    out = mto.parse_pdf_lines(lines, {"401"})
    assert out["description"].tolist() == ["KEELE ST IC-362 NORTH YORK"] * 3
    assert out.loc[1, "collisions"] == 400
    assert out.loc[0, "cr"] == pytest.approx(1.2)
    assert np.isnan(out.loc[2, "aadt"])
    assert set(out["year"]) == {1988, 1989, 2020}


def test_attach_collisions_tolerates_rounded_length():
    sections = pd.DataFrame(
        {
            "hwy": ["401", "401"],
            "lhrs": [1, 2],
            "year": [2019, 2019],
            "description": ["KEELE ST IC-362-NORTH YORK", "HWY 401 IC"],
            "length_km": [3.05, 2.0],
            "order": [0, 1],
        }
    )
    pdf = pd.DataFrame(
        {
            "hwy": ["401", "401"],
            "description": ["KEELE ST IC-362 NORTH YORK", "HWY 401 IC"],
            "dist_km": [3.1, 2.5],
            "year": [2019, 2019],
            "collisions": [456.0, 10.0],
            "cr": [1.0, 0.3],
            "aadt": [404_200.0, 1.0],
        }
    )
    out = mto.attach_collisions(sections, pdf).set_index("lhrs")
    assert out.loc[1, "collisions"] == 456
    assert np.isnan(out.loc[2, "collisions"])  # same name, different length: not this section


def test_chainage_inside_and_beyond_boundaries():
    boundary = np.array([[0.0, 0.0], [100.0, 0.0], [300.0, 0.0]])
    pts = np.array([[50.0, 5.0], [200.0, -3.0], [-20.0, 0.0], [350.0, 1.0]])
    pos = mto.chainage(pts, boundary)
    assert pos == pytest.approx([0.5, 1.5, -0.2, 2.25])


def test_section_summary_pools_rate_over_window():
    rows = []
    for year, coll in [(2015, 100), (2016, 100), (2017, 100), (2018, 100), (2019, 100), (2021, 50)]:
        rows.append(
            {"hwy": "401", "lhrs": 1, "year": year, "description": "X", "length_km": 2.0,
             "aadt": 100_000, "alpha_code": "T", "order": 0, "collisions": coll, "cr": np.nan}
        )  # fmt: skip
    rows.append({**rows[-1], "year": 2024, "aadt": 120_000, "collisions": np.nan})
    out = mto.section_summary(pd.DataFrame(rows), 2024).iloc[0]
    mvkm = 100_000 * 365 * 2.0 / 1e6
    assert out["cr"] == pytest.approx(100 / mvkm)
    assert out["aadt_2024"] == 120_000


def test_fill_rates_uses_highway_pool():
    s = pd.DataFrame(
        {
            "hwy": ["427", "427", "427"],
            "cr": [0.5, np.nan, 1.0],
            "collisions_per_year": [50.0, np.nan, 100.0],
            "mvkm_per_year": [100.0, 80.0, 100.0],
            "mapped_km": [1.0, 1.0, 1.0],
        }
    )
    out = mto.fill_rates(s)
    assert out.loc[1, "cr"] == pytest.approx(150 / 200)
    assert out["cr_source"].tolist() == ["section", "highway pooled", "section"]
