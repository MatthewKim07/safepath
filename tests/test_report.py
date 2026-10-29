import pandas as pd

from safepath.data.report import year_table


def test_year_table_flags_partial_years_and_location_share():
    dates = pd.to_datetime(
        [f"2024-{m:02d}-15" for m in range(1, 13)] + ["2025-01-10", "2025-02-10"]
    )
    parsed = pd.DataFrame({"year": dates.year, "month": dates.month, "occ_date": dates})
    clean = parsed.iloc[[0, 1, 2, 12]].assign(
        injury_or_fatal=[True, False, False, True],
        is_fatal=[False, False, False, True],
        involves_vru=pd.array([True, False, pd.NA, False], dtype="boolean"),
    )
    t = year_table(parsed, clean)
    assert not t.loc[2024, "partial"] and t.loc[2025, "partial"]
    assert t.loc[2024, "pct_located"] == 25.0
    assert t.loc[2025, "pct_injury_or_fatal"] == 100.0
