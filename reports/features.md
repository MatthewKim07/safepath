# Static intersection features

22,435 intersections x 27 features, from the current OSM snapshot (so they describe today's roads, not the roads in each label year). No raw coordinates or neighbourhood IDs. Missing maxspeed / lanes on an approach are imputed with the median for its road class; `maxspeed_missing` / `lanes_missing` mark intersections where no approach carried the tag. Unhandled NaNs: 0.

| feature | type | % NaN | % non-zero / true | median | p99 | levels |
|---|---|---:|---:|---:|---:|---:|
| n_osm_nodes | numeric | 0.0 | 100.0 | 1.0 | 4.0 | 0 |
| street_count | numeric | 0.0 | 100.0 | 3.0 | 5.0 | 0 |
| max_class_rank | numeric | 0.0 | 99.9 | 2.0 | 7.0 | 0 |
| min_class_rank | numeric | 0.0 | 99.7 | 1.0 | 5.0 | 0 |
| class_combo | category | 0.0 | nan | nan | nan | 10 |
| n_legs | numeric | 0.0 | 100.0 | 3.0 | 4.0 | 0 |
| min_leg_angle_deg | numeric | 0.0 | 100.0 | 88.9 | 180.0 | 0 |
| n_approaches | numeric | 0.0 | 100.0 | 3.0 | 6.0 | 0 |
| max_maxspeed_kph | numeric | 0.0 | 100.0 | 40.0 | 100.0 | 0 |
| min_maxspeed_kph | numeric | 0.0 | 100.0 | 30.0 | 100.0 | 0 |
| maxspeed_missing | bool | 0.0 | 29.7 | nan | nan | 2 |
| max_lanes | numeric | 0.0 | 100.0 | 2.0 | 7.0 | 0 |
| lanes_missing | bool | 0.0 | 0.3 | nan | nan | 2 |
| n_oneway_approaches | numeric | 0.0 | 28.5 | 0.0 | 6.0 | 0 |
| any_bridge_tunnel | bool | 0.0 | 5.0 | nan | nan | 2 |
| has_motorway_approach | bool | 0.0 | 4.0 | nan | nan | 2 |
| n_signals_30m | numeric | 0.0 | 11.4 | 0.0 | 4.0 | 0 |
| n_stop_signs_30m | numeric | 0.0 | 60.8 | 1.0 | 3.0 | 0 |
| n_crossings_30m | numeric | 0.0 | 71.7 | 1.0 | 8.0 | 0 |
| has_signal | bool | 0.0 | 11.4 | nan | nan | 2 |
| n_bus_stops_50m | numeric | 0.0 | 23.3 | 0.0 | 4.0 | 0 |
| n_pois_100m | numeric | 0.0 | 38.2 | 0.0 | 41.0 | 0 |
| dist_tram_m | numeric | 0.0 | 100.0 | 5612.0 | 16787.5 | 0 |
| streetcar_20m | bool | 0.0 | 3.9 | nan | nan | 2 |
| dist_motorway_m | numeric | 0.0 | 96.4 | 1577.7 | 5752.6 | 0 |
| dist_retail_m | numeric | 0.0 | 99.3 | 609.4 | 2044.1 | 0 |
| dist_nearest_intersection_m | numeric | 0.0 | 100.0 | 80.6 | 219.4 | 0 |

# History features

Built per label year Y from years strictly before Y (see `features/history.py` and the leakage test in `tests/test_history.py`): prev1 and prev3-mean crash and severe counts, the same as shares of the citywide intersection total of those years, years since the last crash (capped at 3), and `covid_in_window` when Y-3..Y-1 touches 2020-2021. Label year 2016 only has 2014-2015 behind it, so its prev3 mean is over two years (`n_history_years`).

## Drift check (clean intersections)

| split | rows | mean_label | mean_prev1 | mean_prev3 | mean_prev1_share_x1e4 | covid_window_rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 134,424 | 2.5 | 2.3 | 2.2 | 0.4 | 0.3 |
| val | 22,404 | 2.4 | 2.4 | 2.0 | 0.4 | 1.0 |
| test | 22,404 | 2.4 | 2.4 | 2.3 | 0.4 | 0.0 |

| year | mean_label | mean_prev1 | mean_prev3 |
|---|---:|---:|---:|
| 2016 | 2.6 | 2.4 | 2.3 |
| 2017 | 2.7 | 2.6 | 2.4 |
| 2018 | 2.7 | 2.7 | 2.6 |
| 2019 | 2.7 | 2.7 | 2.6 |
| 2022 | 2.0 | 1.5 | 1.9 |
| 2023 | 2.4 | 2.0 | 1.7 |
| 2024 | 2.4 | 2.4 | 2.0 |
| 2025 | 2.4 | 2.4 | 2.3 |
