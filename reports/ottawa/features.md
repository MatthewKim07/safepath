# Static intersection features

15,276 intersections x 27 features, from the current OSM snapshot (so they describe today's roads, not the roads in each label year). No raw coordinates or neighbourhood IDs. Missing maxspeed / lanes on an approach are imputed with the median for its road class; `maxspeed_missing` / `lanes_missing` mark intersections where no approach carried the tag. Unhandled NaNs: 0.

| feature | type | % NaN | % non-zero / true | median | p99 | levels |
|---|---|---:|---:|---:|---:|---:|
| n_osm_nodes | numeric | 0.0 | 100.0 | 1.0 | 4.0 | 0 |
| street_count | numeric | 0.0 | 100.0 | 3.0 | 6.0 | 0 |
| max_class_rank | numeric | 0.0 | 100.0 | 2.0 | 6.0 | 0 |
| min_class_rank | numeric | 0.0 | 99.9 | 1.0 | 5.0 | 0 |
| class_combo | category | 0.0 | nan | nan | nan | 10 |
| n_legs | numeric | 0.0 | 100.0 | 3.0 | 4.0 | 0 |
| min_leg_angle_deg | numeric | 0.0 | 100.0 | 88.7 | 180.0 | 0 |
| n_approaches | numeric | 0.0 | 100.0 | 3.0 | 7.0 | 0 |
| max_maxspeed_kph | numeric | 0.0 | 100.0 | 40.0 | 80.0 | 0 |
| min_maxspeed_kph | numeric | 0.0 | 100.0 | 40.0 | 65.0 | 0 |
| maxspeed_missing | bool | 0.0 | 25.3 | nan | nan | 2 |
| max_lanes | numeric | 0.0 | 100.0 | 2.0 | 5.0 | 0 |
| lanes_missing | bool | 0.0 | 2.7 | nan | nan | 2 |
| n_oneway_approaches | numeric | 0.0 | 28.2 | 0.0 | 6.0 | 0 |
| any_bridge_tunnel | bool | 0.0 | 3.0 | nan | nan | 2 |
| has_motorway_approach | bool | 0.0 | 2.4 | nan | nan | 2 |
| n_signals_30m | numeric | 0.0 | 9.9 | 0.0 | 4.0 | 0 |
| n_stop_signs_30m | numeric | 0.0 | 50.8 | 1.0 | 4.0 | 0 |
| n_crossings_30m | numeric | 0.0 | 31.2 | 0.0 | 8.0 | 0 |
| has_signal | bool | 0.0 | 9.9 | nan | nan | 2 |
| n_bus_stops_50m | numeric | 0.0 | 24.3 | 0.0 | 2.0 | 0 |
| n_pois_100m | numeric | 0.0 | 26.6 | 0.0 | 21.0 | 0 |
| dist_tram_m | numeric | 0.0 | 100.0 | 100000.0 | 100000.0 | 0 |
| streetcar_20m | bool | 0.0 | 0.0 | nan | nan | 1 |
| dist_motorway_m | numeric | 0.0 | 97.9 | 2144.7 | 8579.7 | 0 |
| dist_retail_m | numeric | 0.0 | 99.5 | 758.4 | 2803.3 | 0 |
| dist_nearest_intersection_m | numeric | 0.0 | 100.0 | 72.8 | 232.2 | 0 |

# History features

Built per label year Y from years strictly before Y (see `features/history.py` and the leakage test in `tests/test_history.py`): prev1 and prev3-mean crash and severe counts, the same as shares of the citywide intersection total of those years, years since the last crash (capped at 3), and `covid_in_window` when the window touches 2020-2021. The window is the three most recent years with data before Y; the first label years have fewer behind them (`n_history_years`), and an unpublished year is skipped rather than read as zero.

## Drift check (clean intersections)

| split | rows | mean_label | mean_prev1 | mean_prev3 | mean_prev1_share_x1e4 | covid_window_rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 61,104 | 0.6 | 0.6 | 0.6 | 0.7 | 0.0 |
| val | 15,276 | 0.5 | 0.4 | 0.5 | 0.7 | 1.0 |
| test | 15,276 | 0.6 | 0.5 | 0.4 | 0.7 | 1.0 |

| year | mean_label | mean_prev1 | mean_prev3 |
|---|---:|---:|---:|
| 2016 | 0.6 | 0.6 | 0.6 |
| 2017 | 0.6 | 0.6 | 0.6 |
| 2018 | 0.6 | 0.6 | 0.6 |
| 2019 | 0.7 | 0.6 | 0.6 |
| 2021 | 0.4 | 0.4 | 0.5 |
| 2022 | 0.5 | 0.4 | 0.5 |
| 2024 | 0.6 | 0.5 | 0.4 |
