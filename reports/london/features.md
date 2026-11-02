# Static intersection features

5,637 intersections x 27 features, from the current OSM snapshot (so they describe today's roads, not the roads in each label year). No raw coordinates or neighbourhood IDs. Missing maxspeed / lanes on an approach are imputed with the median for its road class; `maxspeed_missing` / `lanes_missing` mark intersections where no approach carried the tag. Unhandled NaNs: 0.

| feature | type | % NaN | % non-zero / true | median | p99 | levels |
|---|---|---:|---:|---:|---:|---:|
| n_osm_nodes | numeric | 0.0 | 100.0 | 1.0 | 4.0 | 0 |
| street_count | numeric | 0.0 | 100.0 | 3.0 | 5.0 | 0 |
| max_class_rank | numeric | 0.0 | 100.0 | 2.0 | 6.0 | 0 |
| min_class_rank | numeric | 0.0 | 100.0 | 1.0 | 5.0 | 0 |
| class_combo | category | 0.0 | nan | nan | nan | 10 |
| n_legs | numeric | 0.0 | 100.0 | 3.0 | 4.0 | 0 |
| min_leg_angle_deg | numeric | 0.0 | 100.0 | 88.8 | 180.0 | 0 |
| n_approaches | numeric | 0.0 | 100.0 | 3.0 | 6.0 | 0 |
| max_maxspeed_kph | numeric | 0.0 | 100.0 | 40.0 | 100.0 | 0 |
| min_maxspeed_kph | numeric | 0.0 | 100.0 | 40.0 | 96.4 | 0 |
| maxspeed_missing | bool | 0.0 | 59.3 | nan | nan | 2 |
| max_lanes | numeric | 0.0 | 100.0 | 2.0 | 5.0 | 0 |
| lanes_missing | bool | 0.0 | 8.4 | nan | nan | 2 |
| n_oneway_approaches | numeric | 0.0 | 18.1 | 0.0 | 4.0 | 0 |
| any_bridge_tunnel | bool | 0.0 | 3.7 | nan | nan | 2 |
| has_motorway_approach | bool | 0.0 | 2.0 | nan | nan | 2 |
| n_signals_30m | numeric | 0.0 | 6.9 | 0.0 | 2.0 | 0 |
| n_stop_signs_30m | numeric | 0.0 | 6.8 | 0.0 | 1.0 | 0 |
| n_crossings_30m | numeric | 0.0 | 71.3 | 1.0 | 7.0 | 0 |
| has_signal | bool | 0.0 | 6.9 | nan | nan | 2 |
| n_bus_stops_50m | numeric | 0.0 | 7.7 | 0.0 | 2.0 | 0 |
| n_pois_100m | numeric | 0.0 | 17.6 | 0.0 | 11.0 | 0 |
| dist_tram_m | numeric | 0.0 | 100.0 | 100000.0 | 100000.0 | 0 |
| streetcar_20m | bool | 0.0 | 0.0 | nan | nan | 1 |
| dist_motorway_m | numeric | 0.0 | 98.3 | 4932.1 | 12280.1 | 0 |
| dist_retail_m | numeric | 0.0 | 99.9 | 756.9 | 5453.8 | 0 |
| dist_nearest_intersection_m | numeric | 0.0 | 100.0 | 87.1 | 442.1 | 0 |
