# Ottawa graph, location precision, snapping and labels

## Study area

City of Ottawa Official Plan urban boundary (https://maps.ottawa.ca/arcgis/rest/services/Planning/MapServer/13), 643 km² (Toronto: 630 km²). The amalgamated city is 2,790 km², mostly rural townships and villages with no counterpart in Toronto. 13,019 non-freeway crashes fall outside the boundary and are out of scope; 122,623 are inside.

OSM drive graph: 19,013 nodes, 49,379 directed edges, 28,535 undirected segments. Consolidation at 10 m (as Toronto): 15,276 intersections.

| tolerance_m | merge_distance_m | intersections | merged_clusters | pct_crashes_within_5m | pct_crashes_within_10m | pct_crashes_within_20m | median_dist_m | clusters_with_2plus_tps_points | pct_crashes_in_those_clusters |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20 | 15,276 | 1,313 | 42.9 | 57.8 | 69.0 | 7.1 | 4,296 | 66.6 |
| 15 | 30 | 14,474 | 1,480 | 35.1 | 49.4 | 65.4 | 10.2 | 4,067 | 63.2 |
| 20 | 40 | 13,515 | 1,757 | 31.0 | 43.6 | 61.6 | 13.6 | 3,715 | 59.4 |

## Location precision

| measure | value |
|---|---:|
| crashes | 122623.0 |
| distinct crash coordinates | 86371.0 |
| crashes per distinct point (mean) | 1.4 |
| distinct City location ids (Geo_ID) | 22148.0 |
| % of Geo_IDs with a single coordinate | 49.4 |
| % crashes on points with >= 10 crashes | 24.7 |

Where crash points sit (distinct points in the study area, crash-weighted):

| category | crashes | % crashes | points | % points |
|---|---:|---:|---:|---:|
| within 5 m of an intersection | 52,591 | 42.9 | 27,767 | 32.1 |
| within 10 m of an intersection | 70,818 | 57.8 | 38,872 | 45.0 |
| within 20 m of an intersection | 84,586 | 69.0 | 49,465 | 57.3 |
| within 30 m of an intersection | 91,355 | 74.5 | 56,073 | 64.9 |
| within 50 m of an intersection | 100,648 | 82.1 | 65,180 | 75.5 |
| 10-30 m from an intersection | 20,537 | 16.7 | 17,201 | 19.9 |
| mid-block: <= 20 m from an edge, > 30 m from any intersection | 30,234 | 24.7 | 29,371 | 34.0 |
| off-network: > 50 m from any drivable edge | 726 | 0.6 | 624 | 0.7 |
| inside an OSM parking polygon | 77 | 0.1 | 74 | 0.1 |
| within 25 m of a motorway / ramp | 7,336 | 6.0 | 5,176 | 6.0 |
| within 25 m of a motorway and > 30 m from any intersection | 2,145 | 1.7 | 2,049 | 2.4 |

Distance to the nearest consolidated intersection by the City's location string ('A @ B' = at, 'A btwn B & C' = btwn), % of crashes per row:

| location_kind | <=5 m | 5-10 | 10-30 | 30-50 | > 50 m | crashes | median m |
|---|---:|---:|---:|---:|---:|---:|---:|
| at | 67.4 | 20.0 | 9.5 | 0.6 | 2.6 | 75,296 | 3.0 |
| btwn | 3.9 | 6.5 | 28.4 | 18.8 | 42.5 | 47,140 | 40.5 |
| other | 28.3 | 28.3 | 20.9 | 10.2 | 12.3 | 187 | 8.6 |

By the City's own intersection field (2013-2016 layers only; the 2017-2024 layer dropped it):

| accident_location | <=5 m | 5-10 | 10-30 | 30-50 | > 50 m | crashes | median m |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 - Non intersection | 2.2 | 6.5 | 32.4 | 19.0 | 39.9 | 14,786 | 37.5 |
| 02 - Intersection related | 60.4 | 24.6 | 11.2 | 0.8 | 3.1 | 16,887 | 3.8 |
| 03 - At intersection | 65.6 | 19.9 | 11.0 | 0.5 | 3.1 | 11,797 | 3.3 |
| 04 - At/near private drive | 3.0 | 6.2 | 30.2 | 19.0 | 41.6 | 5,004 | 39.8 |
| 05 - At railway crossing | 0.0 | 0.0 | 0.0 | 5.0 | 95.0 | 40 | 189.2 |
| 06 - Underpass or tunnel | 0.0 | 8.3 | 20.8 | 20.8 | 50.0 | 24 | 58.4 |
| 07 - Overpass or bridge | 0.8 | 1.2 | 27.5 | 16.2 | 54.2 | 240 | 56.4 |

## Freeway cross-check (OSM)

Of the 13,037 removed freeway crashes inside the study area, 11,198 sit within 30 m of an OSM Hwy 416 / 417 motorway or ramp edge (median 2 m). Kept crashes within 30 m of a 416 / 417 edge and farther than that from any surface road: 45 of 122,623.

## Snapping

Same rule order and thresholds as Toronto: parking polygon -> excluded; <= 30 m from an intersection -> intersection; <= 20 m from an edge -> segment; else excluded. Crash counts by the City's location kind:

| location_kind | excluded | intersection | segment | All |
|---|---:|---:|---:|---:|
| at | 219 | 72,930 | 2,147 | 75,296 |
| btwn | 862 | 18,270 | 28,008 | 47,140 |
| other | 13 | 145 | 29 | 187 |
| All | 1,094 | 91,345 | 30,184 | 122,623 |

## Proxy rules

- Retail: cannot fire (Ottawa publishes no fail-to-remain flag). Busy (>= 150 train crashes per 6 years, scaled to 4), severe share <= 8% and within 150 m of a mall / large retail area: 1 intersections.

| intersection_id | street_names | train_crashes | train_severe_share | dist_retail | dist_motorway | class_combo | ratio_to_combo_median | % roundabout |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 12,389 | Jeanne-d'Arc Boulevard South / St. Joseph Boulevard | 209 | 0.0 | 36.2 | 404.3 | arterial x arterial | 104.5 | 95.0 |

A candidate whose crashes are mostly coded '11 - Roundabout' is a real roundabout (many low-severity sideswipes at the junction itself, recorded as 'A @ B' at many distinct points), not parking-lot crashes moved to a public intersection, so it is kept. No retail exclusions were needed.

- Expressway: 0 intersections meet Toronto's rule.



## Labels

| split | years | rows (clean) | % zero-crash | mean crashes | max crashes | mean severe | % any severe | crashes (clean) | crashes (excluded ints) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| history | 2013, 2014, 2015, 2020, 2021 | 76,380 | 83.9 | 0.5 | 60 | 0.1 | 6.0 | 38,437 | 0 |
| train | 2016, 2017, 2018, 2019 | 61,104 | 81.7 | 0.6 | 59 | 0.1 | 6.8 | 36,762 | 0 |
| val | 2022 | 15,276 | 84.7 | 0.5 | 44 | 0.1 | 5.3 | 7,391 | 0 |
| test | 2024 | 15,276 | 83.2 | 0.6 | 74 | 0.0 | 3.3 | 8,755 | 0 |
