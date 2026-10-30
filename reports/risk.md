# Risk scoring (CP6: per-vehicle risk, 2026)

Every directed edge carries the expected number of crashes for one vehicle that drives it, stored per million vehicles, so a route's total is expected crashes per million trips. Intersection risk is charged once, on the edge entering the consolidated intersection; segment risk is a rate per million vehicle-km (MVKM) x length. Crash years for rates: 2022-2025.

## Gardiner / DVP attribution

Surface intersections within 150 m of the Gardiner or DVP mainline (not expressway clusters, not CP3 proxies): crashes above the SPF expectation over 2022-2025 move to the expressway; the intersection keeps the rest (its EB is recomputed on the reduced history). CP3 expressway proxies add their excess over the SPF to the nearest expressway (excess at 400-series proxies is dropped: MTO already counts those crashes).

| expressway | surface intersections within 150 m | with excess | reassigned crashes/yr | proxy excess/yr | mid-block crashes/yr | MVKM/yr | rate before attribution | rate after | in MTO range 0.5-1.1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gardiner | 249 | 89 | 1,221.60 | 56.66 | 17.75 | 1,258.67 | 0.06 | 1.03 | yes |
| DVP | 107 | 24 | 95.04 | 456.90 | 2.75 | 907.87 | 0.51 | 0.61 | yes |
| Allen Road | 0 | 0 | 0.00 | 0.00 | 30.00 | 281.56 | 0.11 | 0.11 | no |

Rate top 20 (EB per million entering vehicles, entering AADT >= 5,000): 8 attributed intersections before attribution, 0 after; 5 Lake Shore Blvd intersections dropped out.

Largest reassignments:

| intersection | expressway | dist_m | observed_per_year | spf_expected_per_year | reassigned_per_year |
|---|---|---:|---:|---:|---:|
| Jameson Avenue / Lake Shore Boulevard West | Gardiner | 31.7 | 192.0 | 17.4 | 174.6 |
| Lake Shore Boulevard West / Lower Spadina Avenue / Spadina Avenue | Gardiner | 7.4 | 156.2 | 28.4 | 127.9 |
| Lake Shore Boulevard West / Lower Jarvis Street | Gardiner | 16.0 | 124.2 | 22.8 | 101.5 |
| Jameson Avenue | Gardiner | 30.7 | 91.2 | 1.0 | 90.2 |
| Park Lawn Road | Gardiner | 88.9 | 65.8 | 6.9 | 58.8 |
| Lake Shore Boulevard West / York Street | Gardiner | 4.6 | 78.2 | 19.6 | 58.6 |
| Bay Street / Lake Shore Boulevard West | Gardiner | 6.8 | 90.0 | 37.3 | 52.7 |
| British Columbia Road / Lake Shore Boulevard West | Gardiner | 126.8 | 62.2 | 10.4 | 51.9 |
| Lake Shore Boulevard West / Yonge Street | Gardiner | 5.1 | 79.5 | 33.6 | 45.9 |
| Bathurst Street / Lake Shore Boulevard West | Gardiner | 142.0 | 72.2 | 45.0 | 27.3 |

## City segment rates (mid-block crashes per MVKM)

| rate_group | km | midblock | mvkm | rate_per_mvkm |
|---|---:|---:|---:|---:|
| unclassified | 385.665 | 104.500 | 169.304 | 0.617 |
| residential | 6,791.868 | 517.000 | 1,109.137 | 0.466 |
| primary | 314.822 | 164.250 | 1,180.975 | 0.139 |
| tertiary | 1,509.601 | 192.500 | 1,429.543 | 0.135 |
| secondary | 1,456.883 | 548.250 | 4,645.293 | 0.118 |
| other expressway | 11.353 | 12.750 | 262.079 | 0.049 |
| motorway_link | 227.924 | 22.750 | 665.333 | 0.034 |
| other | 46.509 | 1.500 | 45.698 | 0.033 |

TPS assigns 96% of crashes to intersections, so these mid-block rates are small next to intersection risk; they matter mainly on long links.

## 400-series (MTO section rates, 2015-2019)

| mto_hwy | mvkm | MTO collisions/yr at our AADT | rate_per_mvkm |
|---|---:|---:|---:|
| 400 | 608.962 | 526.852 | 0.865 |
| 401 | 5,460.609 | 5,634.059 | 1.032 |
| 404 | 548.348 | 526.492 | 0.960 |
| 409 | 150.156 | 119.309 | 0.795 |
| 427 | 1,670.236 | 807.083 | 0.483 |
| QEW | 47.249 | 48.247 | 1.021 |

## Risk per trip-km by road class

Expected crashes per million vehicles per km driven on that class, with the intersections a vehicle enters along the way. This is the ordering the router sees.

| road class | km | intersection risk per km | segment risk per km | total per km | intersections per km |
|---|---:|---:|---:|---:|---:|
| primary | 314.822 | 6.128 | 0.139 | 6.267 | 7.388 |
| living_street | 7.226 | 5.876 | 0.033 | 5.909 | 8.580 |
| secondary | 1,456.883 | 4.994 | 0.118 | 5.112 | 6.959 |
| residential | 6,791.868 | 4.168 | 0.466 | 4.634 | 6.006 |
| tertiary | 1,509.601 | 4.316 | 0.135 | 4.450 | 6.666 |
| unclassified | 385.665 | 3.712 | 0.617 | 4.329 | 3.067 |
| secondary_link | 16.039 | 2.757 | 0.033 | 2.789 | 11.035 |
| primary_link | 10.459 | 2.332 | 0.033 | 2.365 | 8.796 |
| 400-series | 237.169 | 0.027 | 0.889 | 0.916 | 1.320 |
| motorway | 91.854 | 0.089 | 0.671 | 0.760 | 1.459 |
| busway | 10.604 | 0.429 | 0.033 | 0.462 | 2.263 |
| motorway_link | 227.924 | 0.162 | 0.034 | 0.196 | 2.584 |

## Intersections

- 22,435 intersections; EB 2026 expected crashes/yr total 51,484 (median 0.40). 31 proxy intersections use the SPF alone.
- Crashes per million entering vehicles: median 0.33, p90 1.49, p99 4.42.

Highest per-vehicle risk (entering AADT >= 5,000), after attribution:

| intersection | risk | entering_aadt | risk_per_mev | attributed_share |
|---|---:|---:|---:|---:|
| Cherry Street / Polson Street | 31.57 | 6,220.78 | 13.91 | 0.00 |
| Grandstand Place / Milepost Place / Thorncliffe Park Drive | 52.20 | 11,380.57 | 12.57 | 0.00 |
| Canadian Road / Warden Avenue | 138.88 | 40,429.81 | 9.41 | 0.00 |
| East Liberty Street / Lynn Williams Street | 23.76 | 7,672.74 | 8.48 | 0.00 |
| Ashtonbee Road / Thermos Road | 22.87 | 8,001.58 | 7.83 | 0.00 |
| Harness Road / Queen's Plate Drive | 17.81 | 6,428.05 | 7.59 | 0.00 |
| Covington Road / Saranac Boulevard | 22.67 | 8,218.71 | 7.56 | 0.00 |
| Dufferin Street / Gerry Fitzgerald Drive | 98.66 | 38,134.02 | 7.09 | 0.00 |
| Allanford Road / Sheppard Avenue East | 60.94 | 23,577.93 | 7.08 | 0.00 |
| Keele Street / St. Clair Avenue West / Weston Road | 105.78 | 43,351.83 | 6.68 | 0.00 |
| Eglinton Avenue East / Eglinton Avenue West / Yonge Street | 69.15 | 28,736.37 | 6.59 | 0.00 |
| Fenmar Drive / Kenhar Drive | 26.41 | 11,746.31 | 6.16 | 0.00 |
| Ellesmere Road / Markham Road | 129.95 | 59,430.37 | 5.99 | 0.00 |
| Ellesmere Road / Kennedy Road | 171.34 | 78,552.71 | 5.98 | 0.00 |
| Kennedy Road / Lawrence Avenue East | 127.07 | 58,547.01 | 5.95 | 0.00 |
| Danforth Avenue / Victoria Park Avenue | 85.48 | 40,233.35 | 5.82 | 0.00 |
| Gerrard Street East / Marjory Avenue | 35.99 | 17,061.19 | 5.78 | 0.00 |
| Lloyd Manor Road / Warrender Avenue | 10.31 | 5,137.79 | 5.50 | 0.00 |
| Midland Avenue / Silver Star Boulevard / South Shields Avenue | 49.05 | 24,616.48 | 5.46 | 0.00 |
| Bathurst Street / Dundas Street West | 53.44 | 27,294.73 | 5.36 | 0.00 |
