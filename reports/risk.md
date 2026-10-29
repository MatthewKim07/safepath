# Risk scoring (2026)

Units are expected police-reported crashes per year at a location. There is no traffic-volume term yet, so busy roads look risky partly because they are busy.

## Intersection risk

- 22,435 intersections; lgbm_main prediction for 2026 from 2023-2025 history. 31 proxy intersections use the static-only deploy model instead.
- Total 57,515 crashes/yr; median 0.42, p99 37.4, max 179.3.

Proxy fallbacks (own 2023-2025 mean vs the value used):

| intersection | exclusion_reason | prev3_crashes_mean | pred_main | risk |
|---|---|---:|---:|---:|
| Alcester Court / Highland Hill | retail_proxy | 223.00 | 117.29 | 0.72 |
| Borough Drive / Progress Avenue | retail_proxy | 186.00 | 142.70 | 3.39 |
| Airley Crescent / Four Oaks Gate | expressway_proxy | 182.00 | 125.07 | 0.99 |
| Parkway Forest Drive / Sheppard Avenue East | retail_proxy | 166.67 | 130.81 | 6.49 |
| Laurentide Drive / York Mills Road | expressway_proxy | 138.67 | 135.82 | 2.32 |
| Ferrand Drive / Seton Park Road | expressway_proxy | 112.67 | 78.79 | 0.64 |
| Sherway Gardens Road / Sherway Gate | retail_proxy | 89.33 | 65.41 | 11.07 |
| Eglinton Avenue East / Hakimi Avenue / Lebovic Avenue | retail_proxy | 66.33 | 61.62 | 40.55 |
| Daphne Road / Greencedar Circuit | retail_proxy | 65.67 | 53.44 | 6.22 |
| Antrim Crescent / Kennedy Road | expressway_proxy | 63.33 | 63.22 | 19.98 |
| Concorde Place / Wynford Drive / Wynford Heights Crescent | expressway_proxy | 59.00 | 61.83 | 9.90 |
| Brimwood Boulevard / Sandhurst Circle | retail_proxy | 57.67 | 49.64 | 4.87 |
| Abitibi Avenue / Yonge Street | retail_proxy | 57.33 | 41.72 | 6.86 |
| Archer Street / Cobourg Avenue | retail_proxy | 52.00 | 37.82 | 0.77 |
| Karl Fraser Road / Lawrence Avenue East | retail_proxy | 48.33 | 39.48 | 3.77 |
| Blossomfield Drive / Dane Avenue | retail_proxy | 47.67 | 43.27 | 6.93 |
| Scarsdale Road / York Mills Road | retail_proxy | 46.67 | 45.76 | 37.09 |
| Esther Shiner Boulevard / Provost Drive | retail_proxy | 43.67 | 49.51 | 8.20 |
| Beverly Glen Boulevard / Bridletowne Circle | retail_proxy | 43.67 | 37.88 | 10.89 |
| Queen Elizabeth Boulevard / Taymall Avenue | expressway_proxy | 41.33 | 37.68 | 0.67 |
| Don Mills Road / Skymark Drive | retail_proxy | 38.67 | 34.96 | 5.63 |
| Finch Avenue West / San Romanoway | retail_proxy | 38.00 | 32.58 | 5.02 |
| Elkhorn Drive / Hawksbury Drive | retail_proxy | 36.33 | 36.61 | 1.99 |
| Index Road / North Queen Street / Service Road | retail_proxy | 35.33 | 38.45 | 7.61 |
| Jane Street / San Romanoway | retail_proxy | 28.33 | 24.60 | 2.53 |
| Don Mills Road East | retail_proxy | 26.00 | 27.62 | 0.91 |
| Highway 401 Express | expressway_proxy | 24.67 | 18.69 | 0.37 |
| Lynnford Drive / The East Mall | retail_proxy | 22.00 | 17.79 | 3.62 |
| Blackwell Avenue / Tapscott Road | retail_proxy | 20.67 | 18.89 | 14.13 |
| Bamburgh Circle / Wintermute Boulevard | retail_proxy | 19.33 | 18.01 | 10.13 |
| Eastwick Road / Oxford Street | expressway_proxy | 18.33 | 17.01 | 3.18 |

## Segment rates (crashes per km per year, train label years)

| highway | km | train_midblock_crashes | crashes_per_km_year | source |
|---|---:|---:|---:|---|
| motorway | 328.169 | 3336.000 | 1.154 | expressway-proxy excess |
| motorway_link | 227.854 | 134.000 | 1.154 | expressway-proxy excess |
| secondary | 900.117 | 3665.000 | 0.679 | mid-block assignments |
| primary | 229.340 | 820.000 | 0.596 | mid-block assignments |
| unclassified | 201.220 | 553.000 | 0.458 | mid-block assignments |
| living_street | 3.761 | 6.000 | 0.266 | mid-block assignments |
| tertiary | 792.916 | 1107.000 | 0.233 | mid-block assignments |
| residential | 3551.545 | 2943.000 | 0.138 | mid-block assignments |
| busway | 6.023 | 0.000 | 0.000 | mid-block assignments |
| escape | 0.037 | 0.000 | 0.000 | mid-block assignments |
| primary_link | 10.317 | 0.000 | 0.000 | mid-block assignments |
| secondary_link | 15.146 | 0.000 | 0.000 | mid-block assignments |
| tertiary_link | 1.868 | 0.000 | 0.000 | mid-block assignments |

Motorway rate: 8 expressway-proxy intersections average 680 crashes/yr in train years against 38.0 expected from static features, an excess of 642/yr spread over 556 km of motorway carriageway = **1.154 crashes/km/yr**. For comparison, mid-block crashes TPS placed directly on motorway segments give 1.040/km/yr; the two sources are separate crashes, so the rate used is a lower bound on motorway crashes that TPS located.

## Total risk by source

| source | crashes/yr | share |
|---|---:|---:|
| intersections | 57,515 | 96.4% |
| non-motorway segments | 1,516 | 2.5% |
| motorways | 642 | 1.1% |

Segment totals are rate x km over undirected segments (a two-way street counts once; each motorway carriageway is its own segment).

## Risk per directed km by road class (sanity check: motorways are not free)

| highway | km | risk | seg | risk_per_km | segment_risk_per_km |
|---|---:|---:|---:|---:|---:|
| primary | 314.822 | 23934.708 | 187.606 | 76.026 | 0.596 |
| secondary | 1456.883 | 90543.525 | 988.664 | 62.149 | 0.679 |
| primary_link | 10.459 | 456.207 | 0.000 | 43.617 | 0.000 |
| secondary_link | 16.039 | 535.325 | 0.000 | 33.376 | 0.000 |
| tertiary | 1509.601 | 35425.304 | 351.262 | 23.467 | 0.233 |
| unclassified | 385.665 | 5123.652 | 176.649 | 13.285 | 0.458 |
| living_street | 7.226 | 81.381 | 1.921 | 11.262 | 0.266 |
| residential | 6791.868 | 44069.486 | 938.017 | 6.489 | 0.138 |
| tertiary_link | 2.107 | 12.320 | 0.000 | 5.848 | 0.000 |
| motorway_link | 227.924 | 1243.498 | 263.081 | 5.456 | 1.154 |
| motorway | 329.023 | 637.789 | 379.774 | 1.938 | 1.154 |
| busway | 10.604 | 13.916 | 0.000 | 1.312 | 0.000 |
| escape | 0.073 | 0.010 | 0.000 | 0.136 | 0.000 |

## Highest-risk intersections

| intersection | risk | prev3_crashes_mean | is_proxy_fallback |
|---|---:|---:|---:|
| Jameson Avenue / Lake Shore Boulevard West | 179.3 | 226.7 | 0 |
| Ellesmere Road / Markham Road | 164.3 | 132.0 | 0 |
| Kennedy Road / Lawrence Avenue East | 163.4 | 141.3 | 0 |
| Ellesmere Road / Kennedy Road | 158.1 | 189.0 | 0 |
| Lake Shore Boulevard West / Lower Spadina Avenue / Spadina Avenue | 154.5 | 158.3 | 0 |
| Dufferin Street / Finch Avenue West | 131.4 | 144.3 | 0 |
| Kennedy Road / Sheppard Avenue East | 119.4 | 106.3 | 0 |
| Canadian Road / Warden Avenue | 117.3 | 135.7 | 0 |
| Lawrence Avenue East / Victoria Park Avenue | 112.1 | 96.7 | 0 |
| Bayview Avenue / Sheppard Avenue East | 107.0 | 96.7 | 0 |
| Lawrence Avenue East / Midland Avenue | 106.2 | 107.0 | 0 |
| Ellesmere Road / Warden Avenue | 105.6 | 111.7 | 0 |
| Lake Shore Boulevard West / Lower Jarvis Street | 105.5 | 118.0 | 0 |
| Dufferin Street / Gerry Fitzgerald Drive | 103.6 | 122.0 | 0 |
| Evans Avenue / Kipling Avenue | 100.8 | 96.7 | 0 |
