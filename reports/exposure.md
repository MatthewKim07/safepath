# Exposure sanity checks (CP5)

What happens to TPS crash counts when they are divided by traffic. Crashes: TPS, label years 2016-2019 and 2022-2025 (2020-21 left out), per year. Volumes: reports/volumes.md (counts), reports/volume_model.md (model), reports/mto.md (400-series). Proxy intersections (CP3) are excluded from sections 1-3.

## 1. Crashes per million entering vehicles, counted intersections

Intersections with a City count from 2010 or later (count-based AADT, no model). MEV = entering AADT x 365 / 1e6. The last column repeats the pooled rate over every intersection using the final (blended / modelled) AADT.

| class combo | counted | crashes/yr | MEV/yr | pooled crashes per MEV | median per MEV | median crashes/yr | pooled per MEV, all (modelled AADT) |
|---|---:|---:|---:|---:|---:|---:|---:|
| arterial x arterial | 346 | 13,503.125 | 5,214.004 | 2.590 | 2.562 | 32.750 | 0.700 |
| highway x collector | 24 | 506.250 | 319.677 | 1.584 | 0.887 | 12.875 | 1.431 |
| arterial x collector | 629 | 9,294.250 | 6,655.441 | 1.396 | 1.220 | 11.750 | 1.316 |
| collector x collector | 179 | 1,070.750 | 820.872 | 1.304 | 1.081 | 4.250 | 0.656 |
| all | 4,845 | 41,738.625 | 35,240.443 | 1.184 | 0.742 | 3.500 | 0.637 |
| local x local | 798 | 914.000 | 862.407 | 1.060 | 0.817 | 0.625 | 0.728 |
| arterial x local | 1,925 | 13,488.875 | 16,886.116 | 0.799 | 0.599 | 4.750 | 0.592 |
| highway x local | 8 | 73.500 | 92.896 | 0.791 | 0.225 | 1.312 | 0.683 |
| collector x local | 858 | 2,310.500 | 3,193.570 | 0.723 | 0.533 | 1.750 | 0.526 |
| highway x arterial | 77 | 577.375 | 1,180.947 | 0.489 | 0.000 | 0.000 | 0.212 |
| highway x highway | 1 | 0.000 | 14.514 | 0.000 | 0.000 | 0.000 | 0.019 |

## 2. Does dividing by volume reorder the worst intersections?

Rate ranking among counted intersections with entering AADT >= 5,000 and >= 5 crashes/yr (1,998 intersections). **8 of the top 20** by count are also in the top 20 by rate; Spearman between crash count and crash rate among eligible intersections: 0.75.

Top 20 by crashes per year:

| intersection | class combo | crashes/yr | entering AADT | count year | per MEV | rank by count | rank by rate |
|---|---|---:|---:|---|---:|---:|---|
| Lake Shore Boulevard West / Lower Spadina Avenue / Spadina Avenue | highway x arterial | 173.50 | 57,425.26 | 2025 | 8.28 | 1 | 4 |
| Ellesmere Road / Kennedy Road | arterial x arterial | 159.38 | 80,544.61 | 2025 | 5.42 | 2 | 34 |
| Jameson Avenue / Lake Shore Boulevard West | highway x collector | 156.88 | 26,200.01 | 2017 | 16.40 | 3 | 2 |
| Dufferin Street / Finch Avenue West | arterial x arterial | 147.25 | 76,455.50 | 2023 | 5.28 | 4 | 40 |
| Ellesmere Road / Markham Road | arterial x arterial | 143.88 | 60,280.73 | 2022 | 6.54 | 5 | 12 |
| Kennedy Road / Lawrence Avenue East | arterial x arterial | 122.88 | 59,301.32 | 2026 | 5.68 | 6 | 22 |
| Finch Avenue West / Jane Street | arterial x arterial | 119.88 | 57,247.84 | 2025 | 5.74 | 7 | 21 |
| Kennedy Road / Sheppard Avenue East | arterial x arterial | 119.12 | 67,088.01 | 2025 | 4.86 | 8 | 56 |
| Canadian Road / Warden Avenue | arterial x local | 118.50 | 41,203.12 | 2024 | 7.88 | 9 | 7 |
| Bayview Avenue / Sheppard Avenue East | arterial x arterial | 117.88 | 82,348.74 | 2024 | 3.92 | 10 | 120 |
| Kipling Avenue / Rexdale Boulevard | arterial x arterial | 113.38 | 56,762.79 | 2022 | 5.47 | 11 | 30 |
| Evans Avenue / Kipling Avenue | arterial x arterial | 111.62 | 47,715.02 | 2025 | 6.41 | 12 | 15 |
| Ellesmere Road / Warden Avenue | arterial x arterial | 108.62 | 80,536.13 | 2022 | 3.70 | 13 | 148 |
| Markham Road / Progress Avenue | arterial x arterial | 108.38 | 67,613.42 | 2023 | 4.39 | 14 | 79 |
| Lake Shore Boulevard West / Lower Jarvis Street | highway x arterial | 107.00 | 38,924.53 | 2024 | 7.53 | 15 | 8 |
| Keele Street / St. Clair Avenue West / Weston Road | arterial x arterial | 104.12 | 43,867.16 | 2025 | 6.50 | 16 | 13 |
| Finch Avenue / Finch Avenue East / Finch Avenue West / Yonge Street | arterial x arterial | 102.88 | 48,644.92 | 2026 | 5.79 | 17 | 20 |
| Finch Avenue West / Weston Road | arterial x arterial | 100.88 | 50,837.18 | 2025 | 5.44 | 18 | 32 |
| Jane Street / Wilson Avenue | arterial x arterial | 100.75 | 55,189.46 | 2025 | 5.00 | 19 | 48 |
| Eglinton Avenue East / Kennedy Road | arterial x arterial | 99.88 | 48,869.69 | 2025 | 5.60 | 20 | 24 |

Top 20 by crashes per million entering vehicles:

| intersection | class combo | crashes/yr | entering AADT | count year | per MEV | rank by count | rank by rate |
|---|---|---:|---:|---|---:|---:|---|
| Cherry Street / Polson Street | collector x local | 37.12 | 6,173.08 | 2026 | 16.48 | 221 | 1 |
| Jameson Avenue / Lake Shore Boulevard West | highway x collector | 156.88 | 26,200.01 | 2017 | 16.40 | 3 | 2 |
| Bay Street / Lake Shore Boulevard West | arterial x arterial | 94.25 | 29,821.86 | 2012 | 8.66 | 25 | 3 |
| Lake Shore Boulevard West / Lower Spadina Avenue / Spadina Avenue | highway x arterial | 173.50 | 57,425.26 | 2025 | 8.28 | 1 | 4 |
| East Liberty Street / Lynn Williams Street | collector x local | 22.25 | 7,649.89 | 2022 | 7.97 | 463 | 5 |
| Allanford Road / Sheppard Avenue East | arterial x local | 67.75 | 23,515.47 | 2025 | 7.89 | 68 | 6 |
| Canadian Road / Warden Avenue | arterial x local | 118.50 | 41,203.12 | 2024 | 7.88 | 9 | 7 |
| Lake Shore Boulevard West / Lower Jarvis Street | highway x arterial | 107.00 | 38,924.53 | 2024 | 7.53 | 15 | 8 |
| Dunn Avenue / Springhurst Avenue | arterial x local | 24.12 | 8,797.26 | 2026 | 7.51 | 419 | 9 |
| Eglinton Avenue East / Eglinton Avenue West / Yonge Street | arterial x arterial | 76.25 | 28,098.49 | 2026 | 7.43 | 48 | 10 |
| Lake Shore Boulevard West / Yonge Street | arterial x arterial | 76.50 | 30,121.49 | 2012 | 6.96 | 47 | 11 |
| Ellesmere Road / Markham Road | arterial x arterial | 143.88 | 60,280.73 | 2022 | 6.54 | 5 | 12 |
| Keele Street / St. Clair Avenue West / Weston Road | arterial x arterial | 104.12 | 43,867.16 | 2025 | 6.50 | 16 | 13 |
| Gerrard Street East / Marjory Avenue | arterial x local | 39.00 | 16,510.18 | 2017 | 6.47 | 211 | 14 |
| Evans Avenue / Kipling Avenue | arterial x arterial | 111.62 | 47,715.02 | 2025 | 6.41 | 12 | 15 |
| Lawrence Avenue East | arterial x local | 28.12 | 12,139.53 | 2022 | 6.35 | 347 | 16 |
| Harness Road / Queen's Plate Drive | collector x local | 14.25 | 6,177.23 | 2024 | 6.32 | 764 | 17 |
| Bathurst Street / Dundas Street West | arterial x arterial | 61.62 | 27,349.71 | 2026 | 6.17 | 86 | 18 |
| Bathurst Street / King Street West | arterial x arterial | 46.00 | 21,613.58 | 2026 | 5.83 | 154 | 19 |
| Finch Avenue / Finch Avenue East / Finch Avenue West / Yonge Street | arterial x arterial | 102.88 | 48,644.92 | 2026 | 5.79 | 17 | 20 |

## 3. Crashes per million vehicle-km by road class (whole network)

Each clean intersection's crashes are split across the edges entering it in proportion to their directional AADT; mid-block crashes stay on their segment. Vehicle-km use the final AADT on every directed edge (counts, model, MTO, lane-scaled). The per-km column is the CP4 view (no exposure) for contrast.

| road class | km (segments) | MVKM/yr | TPS crashes/yr | of which at intersections | crashes/km/yr (CP4 view) | crashes per MVKM |
|---|---:|---:|---:|---:|---:|---:|
| primary | 229.34 | 1,180.97 | 9,139.76 | 8,991.89 | 39.85 | 7.74 |
| secondary | 900.12 | 4,645.29 | 30,376.38 | 29,774.63 | 33.75 | 6.54 |
| residential | 3,551.55 | 1,109.14 | 7,072.24 | 6,569.61 | 1.99 | 6.38 |
| tertiary | 792.92 | 1,429.54 | 8,598.96 | 8,409.71 | 10.84 | 6.02 |
| other | 37.15 | 45.70 | 245.20 | 243.95 | 6.60 | 5.37 |
| unclassified | 201.22 | 169.30 | 828.32 | 731.20 | 4.12 | 4.89 |
| all | 6,268.31 | 20,441.02 | 57,406.75 | 55,320.00 | 9.16 | 2.81 |
| expressway ramp | 227.85 | 665.33 | 418.10 | 394.60 | 1.83 | 0.63 |
| expressway (city) | 91.00 | 2,710.18 | 187.94 | 117.82 | 2.07 | 0.07 |
| 400-series (OPP) | 237.17 | 8,485.56 | 539.84 | 86.59 | 2.28 | 0.06 |

## 4. Expressways: MTO vs TPS

TPS crashes on an expressway = mid-block crashes on its segments, plus intersection crashes charged to its edges at ramp-merge clusters, plus the excess (actual - SPF-expected) at expressway-proxy intersections attributed to the nearest expressway. MTO rates are reportable collisions per MVKM from MTO's own volumes (reports/mto.md). DVP and Gardiner vehicle-km use the lane-scaled AADT; the last column shows the rate if the uncorrected model AADT were used instead.

| expressway | carriageway km | MVKM/yr (our AADT) | TPS crashes/yr (mid-block + ramp merges + proxy excess) | TPS per MVKM | MTO collisions/yr (2015-19) | MTO per MVKM (2015-19) | TPS per MVKM if model AADT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 401 | 155.897 | 5,460.609 | 461.355 | 0.084 | 5,316.400 | 1.023 | nan |
| 400 | 16.226 | 608.962 | 34.784 | 0.057 | 497.800 | 0.857 | nan |
| 404 | 11.138 | 548.348 | 37.666 | 0.069 | 440.200 | 0.916 | nan |
| 409 | 8.278 | 150.156 | 6.750 | 0.045 | 85.000 | 0.750 | nan |
| 427 | 44.222 | 1,670.236 | 81.398 | 0.049 | 726.800 | 0.472 | nan |
| QEW | 1.407 | 47.249 | 3.203 | 0.068 | 41.200 | 1.021 | nan |
| DVP | 29.959 | 907.874 | 527.616 | 0.581 | nan | nan | 0.374 |
| Gardiner | 40.515 | 1,258.668 | 77.578 | 0.062 | nan | nan | 0.038 |
| Allen Road | 10.027 | 281.558 | 77.172 | 0.274 | nan | nan | 0.183 |

The Gardiner's TPS rate is implausibly low. 74 Lake Shore Blvd intersections lie within 150 m of it and carry 1,312 crashes/yr; 5 of them are in the top 20 by rate above. TPS appears to place many Gardiner crashes at the surface intersections beneath it, as it places DVP crashes at side streets, but the CP3 expressway-proxy rule (residential corners far above their class median) does not catch arterial intersections. If every one of those crashes were a Gardiner crash the rate would be 1.10 per MVKM; the truth lies between the two.

## What this says

- **Residential vs arterial.** Per kilometre (CP4 view) primary roads carry 40 crashes/km/yr against 2.0 on residential streets. Per vehicle-km the gap shrinks to 7.74 vs 6.38 crashes per MVKM (1.2x instead of 20x). Local x local intersections have 1.06 crashes per MEV vs 2.59 at arterial x arterial ones. A residential cut-through is therefore not the near-free option count-based risk made it look like.
- **Freeways.** MTO's 400-series rate in Toronto is 0.47-1.02 collisions per MVKM, while TPS records 0.084 per MVKM on the 401, i.e. about 8% of the collisions MTO counts. With MTO's rates the 401 is 8x safer per vehicle-km than primary arterials (7.74 per MVKM); CP4's per-km TPS risk put the gap at ~40x (reports/risk.md, risk per directed km 76 vs 1.9). Freeways are safer per kilometre driven, but not nearly free.
- **DVP** (TPS jurisdiction): 0.58 crashes per MVKM with lane-scaled AADT (0.37 with the model's AADT), the same order as MTO's 400-series rates, mostly through the expressway-proxy excess. **Gardiner**: 0.06 as recorded, up to 1.10 if its likely crashes at Lake Shore Blvd are counted (section 4). CP6 should attribute those, or give the city expressways an MTO-like rate.
- **Hotspot lists.** Normalising keeps 8 of the top 20; the rate list adds lower-volume intersections with many crashes for their traffic. Rates at low-volume intersections are noisy, which is what the Empirical Bayes step in CP6 is for.
