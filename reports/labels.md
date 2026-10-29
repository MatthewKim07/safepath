# Labels and proxy exclusions

Intersection-year table: 22,435 consolidated intersections x 12 years (2014-2025) = 269,220 rows, 60.6% of them zero-crash.

## Proxy rules (train label years only: 2016, 2017, 2018, 2019, 2022, 2023)

- **Retail proxy**: >= 150 train crashes AND severe share <= 8% AND fail-to-remain share >= 25% AND within 150 m of an OSM shop=mall or a landuse=retail polygon >= 2 ha. (Busy intersections citywide have a median severe share of ~14% and FTR share of ~13%.)
- **Expressway proxy**: within 200 m of a motorway / motorway_link AND >= 150 train crashes AND >= 20x the median train crashes of intersections with the same approach-class combo (median over intersections with >= 1 train crash).
- When both fire, the reason is `retail_proxy` (the more specific signature). Both fired for 1: Sherway Gardens Road / Sherway Gate.

Tuning notes:

- Retail: the four known lot proxies are caught (Highland Hill by Yorkdale, Borough / Progress at STC, Parkway Forest / Sheppard at Fairview, Sherway Gardens). Bremner / York is not: it has low severity and 21% FTR but no mall or large retail within 150 m. An FTR cutoff of 20% instead of 25% would also flag Yonge / Eglinton, a genuinely busy intersection.
- Expressway: the suggested ~100 m radius misses the two clearest cases, Airley Crescent (173 m from the DVP, 353x its class median) and Ferrand Drive (122 m, 257x). At 200 m a 5x or 10x cutoff starts flagging real downtown intersections (Bremner / York, Fleet / Strachan, Billy Bishop Way), so the ratio bar is 20x.

Excluded: 31 intersections (23 retail, 8 expressway) carrying 12,421 of 347,901 train-year intersection crashes (3.6%). Full list in `reports/excluded_intersections.csv`.

### Retail proxies (23)

| id | intersection | train crashes | % severe | % FTR | m to retail | m to motorway | class combo | x combo median |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 12553 | Alcester Court / Highland Hill | 1,250 | 2.5 | 44.6 | 52 | 279 | local x local | 416.7 |
| 3835 | Borough Drive / Progress Avenue | 1,148 | 3.8 | 46.9 | 20 | 283 | arterial x arterial | 7.0 |
| 27 | Parkway Forest Drive / Sheppard Avenue East | 947 | 5.3 | 37.9 | 7 | 208 | arterial x local | 49.8 |
| 16193 | Sherway Gardens Road / Sherway Gate | 495 | 4.2 | 43.4 | 7 | 175 | collector x collector | 33.0 |
| 3832 | Eglinton Avenue East / Hakimi Avenue / Lebovic Avenue | 425 | 3.5 | 40.7 | 19 | 2,898 | arterial x collector | 6.1 |
| 8745 | Daphne Road / Greencedar Circuit | 365 | 5.8 | 39.7 | 0 | 2,910 | collector x local | 73.0 |
| 6085 | Abitibi Avenue / Yonge Street | 322 | 6.8 | 35.7 | 15 | 4,307 | arterial x local | 16.9 |
| 8603 | Archer Street / Cobourg Avenue | 311 | 4.8 | 42.1 | 53 | 2,078 | local x local | 103.7 |
| 17455 | Scarsdale Road / York Mills Road | 289 | 6.2 | 31.1 | 17 | 1,196 | arterial x local | 15.2 |
| 14515 | Brimwood Boulevard / Sandhurst Circle | 283 | 7.4 | 35.7 | 18 | 3,359 | collector x collector | 18.9 |
| 14749 | Beverly Glen Boulevard / Bridletowne Circle | 274 | 5.1 | 38.0 | 14 | 1,771 | collector x collector | 18.3 |
| 12037 | Esther Shiner Boulevard / Provost Drive | 252 | 4.0 | 34.5 | 17 | 236 | collector x local | 50.4 |
| 7795 | Don Mills Road / Skymark Drive | 226 | 2.2 | 34.5 | 27 | 590 | arterial x local | 11.9 |
| 2811 | Blossomfield Drive / Dane Avenue | 211 | 2.8 | 38.9 | 5 | 295 | local x local | 70.3 |
| 5883 | Index Road / North Queen Street / Service Road | 192 | 4.7 | 31.8 | 16 | 22 | arterial x local | 10.1 |
| 21892 | Karl Fraser Road / Lawrence Avenue East | 181 | 5.5 | 38.1 | 14 | 882 | arterial x local | 9.5 |
| 11458 | Elkhorn Drive / Hawksbury Drive | 180 | 4.4 | 32.2 | 8 | 605 | local x local | 60.0 |
| 18449 | Blackwell Avenue / Tapscott Road | 173 | 4.6 | 39.9 | 12 | 1,223 | collector x collector | 11.5 |
| 17201 | Finch Avenue West / San Romanoway | 172 | 5.2 | 34.9 | 29 | 995 | arterial x local | 9.1 |
| 17186 | Jane Street / San Romanoway | 168 | 7.7 | 41.1 | 17 | 847 | arterial x local | 8.8 |
| 10728 | Lynnford Drive / The East Mall | 168 | 3.0 | 33.3 | 17 | 218 | arterial x local | 8.8 |
| 17445 | Bamburgh Circle / Wintermute Boulevard | 159 | 5.0 | 35.2 | 18 | 2,061 | collector x collector | 10.6 |
| 7781 | Don Mills Road East | 151 | 4.6 | 25.8 | 10 | 806 | arterial x arterial | 0.9 |

### Expressway proxies (8)

| id | intersection | train crashes | % severe | % FTR | m to retail | m to motorway | class combo | x combo median |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 8146 | Airley Crescent / Four Oaks Gate | 1,060 | 15.6 | 9.2 | 1,078 | 173 | local x local | 353.3 |
| 13619 | Laurentide Drive / York Mills Road | 869 | 19.1 | 6.9 | 908 | 56 | arterial x local | 45.7 |
| 8037 | Ferrand Drive / Seton Park Road | 771 | 13.4 | 8.0 | 362 | 122 | local x local | 257.0 |
| 8043 | Concorde Place / Wynford Drive / Wynford Heights Crescent | 390 | 20.3 | 8.5 | 837 | 141 | collector x local | 78.0 |
| 3911 | Antrim Crescent / Kennedy Road | 384 | 8.3 | 25.3 | 21 | 162 | arterial x local | 20.2 |
| 52 | Highway 401 Express | 233 | 21.0 | 8.2 | 785 | 0 | highway x highway | 22.2 |
| 676 | Queen Elizabeth Boulevard / Taymall Avenue | 221 | 4.5 | 23.5 | 17 | 33 | local x local | 73.7 |
| 22073 | Eastwick Road / Oxford Street | 151 | 11.3 | 23.8 | 101 | 24 | local x local | 50.3 |

### Near misses (not excluded, listed for transparency)

Busy, severe <= 10%, FTR >= 20%, but not flagged as retail:

| id | intersection | train crashes | % severe | % FTR | m to retail | m to motorway | class combo | x combo median |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 12973 | Canadian Road / Warden Avenue | 677 | 5.3 | 22.7 | 49 | 1,057 | arterial x local | 35.6 |
| 7579 | Eglinton Avenue East / Eglinton Avenue West / Yonge Street | 489 | 6.1 | 20.0 | 32 | 3,073 | arterial x arterial | 3.0 |
| 3968 | Danforth Avenue / Victoria Park Avenue | 482 | 9.5 | 21.0 | 15 | 3,784 | arterial x arterial | 2.9 |
| 19398 | Allanford Road / Sheppard Avenue East | 421 | 9.0 | 33.7 | 21 | 869 | arterial x local | 22.2 |
| 847 | Billy Bishop Way / Transit Road / Wilson Avenue | 346 | 9.5 | 20.2 | 26 | 196 | arterial x local | 18.2 |
| 2019 | Bremner Boulevard / York Street | 300 | 4.3 | 20.7 | 607 | 132 | arterial x local | 15.8 |
| 4237 | City View Drive / Dixon Road / Kelfield Street | 288 | 8.3 | 21.5 | 1,574 | 252 | arterial x local | 15.2 |
| 14472 | Midland Avenue / Silver Star Boulevard / South Shields Avenue | 288 | 9.4 | 24.3 | 159 | 3,368 | arterial x local | 15.2 |
| 14574 | Anthony Road / Dufferin Street | 264 | 4.5 | 37.9 | 156 | 189 | arterial x local | 13.9 |
| 13242 | Cherry Street / Polson Street | 249 | 9.2 | 36.5 | 1,250 | 670 | collector x local | 49.8 |
| 14031 | Dallner Road / Kirby Road | 243 | 8.2 | 33.3 | 10 | 359 | local x local | 81.0 |
| 4921 | Gerrard Street East / Marjory Avenue | 238 | 8.8 | 38.2 | 13 | 1,532 | arterial x local | 12.5 |
| 4479 | Laird Drive / McRae Drive / Wicksteed Avenue | 192 | 3.1 | 22.4 | 10 | 1,698 | arterial x collector | 2.7 |
| 20372 | Fenmar Drive / Kenhar Drive | 179 | 8.4 | 21.2 | 1,396 | 1,384 | collector x local | 35.8 |
| 10945 | Esquire Road / Victoria Park Avenue | 170 | 8.2 | 21.8 | 228 | 289 | arterial x local | 8.9 |
| 14503 | James Findlay Way / Wilson Avenue | 161 | 8.1 | 27.3 | 26 | 336 | arterial x local | 8.5 |
| 5700 | Finch Avenue West / Tangiers Road | 159 | 8.8 | 25.8 | 23 | 3,290 | arterial x local | 8.4 |
| 2497 | Erskine Avenue / Roselawn Avenue / Yonge Street | 155 | 7.1 | 20.6 | 0 | 3,168 | arterial x local | 8.2 |
| 12893 | Eglinton Avenue East / Prudham Gate | 153 | 7.8 | 24.8 | 23 | 3,458 | arterial x local | 8.1 |

Within the motorway radius at 10-20x their class median:

| id | intersection | train crashes | % severe | % FTR | m to retail | m to motorway | class combo | x combo median |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 92 | Lake Shore Boulevard West / Lower Spadina Avenue / Spadina Avenue | 1,056 | 11.8 | 12.4 | 490 | 5 | highway x arterial | 10.9 |
| 847 | Billy Bishop Way / Transit Road / Wilson Avenue | 346 | 9.5 | 20.2 | 26 | 196 | arterial x local | 18.2 |
| 2019 | Bremner Boulevard / York Street | 300 | 4.3 | 20.7 | 607 | 132 | arterial x local | 15.8 |
| 21076 | Falstaff Avenue / Gordon Mackay Road / Jane Street | 267 | 16.9 | 9.4 | 13 | 163 | arterial x local | 14.1 |
| 14574 | Anthony Road / Dufferin Street | 264 | 4.5 | 37.9 | 156 | 189 | arterial x local | 13.9 |
| 70 | Consumers Road / Meadowacres Drive / Victoria Park Avenue | 250 | 14.8 | 11.2 | 410 | 107 | arterial x local | 13.2 |
| 8842 | Lake Shore Boulevard West / Lower Sherbourne Street | 211 | 11.8 | 7.1 | 1,277 | 9 | arterial x local | 11.1 |
| 14648 | Kennedy Road / Village Green Square | 210 | 20.5 | 10.5 | 369 | 47 | arterial x local | 11.1 |
| 1342 | Fleet Street / Manitoba Drive / Strachan Avenue | 181 | 21.0 | 7.7 | 676 | 131 | collector x collector | 12.1 |
| 4838 | Bremner Boulevard / Lower Simcoe Street | 158 | 10.8 | 11.4 | 513 | 189 | collector x collector | 10.5 |

## Label stats per split (clean intersections)

| split | years | rows (clean) | % zero-crash | mean crashes | max crashes | mean severe | % any severe | crashes (clean) | crashes (excluded ints) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| history | 2014, 2015, 2020, 2021 | 89,616 | 63.3 | 1.9 | 225 | 0.3 | 13.2 | 171,597 | 6,072 |
| train | 2016, 2017, 2018, 2019, 2022, 2023 | 134,424 | 59.5 | 2.5 | 242 | 0.4 | 15.0 | 335,480 | 12,421 |
| val | 2024 | 22,404 | 59.0 | 2.4 | 277 | 0.4 | 15.8 | 54,370 | 2,070 |
| test | 2025 | 22,404 | 59.3 | 2.4 | 256 | 0.4 | 15.4 | 52,719 | 2,110 |
