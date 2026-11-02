# Ottawa collision and volume data

Source: City of Ottawa open data on ArcGIS Online (https://services.arcgis.com/G6F8XLCl5KtAlZ2G/arcgis/rest/services), Open Government Licence - City of Ottawa. Every layer was paginated by object id and its row count checked against `returnCountOnly`.

| kind | service | rows |
|---|---|---:|
| collisions | 2013_Tabular_Transportation_Collision_Data | 15,156 |
| collisions | 2014_Tabular_Transportation_Collision_Data | 14,843 |
| collisions | 2015_Tabular_Transportation_Collision_Data | 15,077 |
| collisions | 2016_Tabular_Transportation_Collision_Data | 14,028 |
| collisions | 2017_Tabular_Transportation_Collision_Data | 14,398 |
| collisions | 2018_Tabular_Transportation_Collision_Data | 14,529 |
| collisions | 2019_Tabular_Transportation_Collision_Data | 16,399 |
| collisions | 2020_Tabular_Transportation_Collision_Data | 10,047 |
| collisions | Traffic_Collisions_by_Location_2017-2024_(excluding_2023) | 94,406 |
| volumes | Transportation_Intersection_Volumes_2016 | 687 |
| volumes | Transportation_Intersection_Volumes_2017 | 584 |
| volumes | Transportation_Intersection_Volumes_2018 | 553 |
| volumes | Transportation_Intersection_Volumes_2019 | 910 |
| volumes | Transportation_Intersection_Volumes_2021 | 180 |
| volumes | Intersection_Volume_2022_w_lat_long | 474 |
| volumes | Transportation_Intersection_Volume_2023 | 483 |
| volumes | Transportation_Intersection_Volumes_2024 | 703 |
| volumes | Transportation_Intersection_Volume_2025 | 655 |

The service named `Traffic_Collision_Data` does not exist on the City's host (the only item with that title is a third-party copy whose URL returns 'Invalid URL'), so the City's own layers above are used. 2023 collisions are not published, and 2025 is not out yet.

## Collisions per year

153,510 records read for the years used; 2,563 non-reportable records dropped (class 04, below the reporting threshold: under 100 a year until 2022, then 2,358 in 2024, a recording change); 72 with zero or out-of-city coordinates dropped. Severe = fatal or non-fatal injury (classes 01-02).

| year | source | records kept | freeway removed | used (non-freeway) | % severe | fatal | months present | non-reportable dropped | bad coords dropped | partial? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2013 | yearly layer | 15,156 | 1,601 | 13,555 | 18.4 | 21 | 12 | 0 | 0 |  |
| 2014 | yearly layer | 14,843 | 1,386 | 13,457 | 18.0 | 28 | 12 | 0 | 0 |  |
| 2015 | yearly layer | 15,077 | 1,257 | 13,820 | 19.1 | 20 | 12 | 0 | 0 |  |
| 2016 | yearly layer | 14,028 | 1,239 | 12,789 | 19.8 | 24 | 12 | 0 | 0 |  |
| 2017 | 2017-2024 layer | 14,399 | 1,559 | 12,840 | 19.3 | 28 | 12 | 25 | 0 |  |
| 2018 | 2017-2024 layer | 14,529 | 1,541 | 12,988 | 18.6 | 26 | 12 | 26 | 1 |  |
| 2019 | 2017-2024 layer | 16,438 | 1,638 | 14,800 | 16.6 | 25 | 12 | 27 | 0 |  |
| 2020 | 2017-2024 layer | 10,048 | 1,121 | 8,927 | 17.6 | 18 | 12 | 18 | 1 |  |
| 2021 | 2017-2024 layer | 9,694 | 1,048 | 8,646 | 19.8 | 23 | 12 | 13 | 41 |  |
| 2022 | 2017-2024 layer | 12,365 | 1,238 | 11,127 | 16.3 | 23 | 12 | 96 | 29 |  |
| 2024 | 2017-2024 layer | 14,298 | 1,605 | 12,693 | 8.8 | 27 | 12 | 2,358 | 0 |  |

2024 looks different: its severe share is 8.8% against 16-20% before. Fatal and major-injury crashes are steady (26 and 118 in 2024 vs 23 and 112 in 2022) but crashes whose worst injury is minimal or minor drop from 1,887 to 1,104, and 2,358 non-reportable records appear. This looks like a change in how minor-injury collisions are recorded in 2024, not a real drop, so the severe target is not comparable in the 2024 test year. The primary target (any crash) is unaffected as far as the totals show (12,693 non-freeway in 2024 vs 11,127 in 2022 and 14,800 in 2019).

Freeway crashes removed: **15,233** (10.1% of located records). Rule: every road the location string names is a Hwy 416 / 417 mainline or ramp (mid-block on the highway or a ramp, or a ramp-to-ramp junction). Ramp terminals on city streets are kept: they are surface intersections, the kind TPS also records at 400-series interchanges. City expressways (Hwy 174, parkways) are kept, as the Gardiner and DVP are in Toronto.

| road | crashes |
|---|---:|
| Hwy 417 mainline | 12,442 |
| Hwy 417 ramp | 1,563 |
| Hwy 416 mainline | 1,068 |
| Hwy 416 ramp | 160 |

Location string kinds (all located records): `at` = 'A @ B' intersection, `btwn` = 'A btwn B & C' mid-block, `other` = blank or free text.

| location_kind | records |
|---|---:|
| at | 79,874 |
| btwn | 70,798 |
| other | 203 |

## Intersection volume counts

5,229 published counts; 123 2025 counts the City did not factor to AADT dropped, 0 without a volume, 0 outside the city; 5,106 kept. Values are the City's factored 24-hour AADT (all motorised vehicles entering the intersection). 2020 has no layer; 2021 counts are COVID-era and are never used. Matching to intersections and the one-count-per-intersection rule are in `reports/ottawa.md`.

| layer_year | counts | median_aadt | p95_aadt | first_date | last_date |
|---|---:|---:|---:|---|---|
| 2016 | 687 | 16196 | 44725 | 2016-01-12 | 2016-12-08 |
| 2017 | 584 | 14680 | 41267 | 2017-01-11 | 2017-12-21 |
| 2018 | 553 | 14105 | 42871 | 2018-01-09 | 2018-12-19 |
| 2019 | 910 | 8781 | 38827 | 2019-01-08 | 2019-12-19 |
| 2021 | 180 | 6794 | 27701 | 2021-02-17 | 2021-12-15 |
| 2022 | 474 | 11968 | 32721 | 2022-01-11 | 2022-12-20 |
| 2023 | 483 | 10296 | 41867 | 2023-01-10 | 2023-12-21 |
| 2024 | 703 | 12900 | 37893 | 2024-01-09 | 2024-12-19 |
| 2025 | 532 | 9110 | 39188 | 2025-01-07 | 2025-12-10 |
