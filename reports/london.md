# London, Ontario demo (CP10)

**No local crash data.** London publishes no collision records, so London risk is Toronto's Safety Performance Function (no crash history, no Empirical Bayes) on London's road features and the City's traffic volumes. In Ottawa this setting (the count-free SPF) had recall@5% 0.619 against 0.288 for a road-class baseline, with no Ottawa data at all, but it was below Ottawa's own crash history on every ranking metric. Absolute numbers carry Toronto's reporting level and are not calibrated for London: treat them as **relative** risk.

## Graph

OSM drive graph for "London, Ontario, Canada" (the municipal boundary): 7,402 nodes, 19,405 directed edges, 10,655 undirected segments; consolidation at 10 m (as Toronto) gives **5,637** intersections. Static OSM features and context layers come from the same code as Toronto's (London has no on-street tram).

## Traffic volumes

City of London open data, *Traffic Volumes* (`OpenData_Transportation/MapServer/21`), under the City's [Terms of Use - City of London Open Data](https://london.maps.arcgis.com/sharing/rest/content/items/e31458fd0c7e41dd9f93144a9550781d/data): 1,973 line segments, 854 km, AADT rounded to 500s. Every row was last processed on 2023-10-04; the City does not publish count dates, so the volumes' age is unknown (the layer is refreshed about yearly from its count programme). The companion *Intersection Types* layer (MapServer/16) only marks where volume segments start and end, so it is downloaded for reference but not used.

Matched to OSM (samples at 25/50/75% of each OSM segment within 15 m of a City line with the same street name, or 8 m if unnamed; two of three samples must agree): **1,780 of 1,973 City segments (90%; 91% of City km)** matched; 4,210 OSM segments carry a City AADT. Unmatched City segments by City class: {'Neighbourhood Connector': 49, 'Civic Boulevard': 34, 'Rural Thoroughfare': 30, 'Neighbourhood Street': 17, 'Rural Connector': 17, 'Urban Thoroughfare': 16, 'Rapid Transit Boulevard': 10, 'Main Street': 8, 'Ramps': 6, 'Expressway': 4, 'Private Miscellaneous': 2}.

| highway | segments | km | % segments with City AADT | % km with City AADT | median City AADT |
|---|---:|---:|---:|---:|---:|
| residential | 5,608 | 929.4 | 6.2 | 5.4 | 1,500.0 |
| tertiary | 2,752 | 403.3 | 72.8 | 83.2 | 3,000.0 |
| secondary | 1,698 | 365.6 | 93.3 | 95.1 | 21,000.0 |
| unclassified | 163 | 68.3 | 20.9 | 36.6 | 1,000.0 |
| motorway | 49 | 56.9 | 26.5 | 22.7 | 40,000.0 |
| primary | 137 | 37.4 | 88.3 | 94.1 | 18,000.0 |
| motorway_link | 84 | 28.3 | 27.4 | 14.4 | 7,500.0 |
| trunk | 54 | 24.7 | 94.4 | 99.9 | 20,000.0 |
| secondary_link | 89 | 6.0 | 27.0 | 23.8 | 13,000.0 |
| trunk_link | 10 | 1.3 | 60.0 | 62.0 | 14,000.0 |
| tertiary_link | 7 | 0.3 | 28.6 | 23.7 | 5,500.0 |
| primary_link | 2 | 0.3 | 0.0 | 0.0 | nan |
| busway | 2 | 0.2 | 0.0 | 0.0 | nan |

OSM segments without a City value take the median City AADT of matched segments of the same OSM class (Ottawa showed Toronto's volume model transfers poorly). Residential streets are mostly unmatched: the City counts few local streets, and the ones it counts are likely busier than typical, so local-street volumes are probably overstated. 37 OSM segments on Hwy 401 / 402 take MTO 2024 AADT.

Entering AADT per intersection = half the two-way AADT of each approach, summed. Checked on Toronto, where entering volume is measured by turning counts: on 5,751 counted intersections this approximation has Spearman 0.89 with the counts and runs low by a median factor of 0.78 (median abs error 28%), so London's entering volumes are divided by that factor. Major / minor approach volumes are defined exactly as Toronto's.

## Hwy 401 / 402 (MTO)

Sections mapped with Toronto's interchange method (naming rule per highway: {'401': 'end', '402': 'start'}); 2015-2019 pooled collision rates per million vehicle-km:

| hwy | lhrs | description | aadt_2024 | cr | cr_source | mapped_km |
|---|---:|---|---:|---:|---|---:|
| 401 | 47,830 | HWY 4/COL. TALBOT RD IC-177 | 28,000.00 | 0.27 | section | 0.72 |
| 401 | 47,828 | WONDERLAND RD IC 180 | 29,800.00 | 0.45 | section | 5.42 |
| 401 | 47,826 | HWY 402 IC-183 | 34,000.00 | 0.36 | section | 9.11 |
| 401 | 47,823 | WELLINGTON RD IC-186 | 66,100.00 | 0.32 | section | 3.92 |
| 401 | 47,820 | HIGHBURY AV IC 189 | 77,300.00 | 0.35 | section | 6.41 |
| 401 | 47,815 | VETERANS MEMORIAL  IC 194 | 78,700.00 | 0.25 | section | 8.95 |
| 401 | 47,810 | WESTCHESTER BOURNE-MIDDLESEX RD 74 IC 195 | 71,700.00 | 0.43 | section | 0.61 |
| 402 | 48,111 | HWY 401 IC - LONDON | 32,400.00 | 0.26 | section | 7.43 |
| 402 | 48,112 | WONDERLAND RD IC | 28,000.00 | 0.40 | section | 4.31 |
| 402 | 48,115 | HWY 4 IC-98  COL. TALBOT RD | 26,000.00 | 0.22 | section | 1.17 |

## Risk

Intersection: Toronto SPF expected crashes/yr at Toronto's 2025 level / (entering AADT x 365), charged once per consolidated intersection. Segments: Toronto's mid-block crashes per million vehicle-km by class; 401 / 402 MTO section rates; Highbury Avenue's freeway section Toronto's 'other expressway' rate.

SPF expected crashes/yr: total 14,458, median 0.83; per million entering vehicles median 0.66. Highest SPF expectation:

| intersection | risk | entering_aadt | risk_per_mev | class_combo |
|---|---:|---:|---:|---|
| Fanshawe Park Road East / Fanshawe Park Road West / Richmond Street | 109.77 | 74,281.73 | 4.05 | arterial x arterial |
| Oxford Street West / Wonderland Road North | 106.25 | 97,334.68 | 2.99 | arterial x arterial |
| Springbank Drive / Wonderland Road / Wonderland Road North / Wonderland Road South | 99.77 | 77,483.52 | 3.53 | arterial x arterial |
| Commissioners Road East / Wellington Road | 99.23 | 89,650.36 | 3.03 | arterial x arterial |
| Highbury Avenue North / Oxford Street East | 96.18 | 90,931.08 | 2.90 | arterial x arterial |
| Fanshawe Park Road / Fanshawe Park Road West / Wonderland Road North | 91.05 | 74,922.09 | 3.33 | arterial x arterial |
| Sarnia Road / Wonderland Road North | 87.45 | 83,887.12 | 2.86 | arterial x arterial |
| Commissioners Road East / Commissioners Road West / Wharncliffe Road South | 85.70 | 81,966.04 | 2.86 | arterial x arterial |
| Oxford Street East / Richmond Street | 83.45 | 75,562.45 | 3.03 | arterial x arterial |
| Southdale Road West / Wonderland Road South | 83.44 | 76,843.17 | 2.97 | arterial x arterial |
| Southdale Road East / Wellington Road | 82.26 | 79,404.60 | 2.84 | arterial x arterial |
| Commissioners Road West / Wonderland Road South | 80.23 | 74,281.73 | 2.96 | arterial x arterial |
| Oxford Street West / Wharncliffe Road North | 79.47 | 76,843.17 | 2.83 | arterial x arterial |
| Adelaide Street North / Oxford Street East | 78.29 | 71,079.93 | 3.02 | arterial x arterial |
| Adelaide Street North / Fanshawe Park Road East | 77.95 | 65,957.05 | 3.24 | arterial x arterial |

## Sanity check: London Police's published top-10 collision intersections

London Police published yearly top-10 high-collision intersection lists for 2014-2016 (from news coverage). They are not a rigorous label set: police-reported counts with no exposure adjustment, a decade old, and sometimes ranked by different criteria. Each is matched to the consolidated intersection whose OSM names contain both streets (the busiest if several). Percentile 100 = highest in London.

| year | rank | intersection | OSM names | SPF expected/yr | entering AADT | SPF percentile | volume-only percentile |
|---|---:|---|---|---:|---:|---:|---:|
| 2014 | 1 | Wellington/Commissioners | Commissioners Road East / Wellington Road | 99.2 | 89,650.4 | 99.9 | 99.9 |
| 2014 | 2 | Highbury/Oxford | Highbury Avenue North / Oxford Street East | 96.2 | 90,931.1 | 99.9 | 100.0 |
| 2014 | 3 | Highbury/Hamilton | Hamilton Road / Highbury Avenue North | 66.7 | 77,483.5 | 99.6 | 99.8 |
| 2014 | 4 | Wonderland/Oxford | Oxford Street West / Wonderland Road North | 106.2 | 97,334.7 | 100.0 | 100.0 |
| 2014 | 5 | Fanshawe Park/Adelaide | Adelaide Street North / Fanshawe Park Road East | 77.9 | 65,957.1 | 99.8 | 99.6 |
| 2014 | 6 | Wharncliffe/Oxford | Oxford Street West / Wharncliffe Road North | 79.5 | 76,843.2 | 99.8 | 99.8 |
| 2014 | 7 | Sarnia/Wonderland | Sarnia Road / Wonderland Road North | 87.5 | 83,887.1 | 99.9 | 99.9 |
| 2014 | 8 | Southdale/Wonderland | Southdale Road West / Wonderland Road South | 83.4 | 76,843.2 | 99.8 | 99.8 |
| 2014 | 9 | Oxford/Adelaide | Adelaide Street North / Oxford Street East | 78.3 | 71,079.9 | 99.8 | 99.6 |
| 2014 | 10 | Oxford/Richmond | Oxford Street East / Richmond Street | 83.5 | 75,562.4 | 99.9 | 99.7 |
| 2015 | 1 | Wellington/Exeter | Exeter Road / Wellington Road | 57.9 | 92,852.2 | 99.5 | 100.0 |
| 2015 | 2 | Wellington/Commissioners | Commissioners Road East / Wellington Road | 99.2 | 89,650.4 | 99.9 | 99.9 |
| 2015 | 3 | Highbury/Oxford | Highbury Avenue North / Oxford Street East | 96.2 | 90,931.1 | 99.9 | 100.0 |
| 2015 | 4 | Wharncliffe/Oxford | Oxford Street West / Wharncliffe Road North | 79.5 | 76,843.2 | 99.8 | 99.8 |
| 2015 | 5 | Hamilton/Highbury | Hamilton Road / Highbury Avenue North | 66.7 | 77,483.5 | 99.6 | 99.8 |
| 2015 | 6 | Adelaide/Oxford | Adelaide Street North / Oxford Street East | 78.3 | 71,079.9 | 99.8 | 99.6 |
| 2015 | 7 | Fanshawe Park/Adelaide | Adelaide Street North / Fanshawe Park Road East | 77.9 | 65,957.1 | 99.8 | 99.6 |
| 2015 | 8 | Wonderland/Oxford | Oxford Street West / Wonderland Road North | 106.2 | 97,334.7 | 100.0 | 100.0 |
| 2015 | 9 | Sarnia/Wonderland | Sarnia Road / Wonderland Road North | 87.5 | 83,887.1 | 99.9 | 99.9 |
| 2015 | 10 | Highbury/Huron | Highbury Avenue North / Huron Street | 63.4 | 64,036.0 | 99.6 | 99.5 |
| 2016 | 1 | Wellington/Exeter | Exeter Road / Wellington Road | 57.9 | 92,852.2 | 99.5 | 100.0 |
| 2016 | 2 | Wellington/Commissioners | Commissioners Road East / Wellington Road | 99.2 | 89,650.4 | 99.9 | 99.9 |
| 2016 | 3 | Highbury/Oxford | Highbury Avenue North / Oxford Street East | 96.2 | 90,931.1 | 99.9 | 100.0 |
| 2016 | 4 | Wharncliffe/Oxford | Oxford Street West / Wharncliffe Road North | 79.5 | 76,843.2 | 99.8 | 99.8 |
| 2016 | 5 | Oxford/Richmond | Oxford Street East / Richmond Street | 83.5 | 75,562.4 | 99.9 | 99.7 |
| 2016 | 6 | Oxford/Wonderland | Oxford Street West / Wonderland Road North | 106.2 | 97,334.7 | 100.0 | 100.0 |
| 2016 | 7 | Fanshawe Park/Adelaide | Adelaide Street North / Fanshawe Park Road East | 77.9 | 65,957.1 | 99.8 | 99.6 |
| 2016 | 8 | Adelaide/Oxford | Adelaide Street North / Oxford Street East | 78.3 | 71,079.9 | 99.8 | 99.6 |
| 2016 | 9 | Wonderland/Southdale | Southdale Road West / Wonderland Road South | 83.4 | 76,843.2 | 99.8 | 99.8 |
| 2016 | 10 | Highbury/Huron | Highbury Avenue North / Huron Street | 63.4 | 64,036.0 | 99.6 | 99.5 |
| 2016 | 11 | Highbury/Hamilton | Hamilton Road / Highbury Avenue North | 66.7 | 77,483.5 | 99.6 | 99.8 |

| ranking | listed intersections | in top 1% | in top 5% | in top 10% | median percentile |
|---|---:|---:|---:|---:|---:|
| SPF | 12 | 12 | 12 | 12 | 99.8 |
| volume-only | 12 | 12 | 12 | 12 | 99.8 |

Of 12 distinct listed intersections, the SPF ranks 6 higher than volume alone does; the SPF puts 12 and volume alone 12 in London's top 1%. In a city without crash data, most of what the SPF knows about which intersections see the most crashes is how much traffic they carry, which is also what police counts track. The check shows the SPF is not wrong about London's known hotspots; it cannot show the SPF beats volume.

## Routing (200 random trips, 3-20 km)

k recalibrated with Toronto's rule (at λ = 1 the median route's risk cost equals its travel time): **23.9 s** per expected crash per million trips (Toronto 31.1).

| lam | median extra min | p90 extra min | median % risk reduction | p90 % risk reduction | median detour (km ratio) | p90 detour (km ratio) | motorway share | residential share | routes with a U-turn |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.052 | 0.045 | 0.000 |
| 0.500 | 0.298 | 1.314 | 12.254 | 35.985 | 1.002 | 1.176 | 0.089 | 0.082 | 0.000 |
| 1.000 | 0.675 | 2.957 | 17.460 | 43.452 | 1.023 | 1.386 | 0.103 | 0.107 | 0.000 |
| 2.000 | 1.113 | 4.372 | 20.731 | 44.954 | 1.075 | 1.612 | 0.116 | 0.115 | 0.005 |

## Limitations

- No London crash records: risk is a Toronto model's expectation, never checked against London outcomes beyond the decade-old police top-10 lists. Relative risk only.
- Volume layer age unknown (single processing date, no count dates); AADT rounded to 500; most local streets have no count and use a class median.
- Entering volume is approximated from segment volumes and rescaled with a factor measured in Toronto.
- Segment rates and the SPF level are Toronto's; London's own reporting level and road design are not reflected.
- Static features are today's OSM; free-flow travel times.
