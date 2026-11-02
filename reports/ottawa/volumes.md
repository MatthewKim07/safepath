# Ottawa volume-model transfer

City counts: 5,106; matched to a consolidated intersection within 40 m: 4,124 (80.8%, median 2.1 m, p95 10.9 m; 99% share a street name with the OSM intersection). Unmatched counts are mostly outside the study area. 146 matched 2021 counts dropped (COVID). Counted intersections: **2,297** of 15,276 (15.0%; Toronto 26.5%).

Target per counted intersection: log of the mean AADT in its latest non-COVID count year (2016-2025). Toronto's model sees `count_year` = that year.

Betweenness rescaled to Toronto's graph: Toronto 27,426 nodes, Ottawa 19,013; global x2.08, local x1.44. Class combos Toronto never saw: none.

## Accuracy on Ottawa's counted intersections

Same metrics as Toronto's spatial CV (Toronto CV, overall: R² 0.85, median abs % error 21%; class-median baseline R² 0.72, 33%).

| class combo | model | n | log-RMSE | median abs % err | R² | bias (log) |
|---|---|---:|---:|---:|---:|---:|
| all | Ottawa class median (spatial CV) | 2,297 | 0.656 | 39.214 | 0.721 | 0.031 |
| all | Toronto model, betweenness rescaled | 2,297 | 0.819 | 66.056 | 0.566 | 0.590 |
| all | Toronto model, raw betweenness | 2,297 | 0.750 | 50.001 | 0.636 | 0.490 |
| all | Ottawa-trained LightGBM (spatial CV, upper bound) | 2,297 | 0.487 | 26.215 | 0.846 | 0.005 |
| arterial x arterial | Ottawa class median (spatial CV) | 218 | 0.607 | 36.251 | -0.072 | 0.100 |
| arterial x arterial | Toronto model, betweenness rescaled | 218 | 0.569 | 48.582 | 0.060 | 0.431 |
| arterial x arterial | Toronto model, raw betweenness | 218 | 0.513 | 39.321 | 0.236 | 0.356 |
| arterial x arterial | Ottawa-trained LightGBM (spatial CV, upper bound) | 218 | 0.353 | 18.828 | 0.639 | -0.021 |
| arterial x collector | Ottawa class median (spatial CV) | 350 | 0.486 | 33.466 | -0.040 | 0.056 |
| arterial x collector | Toronto model, betweenness rescaled | 350 | 0.513 | 42.383 | -0.156 | 0.385 |
| arterial x collector | Toronto model, raw betweenness | 350 | 0.443 | 29.249 | 0.137 | 0.291 |
| arterial x collector | Ottawa-trained LightGBM (spatial CV, upper bound) | 350 | 0.318 | 18.675 | 0.555 | -0.006 |
| arterial x local | Ottawa class median (spatial CV) | 596 | 0.549 | 33.428 | -0.019 | 0.017 |
| arterial x local | Toronto model, betweenness rescaled | 596 | 0.596 | 50.698 | -0.198 | 0.435 |
| arterial x local | Toronto model, raw betweenness | 596 | 0.523 | 36.127 | 0.078 | 0.331 |
| arterial x local | Ottawa-trained LightGBM (spatial CV, upper bound) | 596 | 0.424 | 23.829 | 0.393 | 0.034 |
| collector x collector | Ottawa class median (spatial CV) | 176 | 0.608 | 39.328 | -0.049 | 0.110 |
| collector x collector | Toronto model, betweenness rescaled | 176 | 0.643 | 42.975 | -0.174 | 0.425 |
| collector x collector | Toronto model, raw betweenness | 176 | 0.574 | 35.624 | 0.065 | 0.317 |
| collector x collector | Ottawa-trained LightGBM (spatial CV, upper bound) | 176 | 0.463 | 27.525 | 0.390 | 0.002 |
| collector x local | Ottawa class median (spatial CV) | 541 | 0.719 | 47.609 | -0.007 | -0.002 |
| collector x local | Toronto model, betweenness rescaled | 541 | 0.930 | 108.422 | -0.688 | 0.752 |
| collector x local | Toronto model, raw betweenness | 541 | 0.828 | 83.127 | -0.337 | 0.621 |
| collector x local | Ottawa-trained LightGBM (spatial CV, upper bound) | 541 | 0.518 | 32.450 | 0.477 | -0.001 |
| highway x arterial | Ottawa class median (spatial CV) | 54 | 0.476 | 30.410 | -0.199 | 0.069 |
| highway x arterial | Toronto model, betweenness rescaled | 54 | 0.551 | 47.871 | -0.611 | 0.389 |
| highway x arterial | Toronto model, raw betweenness | 54 | 0.492 | 39.675 | -0.285 | 0.315 |
| highway x arterial | Ottawa-trained LightGBM (spatial CV, upper bound) | 54 | 0.334 | 24.973 | 0.410 | 0.016 |
| local x local | Ottawa class median (spatial CV) | 341 | 0.897 | 52.859 | -0.009 | -0.015 |
| local x local | Toronto model, betweenness rescaled | 341 | 1.331 | 191.633 | -1.219 | 1.055 |
| local x local | Toronto model, raw betweenness | 341 | 1.277 | 165.349 | -1.045 | 0.987 |
| local x local | Ottawa-trained LightGBM (spatial CV, upper bound) | 341 | 0.733 | 40.488 | 0.326 | -0.006 |

AADT used by the crash models: **Ottawa-trained intersection model**, chosen on overall log-RMSE, blended with the latest count where one exists: w = v_model / (v_model + s0² + tau² x age), with v_model = 0.237 (that model's squared error on Ottawa counts), s0² = 0.0359, tau² = 0.0000 from 1681 Ottawa repeat-count pairs.

Entering AADT by class combo (all intersections, after blending):

| class_combo | intersections | median AADT |
|---|---:|---:|
| local x local | 7,484 | 659 |
| collector x local | 3,160 | 2,859 |
| arterial x arterial | 1,565 | 19,730 |
| arterial x local | 1,370 | 14,558 |
| collector x collector | 614 | 5,938 |
| arterial x collector | 595 | 16,908 |
| highway x highway | 266 | 23,308 |
| highway x arterial | 191 | 22,442 |
| highway x local | 20 | 25,866 |
| highway x collector | 11 | 19,590 |

Segment AADT, used for major / minor approach volumes, neighbour and GNN edge features: Toronto's segment model (Ottawa has no segment counts in this set). The betweenness variant is the one that fits Ottawa's intersection counts better (raw betweenness), and predictions are scaled by x0.61, the inverse of that intersection model's mean over-prediction on Ottawa counts. Medians before and after:

| highway | uncorrected | used |
|---|---:|---:|
| motorway | 327,886 | 200,901 |
| motorway_link | 6,837 | 4,189 |
| other | 4,814 | 2,949 |
| primary | 23,309 | 14,282 |
| residential | 1,211 | 742 |
| secondary | 18,477 | 11,321 |
| tertiary | 5,124 | 3,140 |
| unclassified | 3,032 | 1,858 |
