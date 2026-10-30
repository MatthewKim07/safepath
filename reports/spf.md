# Safety Performance Function (CP6)

Negative binomial (NB2, statsmodels) on clean intersection-years in the train label years (134,424 rows). Volumes from CP5: major/minor road AADT from the two-way AADT of the approach segments, grouped by street name; single-road intersections use entering AADT with an indicator. Static terms: signal, legs, approach-class combo, streetcar within 20 m, minimum leg angle. Year fixed effects for the train years; other years get a citywide level (see `spf.py`).

## Pooled vs grouped

| SPF | parameters | AIC (train) | Poisson deviance val 2024 |
|---|---:|---:|---:|
| pooled | 22 | 379,523.8675 | 3.2496 |
| grouped | 60 | 371,509.8256 | 2.7375 |

Chosen: **grouped** (lower AIC and validation deviance no worse).

## Volume elasticities

Literature range for intersection SPFs: roughly 0.3-0.9 per approach.

| SPF | rows | AIC | k | reference levels | major elasticity [95% CI] | major in 0.3-0.9 | minor elasticity [95% CI] | minor in 0.3-0.9 |
|---|---:|---:|---:|---|---|---|---|---|
| signalised 3-leg | 6,684 | 36,545.440 | 1.104 | class_combo_arterial x local | 0.596 [0.511, 0.680] | yes | 0.590 [0.540, 0.640] | yes |
| signalised 4+ leg | 8,664 | 63,677.288 | 0.528 | legs_4, class_combo_arterial x local | 0.527 [0.466, 0.587] | yes | 0.656 [0.624, 0.688] | yes |
| unsignalised | 119,076 | 271,287.098 | 1.809 | legs_3-, class_combo_local x local | 0.318 [0.297, 0.339] | yes | 0.505 [0.486, 0.525] | yes |

## Coefficients (chosen SPF)

Categorical contrasts are against each group's most common level (the `reference levels` column above). Year effects omitted.

| SPF | term | coef | 95% CI |
|---|---|---:|---|
| signalised 3-leg | const | -7.513 | [-8.374, -6.651] |
| signalised 3-leg | log_major | 0.596 | [0.511, 0.680] |
| signalised 3-leg | log_minor | 0.590 | [0.540, 0.640] |
| signalised 3-leg | log_entering_single | 0.460 | [0.276, 0.644] |
| signalised 3-leg | single_road | 4.754 | [2.911, 6.597] |
| signalised 3-leg | streetcar_20m | 0.223 | [0.125, 0.320] |
| signalised 3-leg | min_leg_angle_10deg | -0.103 | [-0.120, -0.087] |
| signalised 3-leg | class_combo_arterial x arterial | -0.722 | [-0.852, -0.592] |
| signalised 3-leg | class_combo_arterial x collector | -0.168 | [-0.255, -0.082] |
| signalised 3-leg | class_combo_collector x collector | -0.300 | [-0.452, -0.149] |
| signalised 3-leg | class_combo_collector x local | -0.149 | [-0.284, -0.015] |
| signalised 3-leg | class_combo_highway x any | -2.804 | [-2.973, -2.635] |
| signalised 3-leg | class_combo_local x local | -0.599 | [-0.849, -0.349] |
| signalised 3-leg | k (overdispersion) | 1.104 | [1.055, 1.153] |
| signalised 4+ leg | const | -8.531 | [-9.138, -7.924] |
| signalised 4+ leg | log_major | 0.527 | [0.466, 0.587] |
| signalised 4+ leg | log_minor | 0.656 | [0.624, 0.688] |
| signalised 4+ leg | log_entering_single | 1.203 | [0.830, 1.576] |
| signalised 4+ leg | single_road | -1.535 | [-5.159, 2.089] |
| signalised 4+ leg | streetcar_20m | 0.087 | [0.042, 0.132] |
| signalised 4+ leg | min_leg_angle_10deg | 0.065 | [0.051, 0.079] |
| signalised 4+ leg | legs_5+ | 0.393 | [0.271, 0.515] |
| signalised 4+ leg | class_combo_arterial x arterial | 0.239 | [0.170, 0.308] |
| signalised 4+ leg | class_combo_arterial x collector | -0.036 | [-0.087, 0.016] |
| signalised 4+ leg | class_combo_collector x collector | -0.311 | [-0.416, -0.206] |
| signalised 4+ leg | class_combo_collector x local | -0.370 | [-0.449, -0.291] |
| signalised 4+ leg | class_combo_highway x any | -0.373 | [-0.476, -0.270] |
| signalised 4+ leg | class_combo_local x local | -0.535 | [-0.815, -0.254] |
| signalised 4+ leg | k (overdispersion) | 0.528 | [0.510, 0.547] |
| unsignalised | const | -5.777 | [-5.947, -5.607] |
| unsignalised | log_major | 0.318 | [0.297, 0.339] |
| unsignalised | log_minor | 0.505 | [0.486, 0.525] |
| unsignalised | log_entering_single | 0.483 | [0.448, 0.517] |
| unsignalised | single_road | 1.118 | [0.841, 1.395] |
| unsignalised | streetcar_20m | 0.471 | [0.414, 0.529] |
| unsignalised | min_leg_angle_10deg | -0.054 | [-0.060, -0.049] |
| unsignalised | legs_4 | 0.139 | [0.112, 0.165] |
| unsignalised | legs_5+ | -0.160 | [-0.429, 0.108] |
| unsignalised | class_combo_arterial x arterial | -1.428 | [-1.518, -1.338] |
| unsignalised | class_combo_arterial x collector | 0.586 | [0.471, 0.702] |
| unsignalised | class_combo_arterial x local | 0.707 | [0.646, 0.768] |
| unsignalised | class_combo_collector x collector | -0.077 | [-0.153, -0.001] |
| unsignalised | class_combo_collector x local | 0.224 | [0.185, 0.263] |
| unsignalised | class_combo_highway x any | -1.286 | [-1.382, -1.190] |
| unsignalised | k (overdispersion) | 1.809 | [1.775, 1.843] |

## Citywide level by year

Observed clean crashes / sum of base predictions. `spf_forecast` for year t uses the level of t - 1.

| year | level |
|---|---:|
| 2014 | 0.874 |
| 2015 | 0.954 |
| 2016 | 1.007 |
| 2017 | 1.047 |
| 2018 | 1.045 |
| 2019 | 1.062 |
| 2020 | 0.596 |
| 2021 | 0.584 |
| 2022 | 0.791 |
| 2023 | 0.927 |
| 2024 | 0.953 |
| 2025 | 0.924 |

## The SPF not chosen

| SPF | rows | AIC | k | reference levels | major elasticity [95% CI] | major in 0.3-0.9 | minor elasticity [95% CI] | minor in 0.3-0.9 |
|---|---:|---:|---:|---|---|---|---|---|
| pooled | 134,424 | 379,523.868 | 1.514 | legs_3-, class_combo_local x local | 0.296 [0.278, 0.315] | no | 0.595 [0.579, 0.611] | yes |
