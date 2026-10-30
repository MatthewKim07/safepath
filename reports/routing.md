# Routing trade-off (CP6: per-vehicle risk)

Edge cost = travel_time (s, free-flow) + λ · k · risk, where risk is the expected number of crashes for one vehicle driving the edge, per million vehicles (reports/risk.md). A route's risk is therefore expected crashes per million trips. Routing is turn-aware: a U-turn (driving back along the edge just used, or turning more than 150° back onto the same street through an intersection) costs 60 s.

## Scaling constant k

k = **31.1 seconds per (crash per million trips)**, the fixed point of k = median travel time / median route risk on λ = 1 routes over 200 random origin/destination pairs (3-25 km apart, seed 7). Check: median risk cost / median travel time on the final λ = 1 routes = 1.000.

| iteration | k | on |
|---:|---:|---|
| 0 | 21.46 | fastest routes |
| 1 | 30.90 | lambda=1 routes |
| 2 | 31.11 | lambda=1 routes |

## Trade-off over 200 random pairs

| lam | median extra min | p90 extra min | median % risk reduction | p90 % risk reduction | median detour (km ratio) | p90 detour (km ratio) | motorway share | residential share | routes with a U-turn |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | 0.03 | 0.00 |
| 0.50 | 0.58 | 2.30 | 20.99 | 45.98 | 1.02 | 1.28 | 0.57 | 0.06 | 0.01 |
| 1.00 | 1.45 | 3.62 | 26.54 | 50.17 | 1.04 | 1.44 | 0.58 | 0.08 | 0.03 |
| 2.00 | 2.04 | 5.09 | 28.73 | 52.84 | 1.07 | 1.52 | 0.60 | 0.08 | 0.03 |

Mean routing time: 15 ms per route.

## CP4 vs CP6

CP4 routed on expected crashes per year at the locations a route passes (no exposure). The '% risk reduction' columns are each version's own risk measure, so they are not directly comparable; times, detours and road-class shares are.

| λ | version | median extra min | p90 extra min | median % risk reduction | median detour (km ratio) | p90 detour (km ratio) | motorway share | residential share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | CP4 (count-based) | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | 0.03 |
| 0 | CP6 (per-vehicle) | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | 0.03 |
| 0.5 | CP4 (count-based) | 3.14 | 7.12 | 67.48 | 1.11 | 1.53 | 0.59 | 0.18 |
| 0.5 | CP6 (per-vehicle) | 0.58 | 2.30 | 20.99 | 1.02 | 1.28 | 0.57 | 0.06 |
| 1 | CP4 (count-based) | 4.94 | 9.70 | 72.06 | 1.20 | 1.65 | 0.59 | 0.21 |
| 1 | CP6 (per-vehicle) | 1.45 | 3.62 | 26.54 | 1.04 | 1.44 | 0.58 | 0.08 |
| 2 | CP4 (count-based) | 6.36 | 13.04 | 74.73 | 1.26 | 1.81 | 0.58 | 0.23 |
| 2 | CP6 (per-vehicle) | 2.04 | 5.09 | 28.73 | 1.07 | 1.52 | 0.60 | 0.08 |

- Residential share at λ = 1: 21% in CP4, 8% now (fastest routes: 3% / 3%).
- Motorway share at λ = 1: 59% in CP4, 58% now (fastest routes: 50% / 50%).

## Case study: Union Station → Yorkdale at λ = 2

![Before and after](figures/before_after_union_yorkdale.png)

| route | km | minutes | expected crashes per million trips | motorway km |
|---|---:|---:|---:|---:|
| fastest | 12.6 | 14.9 | 82.2 | 3.2 |
| CP4 safest (count-based) | 37.9 | 26.4 | 54.1 | 35.0 |
| CP6 safest (per-vehicle) | 30.1 | 19.9 | 28.8 | 29.3 |

All three are scored with CP6 per-vehicle risk. The CP4 route ends at the node CP4 snapped Yorkdale to (a turning circle on Highland Hill, behind the mall).

**The Yorkdale hook.** CP4 snapped the Yorkdale preset (the middle of the mall) to the nearest graph node: a turning circle at the end of Highland Hill, a residential cul-de-sac behind the mall. Every route therefore left the 401 at Dufferin and wound south through Jane Osler, Groton, Bentworth and Ranee before turning back north into the cul-de-sac, which is the hook. It was a snapping artifact, not a routing one: the route had no repeated nodes. Fix: snap like a navigation app (never to a dead end; a point more than 100 m from any street snaps to the nearest arterial or collector access if it is within 150 m of the nearest street), so Yorkdale now snaps to Yorkdale Road. Separately, node-based routing could not see U-turns through divided-road intersections, which per-vehicle routing uses as a cheap way to dodge a risky intersection; routing now runs over turn movements with a U-turn penalty (tests: no immediate reversals; a median U-turn loses to a 30 s detour).

## Sensitivity: is the behaviour robust to the MTO/TPS comparability assumption?

Freeway segment rates (400-series from MTO, city expressways from TPS) scaled by 0.5 and 2; Gardiner/DVP/Allen volumes ±30% (their per-vehicle rate scales by 1/1.3 and 1/0.7). k stays at its baseline value.

| variant | λ | median extra min | median % risk reduction | median detour (km ratio) | motorway share | residential share |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.50 | 0.58 | 20.99 | 1.02 | 0.57 | 0.06 |
| baseline | 1.00 | 1.45 | 26.54 | 1.04 | 0.58 | 0.08 |
| baseline | 2.00 | 2.04 | 28.73 | 1.07 | 0.60 | 0.08 |
| freeway rates x0.5 | 0.50 | 0.59 | 23.91 | 1.02 | 0.59 | 0.06 |
| freeway rates x0.5 | 1.00 | 1.43 | 30.30 | 1.05 | 0.62 | 0.07 |
| freeway rates x0.5 | 2.00 | 2.13 | 32.93 | 1.10 | 0.63 | 0.08 |
| freeway rates x2 | 0.50 | 0.68 | 19.91 | 1.02 | 0.54 | 0.07 |
| freeway rates x2 | 1.00 | 1.69 | 23.30 | 1.04 | 0.54 | 0.09 |
| freeway rates x2 | 2.00 | 2.95 | 27.38 | 1.10 | 0.52 | 0.10 |
| Gardiner/DVP/Allen volume +30% | 0.50 | 0.58 | 21.17 | 1.02 | 0.58 | 0.06 |
| Gardiner/DVP/Allen volume +30% | 1.00 | 1.44 | 27.15 | 1.04 | 0.59 | 0.08 |
| Gardiner/DVP/Allen volume +30% | 2.00 | 2.08 | 29.19 | 1.07 | 0.61 | 0.08 |
| Gardiner/DVP/Allen volume -30% | 0.50 | 0.60 | 20.24 | 1.02 | 0.56 | 0.07 |
| Gardiner/DVP/Allen volume -30% | 1.00 | 1.41 | 24.77 | 1.04 | 0.57 | 0.09 |
| Gardiner/DVP/Allen volume -30% | 2.00 | 2.17 | 27.56 | 1.08 | 0.58 | 0.09 |

| variant | Union→Yorkdale λ=2 km | minutes |
|---|---:|---:|
| baseline | 30.1 | 19.9 |
| freeway rates x0.5 | 29.9 | 19.7 |
| freeway rates x2 | 31.1 | 20.8 |
| Gardiner/DVP/Allen volume +30% | 30.1 | 19.9 |
| Gardiner/DVP/Allen volume -30% | 30.6 | 20.3 |

- freeway rates x0.5: motorway share 58% -> 62%, median extra minutes 1.45 -> 1.43, median risk reduction 27% -> 30% (λ = 1).
- freeway rates x2: motorway share 58% -> 54%, median extra minutes 1.45 -> 1.69, median risk reduction 27% -> 23% (λ = 1).
- Gardiner/DVP/Allen volume +30%: motorway share 58% -> 59%, median extra minutes 1.45 -> 1.44, median risk reduction 27% -> 27% (λ = 1).
- Gardiner/DVP/Allen volume -30%: motorway share 58% -> 57%, median extra minutes 1.45 -> 1.41, median risk reduction 27% -> 25% (λ = 1).
- Across all variants at λ = 1 the motorway share spans 8% (54%-62%), against 50% on the fastest routes.
- **Conclusion:** the direction of the behaviour does not depend on the MTO/TPS comparability assumption: halving or doubling every freeway rate still leaves safer routes leaning on freeways (motorway share above the fastest routes' in every variant), and moves extra time and risk reduction by a fraction of their baseline values. The size of the freeway lean does depend on it.

## Things to watch

- Free-flow travel times flatter long freeway detours; congestion would cut them.
- Per-vehicle routing minimises the driver's own risk. It can still send traffic through local streets when they are safer per vehicle, which shifts risk onto pedestrians and residents there (a neighbourhood externality the cost does not include).
- EB overpredicts the quietest intersections (lowest decile, reports/model_comparison.md), so per-vehicle risk on quiet local streets is if anything too high, which works against cut-throughs.

## Pareto sweeps

### union station yorkdale

![Pareto union_station_yorkdale](figures/pareto_union_station_yorkdale.png)

| lam | minutes | expected_crashes | km | extra_minutes | risk_reduction_pct |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 14.94 | 82.19 | 12.61 | 0.00 | 0.00 |
| 0.10 | 14.94 | 82.19 | 12.61 | 0.00 | 0.00 |
| 0.25 | 19.61 | 29.45 | 29.79 | 4.67 | 64.18 |
| 0.50 | 19.61 | 29.45 | 29.79 | 4.67 | 64.18 |
| 0.75 | 19.72 | 29.11 | 29.93 | 4.78 | 64.59 |
| 1.00 | 19.72 | 29.11 | 29.93 | 4.78 | 64.59 |
| 1.50 | 19.88 | 28.82 | 30.10 | 4.94 | 64.94 |
| 2.00 | 19.88 | 28.82 | 30.10 | 4.94 | 64.94 |
| 3.00 | 20.27 | 28.48 | 30.60 | 5.33 | 65.35 |
| 5.00 | 20.27 | 28.48 | 30.60 | 5.33 | 65.35 |
| 7.50 | 20.59 | 28.37 | 31.07 | 5.64 | 65.48 |
| 10.00 | 20.59 | 28.37 | 31.07 | 5.64 | 65.48 |

### high park scarborough town centre

![Pareto high_park_scarborough_town_centre](figures/pareto_high_park_scarborough_town_centre.png)

| lam | minutes | expected_crashes | km | extra_minutes | risk_reduction_pct |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 24.29 | 28.47 | 32.77 | 0.00 | 0.00 |
| 0.10 | 24.33 | 27.39 | 32.77 | 0.03 | 3.78 |
| 0.25 | 24.43 | 26.28 | 32.69 | 0.14 | 7.68 |
| 0.50 | 24.43 | 26.28 | 32.69 | 0.14 | 7.68 |
| 0.75 | 24.54 | 25.94 | 32.83 | 0.25 | 8.87 |
| 1.00 | 24.54 | 25.94 | 32.83 | 0.25 | 8.87 |
| 1.50 | 24.54 | 25.94 | 32.83 | 0.25 | 8.87 |
| 2.00 | 24.54 | 25.94 | 32.83 | 0.25 | 8.87 |
| 3.00 | 24.93 | 25.60 | 33.33 | 0.64 | 10.07 |
| 5.00 | 25.94 | 24.99 | 34.59 | 1.65 | 12.21 |
| 7.50 | 26.25 | 24.88 | 35.05 | 1.96 | 12.58 |
| 10.00 | 26.25 | 24.88 | 35.05 | 1.96 | 12.58 |

### uoft st george pearson area

![Pareto uoft_st_george_pearson_area](figures/pareto_uoft_st_george_pearson_area.png)

| lam | minutes | expected_crashes | km | extra_minutes | risk_reduction_pct |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 19.44 | 50.46 | 26.53 | 0.00 | 0.00 |
| 0.10 | 19.45 | 48.95 | 26.50 | 0.01 | 2.99 |
| 0.25 | 20.10 | 40.32 | 27.61 | 0.66 | 20.10 |
| 0.50 | 20.10 | 40.32 | 27.61 | 0.66 | 20.10 |
| 0.75 | 20.10 | 40.32 | 27.61 | 0.66 | 20.10 |
| 1.00 | 20.10 | 40.32 | 27.61 | 0.66 | 20.10 |
| 1.50 | 20.10 | 40.32 | 27.61 | 0.66 | 20.10 |
| 2.00 | 20.58 | 39.77 | 28.02 | 1.14 | 21.18 |
| 3.00 | 20.58 | 39.77 | 28.02 | 1.14 | 21.18 |
| 5.00 | 20.58 | 39.77 | 28.02 | 1.14 | 21.18 |
| 7.50 | 20.58 | 39.77 | 28.02 | 1.14 | 21.18 |
| 10.00 | 21.98 | 39.45 | 28.16 | 2.54 | 21.82 |
