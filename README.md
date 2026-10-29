# SafePath

**Crash-risk-aware routing for Toronto.** SafePath learns where police-reported collisions
concentrate on the road network, scores every intersection and road segment, and routes with
`cost = travel_time + λ · risk`. A single slider moves between the fastest route and the
safest one, and the app shows exactly what the safer route costs in minutes.

![SafePath app: fastest (gray) vs safest (green) route from Union Station to Yorkdale at λ = 2, with the top 5% riskiest intersections](reports/figures/app_lambda2.png)

The goal is a rigorous, honest ML pipeline rather than a flashy number: temporal holdouts,
strong baselines, bootstrap confidence intervals, a unit-tested guarantee against label
leakage, and a written account of where the model and the data fall short. No LLMs are used
anywhere in the system.

## How it works

```mermaid
flowchart LR
    A[TPS collision records<br/>841k rows, 2014-2026] --> B[Clean<br/>dedupe, drop missing coords]
    C[OpenStreetMap<br/>Toronto drive graph] --> D[Consolidate intersections<br/>22,435 nodes]
    B --> E[Snap crashes to intersections<br/>+ proxy exclusion]
    D --> E
    E --> F[Intersection-year labels<br/>+ static and history features]
    F --> G[LightGBM Poisson<br/>vs baselines]
    G --> H[2026 risk scoring<br/>intersections + segments]
    H --> I[λ-weighted routing<br/>Streamlit app]
```

1. **Collisions.** 841,092 Toronto Police Service records are downloaded from the open-data
   FeatureServer (paginated, cached, verified against the service's own count), cleaned, and
   deduplicated. 136,730 records (16.3%) have no location and are dropped.
2. **Road network.** The OSM drive graph is projected to UTM 17N and its intersections are
   consolidated (nodes within 20 m merge), so a divided-road junction drawn as four OSM nodes
   becomes one intersection.
3. **Snapping.** Each crash point is assigned to an intersection (≤ 30 m), a mid-block
   segment (≤ 20 m from an edge), or excluded (inside a parking lot or off-network).
4. **Labels and features.** One row per intersection per year: crash count, severe count, and
   features from OSM (approach-road classes, legs, skew, speed limits, lanes, signals, stop
   signs, crossings, streetcar tracks, bus stops, nearby activity, distance to motorways and
   malls) plus crash history drawn only from earlier years.
5. **Model.** LightGBM with a Poisson objective, early-stopped on 2024 and tested on 2025,
   compared against three baselines.
6. **Risk and routing.** Each intersection gets a predicted 2026 crash count; road segments
   get a per-km rate by road class. A directed edge carries the risk of the intersection it
   enters (charged once per intersection) plus its segment risk. Dijkstra on
   `travel_time + λ · k · risk` gives the route for any λ.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m safepath all     # download, build the graph, train, evaluate, score (10-15 min, mostly OSM downloads)
python -m safepath app     # open the Streamlit app
```

Individual steps can be run on their own: `download`, `clean`, `report-data`, `graph`,
`locations`, `snap`, `report-snap`, `features`, `train`, `evaluate`, `score`, `route`, `app`.
Network fetches retry with backoff and then fail with a non-zero exit; `all` stops at the
first failing step. Configuration lives in `configs/default.yaml`; `pytest` runs the test
suite.

The app accepts query parameters for shareable views, e.g.
`?origin=Union+Station&destination=Yorkdale&lam=2`.

## Data and design decisions

**Intersections are the unit of analysis.** TPS places collisions on a small set of points:
704,362 located crashes sit on only 18,651 distinct coordinates. Against the consolidated OSM
graph, 96.3% of crashes fall within 30 m of an intersection (median 2 m), and only 3.5% sit
mid-block, a large share of them beside expressways. Modelling segments would mean training
on approximate locations, so the model works at intersection level and segments get simple
class-level rates. Details: [`reports/location_analysis.md`](reports/location_analysis.md).

**Consolidation tolerance.** OSMnx tolerances of 10, 15 and 20 m were compared. 10 m (nodes
within 20 m merge) keeps 91.6% of crashes within 10 m of their intersection; larger
tolerances pull cluster centres away from where TPS places crashes and start merging
genuinely separate intersections.

**Proxy exclusions.** Some intersections inherit crashes that happened elsewhere, and both
patterns were flagged with explicit rules computed from training years only:

- *Retail proxies (23):* parking-lot crashes geocoded to the nearest public intersection.
  These have ≥ 150 train crashes, ≤ 8% severe, ≥ 25% hit-and-run, and lie within 150 m of a
  mall or large retail area. Examples: Highland Hill by Yorkdale, Borough / Progress at
  Scarborough Town Centre, Fairview, Sherway.
- *Expressway proxies (8):* expressway crashes placed at a nearby surface intersection.
  These lie within 200 m of a motorway and have ≥ 20× the median crashes for their road-class
  combination. Examples: residential corners beside the DVP with 1,000+ crashes.

Flagged intersections are excluded from training and headline evaluation, and at
scoring time they get a features-only estimate instead of their inflated counts. The full
list with reasons is in
[`reports/excluded_intersections.csv`](reports/excluded_intersections.csv).

**Motorway base rate.** Motorways have no intersections to carry risk, so they get a per-km
rate: the yearly excess crashes at expressway-proxy intersections (actual minus the
features-only expectation) spread over all motorway kilometres, 1.15 crashes/km/yr. Other
road classes get per-km rates from mid-block crashes. Intersections carry 96% of total
scored risk. Details: [`reports/risk.md`](reports/risk.md).

**Temporal splits.** Train label years 2016–2019 and 2022–2023, validation 2024, test 2025.
2020–2021 are used only as history (with a COVID-window indicator, and history features
also expressed as a share of the citywide total to absorb the level shift). 2026 is a
partial year and is never a label.

## Results

### Risk model (test year 2025, 22,404 intersections)

| model | Spearman [95% CI] | PR-AUC | ROC-AUC | recall@5% [95% CI] | Poisson dev. |
|---|---|---:|---:|---|---:|
| Prior 3-year mean | 0.735 [0.727, 0.743] | 0.839 | 0.872 | 0.553 [0.538, 0.567] | 1.125 |
| Persistence (train-period mean per intersection) | 0.741 [0.734, 0.748] | 0.858 | 0.889 | 0.544 [0.528, 0.558] | 1.193 |
| Road-class mean (no history) | 0.396 [0.384, 0.408] | 0.555 | 0.669 | 0.232 [0.219, 0.247] | 5.315 |
| **LightGBM (static + history)** | 0.740 [0.734, 0.747] | **0.879** | **0.898** | 0.554 [0.539, 0.568] | **1.027** |
| LightGBM static only, unseen 2 km blocks | 0.628 [0.619, 0.636] | 0.794 | 0.829 | 0.478 [0.459, 0.495] | 2.531 |

Recall@5% is the share of next-year crashes that occur at the 5% of intersections the model
ranks highest. CIs come from 1,000 bootstrap resamples of intersections.

What this says, plainly:

- **On ranking, LightGBM ties a strong persistence baseline.** The paired Spearman
  difference is −0.000 [−0.004, +0.003]. It does capture slightly more crashes in its top 5%
  (+1.0 point, CI [+0.6, +1.5]), and it is better calibrated (lower Poisson deviance, higher
  PR-AUC) because it predicts small non-zero risk where persistence predicts exactly zero.
  Recent crash history carries over 90% of the model's gain.
- **Road features alone transfer.** Scored on spatial blocks it never saw, the static-only
  model reaches Spearman 0.628, against 0.396 for a road-class lookup. That is the relevant
  number for intersections with no crash record and for other cities. When the same model is
  scored on intersections it trained on, it reaches 0.728, because continuous features act as
  an intersection fingerprint; that number overstates transfer and is not used.
- Full tables, the severe-crash model, and validation-year results:
  [`reports/metrics.md`](reports/metrics.md).

![Calibration by prediction decile](reports/figures/calibration.png)

### Routing trade-off (200 random trips, 3–25 km)

`k` = 6.8 seconds per (crash/yr), chosen so that at λ = 1 the median route's risk cost equals
its median travel time.

| λ | median extra min | p90 extra min | median risk reduction | p90 risk reduction |
|---:|---:|---:|---:|---:|
| 0.5 | 3.1 | 7.1 | 67% | 83% |
| 1 | 4.9 | 9.7 | 72% | 85% |
| 2 | 6.4 | 13.0 | 75% | 86% |

"Risk" here is the sum of expected crashes per year at the locations a route passes. The
reduction means the route avoids crash-prone places; it is not a reduction in an individual
driver's crash probability (see Limitations).

![Pareto curve, UofT St. George to the Pearson area](reports/figures/pareto_uoft_st_george_pearson_area.png)

More sweeps: [`reports/routing.md`](reports/routing.md).

## Limitations

- **No exposure normalisation yet.** Risk is expected crashes at a location, not a rate per
  vehicle. Busy arterials look risky partly because they carry more traffic, so the "safe"
  route shifts traffic toward quieter streets: residential share of route kilometres rises
  from 3% at λ = 0 to 21% at λ = 1.
- **Motorways are undercounted.** Most 400-series highways in Toronto (401, 427, 400, 404) are
  patrolled by the Ontario Provincial Police, so their crashes are largely absent from TPS
  data, and the motorway rate is a lower bound. Combined with free-flow travel times, some
  safer routes take long expressway detours (Union Station to Yorkdale more than doubles in
  distance at λ ≥ 0.75).
- **Police-reported data with non-random gaps.** 16.3% of records have no location, and
  missing locations are more common for property-damage-only collisions (16.8%) than for
  fatal ones (4.3%).
- **Intersection-snapped locations.** TPS offsets crashes to intersections, so mid-block risk
  is only coarsely represented.
- **Current OSM snapshot.** Road features describe today's network, not the network in each
  label year.
- **Heuristic proxy rules.** The retail and expressway rules are documented and conservative,
  but they are heuristics; near-misses are listed in [`reports/labels.md`](reports/labels.md).

## Roadmap

- **Phase 2: exposure and calibration.** Toronto traffic-volume counts for exposure; a
  Safety Performance Function (negative binomial on road features and exposure) blended with
  each intersection's history by Empirical Bayes; calibration checks; SHAP explanations;
  experiment tracking with Weights & Biases.
- **Phase 3: generalisation.** A graph neural network on the road graph against LightGBM; a
  cross-city test on Ottawa collision data; a London, Ontario demo using the static-only / SPF
  model.
- **Phase 4: product.** A deployed demo, a model card, and time- and weather-aware risk.

## Repository layout

```
configs/default.yaml        pipeline configuration (paths, thresholds, splits, proxy rules)
src/safepath/data/          download, cleaning, data report
src/safepath/graph.py       OSM graph, consolidated intersections, segment view
src/safepath/snapping.py    crash-to-intersection / segment assignment
src/safepath/features/      labels, proxy rules, static and history features
src/safepath/models/        baselines and LightGBM training
src/safepath/evaluate.py    metrics, bootstrap CIs, figures
src/safepath/risk.py        2026 risk scoring onto the routing graph
src/safepath/routing.py     λ-weighted routing, k calibration, Pareto sweeps
src/safepath/app/           Streamlit app
reports/                    generated reports and figures
tests/                      unit tests (leakage, routing, snapping, metrics, ...)
```

## Data attribution

- Collision data: Toronto Police Service, Public Safety Data Portal, *Traffic Collisions*,
  used under the [Open Government Licence – Toronto](https://open.toronto.ca/open-data-license/).
- Road network, points of interest and basemap tiles: © [OpenStreetMap](https://www.openstreetmap.org/copyright)
  contributors, available under the Open Database License (ODbL).

## License

MIT, see [LICENSE](LICENSE).
