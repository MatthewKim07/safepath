# Empirical Bayes (CP6)

EB_t = SPF_t * (w + (1 - w) * observed / SPF over the history window), w = 1 / (1 + k * SPF summed over the window). History uses only years before t (tests/test_eb.py perturbs later years and checks EB_t is unchanged).

## History window (chosen on validation 2024)

| history window (years) | Poisson deviance val 2024 | Spearman val 2024 | median weight on SPF |
|---:|---:|---:|---:|
| 3 | 1.1672 | 0.7101 | 0.2847 |
| 5 | 1.1433 | 0.7250 | 0.1879 |
| 8 | 1.1444 | 0.7362 | 0.1149 |

Chosen: **5 years**.

## Weight on the SPF, 2026 estimates

Median 0.180; p10 0.035; p90 0.374. Busy intersections lean on their own history (small w); quiet ones on the SPF.
