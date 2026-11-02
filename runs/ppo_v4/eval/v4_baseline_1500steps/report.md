# Evaluation report — v4_baseline_1500steps

- Model: `runs/ppo_v4/best_model.zip`
- Run root: `runs/ppo_v4`
- Episodes: 100
- Spawn mode: `fixed`
- GP adapter: `stub`

## Plain-language summary

Trained policy `ppo:best_model` achieves a mean clean ratio of **0.939** (mean ep_length 439.1). **70 / 100** episodes ended in SUCCESS and **30 / 100** ended in DEAD. The charger was visited in **72%** of episodes (mean visits/episode: 0.91).

## Aggregate metrics per policy

| policy | n_episodes | ep_length_mean | ep_return_extrinsic_mean | clean_ratio_mean | maintenance_ratio_mean | peak_clean_ratio_mean | coverage_rate_mean | n_charger_visits_mean | success_rate | death_rate | truncation_rate | charger_visit_rate |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ppo:best_model | 100 | 439.140 | 2318.924 | 0.939 | 0.702 | 0.948 | 0.879 | 0.910 | 0.700 | 0.300 | 0.000 | 0.720 |
| greedy_dirt | 100 | 544.570 | 402.621 | 0.757 | 0.624 | 0.815 | 0.749 | 1.830 | 0.000 | 1.000 | 0.000 | 0.360 |

## Pairwise tests vs `ppo:best_model`
Significance threshold (Bonferroni-corrected): **p < 0.005**.

| metric | vs | mean(target) | mean(other) | diff | 95% CI | t/z | p (Welch/z) | p (Mann-Whitney) | sig |
|--------|----|--------------|-------------|------|--------|-----|-------------|------------------|-----|
| ep_return_extrinsic | greedy_dirt | 2318.924 | 402.621 | +1916.303 | [+1689.564, +2143.774] | +16.39 | 0.0000 | 0.0000 | ✓ |
| clean_ratio_final | greedy_dirt | 0.939 | 0.757 | +0.182 | [+0.159, +0.206] | +14.88 | 0.0000 | 0.0000 | ✓ |
| n_charger_visits | greedy_dirt | 0.910 | 1.830 | -0.920 | [-1.640, -0.270] | -2.60 | 0.0093 | 0.0565 |  |
| success_rate | greedy_dirt | 0.700 | 0.000 | +0.700 | — | +10.38 | 0.0000 | — | ✓ |
