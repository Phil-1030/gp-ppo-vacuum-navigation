# Evaluation report — v4_gp_pure_vs_greedy

- Model: `runs/ppo_gp_v4/best_model.zip`
- Run root: `runs/ppo_gp_v4`
- Episodes: 100
- Spawn mode: `fixed`
- GP adapter: `live:eval.gp_adapter:LiveGPAdapter`

## Plain-language summary

Trained policy `ppo:best_model` achieves a mean clean ratio of **0.912** (mean ep_length 906.5). **62 / 100** episodes ended in SUCCESS and **4 / 100** ended in DEAD. The charger was visited in **98%** of episodes (mean visits/episode: 12.26).

## Aggregate metrics per policy

| policy | n_episodes | ep_length_mean | ep_return_extrinsic_mean | clean_ratio_mean | maintenance_ratio_mean | peak_clean_ratio_mean | coverage_rate_mean | n_charger_visits_mean | success_rate | death_rate | truncation_rate | charger_visit_rate |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ppo:best_model | 100 | 906.530 | 2455.724 | 0.912 | 0.738 | 0.941 | 0.912 | 12.260 | 0.620 | 0.040 | 0.340 | 0.980 |
| greedy_dirt | 100 | 544.570 | 402.620 | 0.757 | 0.624 | 0.815 | 0.749 | 1.830 | 0.000 | 1.000 | 0.000 | 0.360 |

## Pairwise tests vs `ppo:best_model`
Significance threshold (Bonferroni-corrected): **p < 0.005**.

| metric | vs | mean(target) | mean(other) | diff | 95% CI | t/z | p (Welch/z) | p (Mann-Whitney) | sig |
|--------|----|--------------|-------------|------|--------|-----|-------------|------------------|-----|
| ep_return_extrinsic | greedy_dirt | 2455.724 | 402.620 | +2053.103 | [+1862.801, +2237.437] | +21.35 | 0.0000 | 0.0000 | ✓ |
| clean_ratio_final | greedy_dirt | 0.912 | 0.757 | +0.155 | [+0.126, +0.184] | +10.58 | 0.0000 | 0.0000 | ✓ |
| n_charger_visits | greedy_dirt | 12.260 | 1.830 | +10.430 | [+8.730, +12.160] | +11.84 | 0.0000 | 0.0000 | ✓ |
| success_rate | greedy_dirt | 0.620 | 0.000 | +0.620 | — | +9.48 | 0.0000 | — | ✓ |
