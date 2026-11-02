# Evaluation report — v5_gp_pure_vs_greedy

- Model: `runs/ppo_gp_v5/best_model.zip`
- Run root: `runs/ppo_gp_v5`
- Episodes: 100
- Spawn mode: `fixed`
- GP adapter: `live:eval.gp_adapter:LiveGPAdapter`

## Plain-language summary

Trained policy `ppo:best_model` achieves a mean clean ratio of **0.949** (mean ep_length 421.6). **91 / 100** episodes ended in SUCCESS and **9 / 100** ended in DEAD. The charger was visited in **99%** of episodes (mean visits/episode: 1.62).

## Aggregate metrics per policy

| policy | n_episodes | ep_length_mean | ep_return_extrinsic_mean | clean_ratio_mean | maintenance_ratio_mean | peak_clean_ratio_mean | coverage_rate_mean | n_charger_visits_mean | success_rate | death_rate | truncation_rate | charger_visit_rate |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ppo:best_model | 100 | 421.580 | 2850.316 | 0.949 | 0.699 | 0.950 | 0.887 | 1.620 | 0.910 | 0.090 | 0.000 | 0.990 |
| greedy_dirt | 100 | 544.570 | 402.620 | 0.757 | 0.624 | 0.815 | 0.749 | 1.830 | 0.000 | 1.000 | 0.000 | 0.360 |

## Pairwise tests vs `ppo:best_model`
Significance threshold (Bonferroni-corrected): **p < 0.005**.

| metric | vs | mean(target) | mean(other) | diff | 95% CI | t/z | p (Welch/z) | p (Mann-Whitney) | sig |
|--------|----|--------------|-------------|------|--------|-----|-------------|------------------|-----|
| ep_return_extrinsic | greedy_dirt | 2850.316 | 402.620 | +2447.696 | [+2293.531, +2583.413] | +33.01 | 0.0000 | 0.0000 | ✓ |
| clean_ratio_final | greedy_dirt | 0.949 | 0.757 | +0.191 | [+0.168, +0.215] | +15.93 | 0.0000 | 0.0000 | ✓ |
| n_charger_visits | greedy_dirt | 1.620 | 1.830 | -0.210 | [-0.940, +0.430] | -0.59 | 0.5519 | 0.0000 |  |
| success_rate | greedy_dirt | 0.910 | 0.000 | +0.910 | — | +12.92 | 0.0000 | — | ✓ |
