# Evaluation report — bc_baseline

- Model: `runs/ppo_gp_v5/best_model.zip`
- Run root: `runs/final_showdown`
- Episodes: 100
- Spawn mode: `fixed`
- GP adapter: `live:eval.gp_adapter:LiveGPAdapter`

## Plain-language summary

Trained policy `ppo:best_model` achieves a mean clean ratio of **0.949** (mean ep_length 421.6). **91 / 100** episodes ended in SUCCESS and **9 / 100** ended in DEAD. The charger was visited in **99%** of episodes (mean visits/episode: 1.62).

## Aggregate metrics per policy

| policy | n_episodes | ep_length_mean | ep_return_extrinsic_mean | clean_ratio_mean | maintenance_ratio_mean | peak_clean_ratio_mean | coverage_rate_mean | n_charger_visits_mean | success_rate | death_rate | truncation_rate | charger_visit_rate |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ppo:best_model | 100 | 421.580 | 2850.316 | 0.949 | 0.699 | 0.950 | 0.887 | 1.620 | 0.910 | 0.090 | 0.000 | 0.990 |
| random | 100 | 629.920 | -153.054 | 0.394 | 0.276 | 0.458 | 0.440 | 5.340 | 0.000 | 1.000 | 0.000 | 0.910 |
| greedy_dirt | 100 | 544.570 | 402.620 | 0.757 | 0.624 | 0.815 | 0.749 | 1.830 | 0.000 | 1.000 | 0.000 | 0.360 |
| lr:lr_model | 100 | 500.330 | 87.617 | 0.583 | 0.330 | 0.674 | 0.608 | 0.090 | 0.000 | 1.000 | 0.000 | 0.090 |
| svm:svm_model | 100 | 499.260 | 110.873 | 0.632 | 0.323 | 0.714 | 0.647 | 0.020 | 0.010 | 0.990 | 0.000 | 0.020 |

## Pairwise tests vs `ppo:best_model`
Significance threshold (Bonferroni-corrected): **p < 0.005**.

| metric | vs | mean(target) | mean(other) | diff | 95% CI | t/z | p (Welch/z) | p (Mann-Whitney) | sig |
|--------|----|--------------|-------------|------|--------|-----|-------------|------------------|-----|
| ep_return_extrinsic | random | 2850.316 | -153.054 | +3003.371 | [+2849.206, +3138.460] | +40.33 | 0.0000 | 0.0000 | ✓ |
| clean_ratio_final | random | 0.949 | 0.394 | +0.554 | [+0.528, +0.582] | +39.94 | 0.0000 | 0.0000 | ✓ |
| n_charger_visits | random | 1.620 | 5.340 | -3.720 | [-4.680, -2.790] | -7.49 | 0.0000 | 0.0000 | ✓ |
| success_rate | random | 0.910 | 0.000 | +0.910 | — | +12.92 | 0.0000 | — | ✓ |
| ep_return_extrinsic | greedy_dirt | 2850.316 | 402.620 | +2447.696 | [+2293.531, +2583.413] | +33.01 | 0.0000 | 0.0000 | ✓ |
| clean_ratio_final | greedy_dirt | 0.949 | 0.757 | +0.191 | [+0.168, +0.215] | +15.93 | 0.0000 | 0.0000 | ✓ |
| n_charger_visits | greedy_dirt | 1.620 | 1.830 | -0.210 | [-0.940, +0.430] | -0.59 | 0.5519 | 0.0000 |  |
| success_rate | greedy_dirt | 0.910 | 0.000 | +0.910 | — | +12.92 | 0.0000 | — | ✓ |
| ep_return_extrinsic | lr:lr_model | 2850.316 | 87.617 | +2762.699 | [+2605.106, +2900.879] | +36.33 | 0.0000 | 0.0000 | ✓ |
| clean_ratio_final | lr:lr_model | 0.949 | 0.583 | +0.366 | [+0.326, +0.407] | +17.66 | 0.0000 | 0.0000 | ✓ |
| n_charger_visits | lr:lr_model | 1.620 | 0.090 | +1.530 | [+1.390, +1.670] | +21.16 | 0.0000 | 0.0000 | ✓ |
| success_rate | lr:lr_model | 0.910 | 0.000 | +0.910 | — | +12.92 | 0.0000 | — | ✓ |
| ep_return_extrinsic | svm:svm_model | 2850.316 | 110.873 | +2739.443 | [+2570.493, +2891.635] | +33.60 | 0.0000 | 0.0000 | ✓ |
| clean_ratio_final | svm:svm_model | 0.949 | 0.632 | +0.316 | [+0.276, +0.358] | +14.99 | 0.0000 | 0.0000 | ✓ |
| n_charger_visits | svm:svm_model | 1.620 | 0.020 | +1.600 | [+1.470, +1.730] | +23.60 | 0.0000 | 0.0000 | ✓ |
| success_rate | svm:svm_model | 0.910 | 0.010 | +0.900 | — | +12.77 | 0.0000 | — | ✓ |
