"""Single + batch episode runners.

Runs every (policy, condition, seed) combination with metric capture, with
read-only access to the model and the env's reward / dynamics / termination.
The GP adapter is the sole switch between RL-only and live-GP modes.
"""

from __future__ import annotations

import random
from typing import Iterable

from environment import CleaningEnv

from .eval_config import EvalConfig
from .eval_env import (
    ROOM_REGIONS,
    EvalEnv,
    _EnvOverrides,
    patch_env_constants,
)
from .gp_adapter import GPAdapter
from .metrics import (
    EpisodeMetrics,
    finalize_metrics,
    update_metrics,
)
from .policies import Policy


def _push_gp_belief(env: EvalEnv, adapter: GPAdapter) -> None:
    mu, sigma = adapter(env.obs_map, env.Dirt_Map, (env.pos_r, env.pos_c), env.t)
    env.update_gp_belief(mu, sigma)


def run_episode(
    eval_env: EvalEnv,
    policy: Policy,
    adapter: GPAdapter,
    *,
    seed: int,
    region: str | None,
    gp_update_every: int,
    capture_steps: bool,
    rng: random.Random,
) -> EpisodeMetrics:
    """Run a single episode and return its metrics.

    ``eval_env.reset()`` handles spawn-mode forcing and (for all_rooms) the
    post-reset position override internally.  ``EvalEnv.step()`` injects
    ground-truth values into the info dict for the metric collector.
    """
    obs, _ = eval_env.reset(seed=seed, options={"region": region, "rng": rng})
    adapter.reset(seed)
    policy.reset(seed)
    _push_gp_belief(eval_env, adapter)  # initial GP belief before first action

    metrics = EpisodeMetrics(
        seed=seed,
        policy=policy.name,
        spawn_mode=eval_env._cfg.spawn_mode,
        region=region,
        gp_adapter_name=adapter.name,
    )

    while True:
        action = policy.act(obs, eval_env)
        prev_pos = (eval_env.pos_r, eval_env.pos_c)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        update_metrics(
            metrics,
            env=eval_env,
            info=info,
            reward=reward,
            action=action,
            prev_pos=prev_pos,
            keep_step_record=capture_steps,
        )
        if gp_update_every >= 1 and eval_env.t % gp_update_every == 0:
            _push_gp_belief(eval_env, adapter)
        if terminated or truncated:
            return finalize_metrics(
                metrics,
                env=eval_env,
                info=info,
                terminated=terminated,
                truncated=truncated,
                final_reward=float(reward),
            )


def _conditions(cfg: EvalConfig) -> list[str | None]:
    """Return the list of regions to evaluate.

    For all_rooms: one entry per room region.
    For fixed / random: a single ``None`` entry.
    """
    if cfg.spawn_mode != "all_rooms":
        return [None]
    return list(ROOM_REGIONS.keys())


def run_suite(
    cfg: EvalConfig,
    policies: Iterable[Policy],
    adapter: GPAdapter,
) -> list[EpisodeMetrics]:
    """Run every (policy × condition × seed) combination and return the flat
    list of episode metrics.
    """
    overrides = _EnvOverrides(
        max_steps=cfg.max_steps,
        sensor_noise_sd=cfg.sensor_noise_sd,
        drift_alpha=cfg.drift_alpha,
    )

    all_episodes: list[EpisodeMetrics] = []
    rng = random.Random(cfg.seed_base)

    with patch_env_constants(overrides):
        eval_env = EvalEnv(CleaningEnv(), cfg)
        conditions = _conditions(cfg)

        if cfg.spawn_mode == "all_rooms":
            per_region = max(1, cfg.n_episodes // len(conditions))
        else:
            per_region = cfg.n_episodes

        for policy in policies:
            for region in conditions:
                seeds_for_cond = _seeds_for_condition(
                    cfg.seed_base, per_region, region
                )
                rep_quota = cfg.representative_episodes
                for i, seed in enumerate(seeds_for_cond):
                    capture = bool(cfg.capture_steps) or (i < rep_quota)
                    metrics = run_episode(
                        eval_env=eval_env,
                        policy=policy,
                        adapter=adapter,
                        seed=seed,
                        region=region,
                        gp_update_every=cfg.gp_update_every,
                        capture_steps=capture,
                        rng=rng,
                    )
                    all_episodes.append(metrics)
        eval_env.close()
    return all_episodes


def _seeds_for_condition(
    seed_base: int, n: int, region: str | None
) -> list[int]:
    """Region offset keeps seeds disjoint across regions for reproducibility."""
    region_offsets = {
        None: 0,
        "bedroom_a": 100_000,
        "bedroom_b": 200_000,
        "living":    300_000,
    }
    offset = region_offsets.get(region, 0)
    return list(range(seed_base + offset, seed_base + offset + n))
