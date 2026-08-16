"""0hecks for ball_launch.py 
    python -m latent_mj.envs.g1_tracking.train.test_ball_launch
"""

import jax
import jax.numpy as jnp
import numpy as np

import train_ball_launch as bl


def check_landing_accuracy():
    """Sampled launches (with the correct  per-episode k) should land within a
    few cm of their target (woth the drag correctio)."""
    rng = jax.random.PRNGKey(0)
    errors = []
    for i in range(30):
        rng, r_origin, r_target, r_time, r_k = jax.random.split(rng, 5)
        p0 = bl.sample_service_origin(r_origin)
        target_xy = bl.sample_landing_target(r_target)
        T = bl.sample_flight_time(r_time, p0, target_xy)
        k = jax.random.uniform(r_k, (), minval=0.01, maxval=0.04)

        v0 = bl.solve_launch_velocity(p0, target_xy, T, k)
        land_xy, _ = bl._forward_sim_landing(p0, v0, k)
        err = float(jnp.linalg.norm(target_xy - land_xy))
        errors.append(err)
    errors = np.array(errors)
    assert errors.mean() < 0.15, f"mean landing error {errors.mean():.3f}m too high"
    assert errors.max() < 0.5, f"worst-case landing error {errors.max():.3f}m too high"
    print(f"[PASS] check_landing_accuracy (mean={errors.mean():.4f}m, max={errors.max():.4f}m)")


def check_speed_range():
    """Resultant 3D launch speeds should mostly land in the paper's stated
    15-30 m/s band"""
    rng = jax.random.PRNGKey(1)
    speeds = []
    for i in range(50):
        rng, r_launch, r_k = jax.random.split(rng, 3)
        k = jax.random.uniform(r_k, (), minval=0.01, maxval=0.04)
        _, v0 = bl.sample_ball_launch(r_launch, k)
        speeds.append(float(jnp.linalg.norm(v0)))
    speeds = np.array(speeds)
    frac_in_range = np.mean((speeds >= 14.0) & (speeds <= 31.0))  # small tolerance at the edges
    assert frac_in_range > 0.9, (
        f"only {frac_in_range:.0%} of speeds in [14,31] m/s "
        f"(range: {speeds.min():.1f}-{speeds.max():.1f})"
    )
    print(f"[PASS] check_speed_range (range={speeds.min():.1f}-{speeds.max():.1f} m/s, "
          f"{frac_in_range:.0%} within [14,31])")


def check_k_mismatch_degrades_accuracy():
    """making k a required
    (not nominal-default) argument: solving with the wrong k should be
    substantially worse than solving with the correct k."""
    rng = jax.random.PRNGKey(2)
    errs_correct, errs_wrong = [], []
    nominal_k = 0.025
    for i in range(20):
        rng, r_origin, r_target, r_time, r_k = jax.random.split(rng, 5)
        p0 = bl.sample_service_origin(r_origin)
        target_xy = bl.sample_landing_target(r_target)
        T = bl.sample_flight_time(r_time, p0, target_xy)
        true_k = jax.random.uniform(r_k, (), minval=0.01, maxval=0.04)

        v_correct = bl.solve_launch_velocity(p0, target_xy, T, true_k)
        land_correct, _ = bl._forward_sim_landing(p0, v_correct, true_k)
        errs_correct.append(float(jnp.linalg.norm(target_xy - land_correct)))

        v_wrong = bl.solve_launch_velocity(p0, target_xy, T, nominal_k)
        land_wrong, _ = bl._forward_sim_landing(p0, v_wrong, true_k)
        errs_wrong.append(float(jnp.linalg.norm(target_xy - land_wrong)))

    mean_correct = np.mean(errs_correct)
    mean_wrong = np.mean(errs_wrong)
    assert mean_wrong > 5 * mean_correct, (
        f"expected k-mismatch to clearly degrade accuracy, got "
        f"correct={mean_correct:.3f}m wrong={mean_wrong:.3f}m"
    )
    print(f"[PASS] check_k_mismatch_degrades_accuracy "
          f"(correct-k mean err={mean_correct:.4f}m, wrong-k mean err={mean_wrong:.4f}m)")


def check_region_sampling():
    rng = jax.random.PRNGKey(3)
    for region, (lo, hi) in [("forecourt", bl.FORECOURT_X_RANGE), ("backcourt", bl.BACKCOURT_X_RANGE)]:
        for i in range(20):
            rng, sub = jax.random.split(rng)
            target = bl.sample_landing_target(sub, region=region)
            assert lo <= float(target[0]) <= hi, f"{region} target x={target[0]} outside [{lo},{hi}]"
    print("[PASS] check_region_sampling")


def check_jit_vmap_batch():
    N = 32
    rng = jax.random.PRNGKey(4)
    rngs = jax.random.split(rng, N)
    ks = jax.random.uniform(jax.random.PRNGKey(5), (N,), minval=0.01, maxval=0.04)

    batched = jax.jit(jax.vmap(bl.sample_ball_launch, in_axes=(0, 0, None)), static_argnums=(2,))
    p0s, v0s = batched(rngs, ks, "mixed")
    assert p0s.shape == (N, 3)
    assert v0s.shape == (N, 3)
    assert jnp.all(jnp.isfinite(v0s))
    print(f"[PASS] check_jit_vmap_batch (batch of {N})")


def main():
    check_landing_accuracy()
    check_speed_range()
    check_k_mismatch_degrades_accuracy()
    check_region_sampling()
    check_jit_vmap_batch()
    print("\nAll ball-launch checks passed.")


if __name__ == "__main__":
    main()
