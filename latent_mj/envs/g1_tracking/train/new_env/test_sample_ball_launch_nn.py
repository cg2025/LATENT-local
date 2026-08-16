"""Checks for ball_launch_nn.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import train_ball_launch as bl
import sample_ball_launch_nn as bln


def check_landing_accuracy_comparison(n: int = 500):
    def make_shot(rng):
        ro, rt, rtime, rk = jax.random.split(rng, 4)
        p0 = bl.sample_service_origin(ro)
        tgt = bl.sample_landing_target(rt)
        T = bl.sample_flight_time(rtime, p0, tgt)
        k = jax.random.uniform(rk, (), minval=0.01, maxval=0.04)
        return p0, tgt, T, k

    def orig_err(rng):
        p0, tgt, T, k = make_shot(rng)
        v = bl.solve_launch_velocity(p0, tgt, T, k)
        land, _ = bl._forward_sim_landing(p0, v, k)
        return jnp.linalg.norm(tgt - land)

    def nn3_err(rng):
        p0, tgt, T, k = make_shot(rng)
        v = bln.solve_launch_velocity_nn(p0, tgt, T, k)
        land, _ = bl._forward_sim_landing(p0, v, k)
        return jnp.linalg.norm(tgt - land)

    def nn4_err(rng):
        p0, tgt, T, k = make_shot(rng)
        v = bln.solve_launch_velocity_nn_4feat(p0, tgt, T, k)
        land, _ = bl._forward_sim_landing(p0, v, k)
        return jnp.linalg.norm(tgt - land)

    rng = jax.random.PRNGKey(777)  # fresh seed
    rngs = jax.random.split(rng, n)

    errs_orig = np.asarray(jax.jit(jax.vmap(orig_err))(rngs))
    errs_3 = np.asarray(jax.jit(jax.vmap(nn3_err))(rngs))
    errs_4 = np.asarray(jax.jit(jax.vmap(nn4_err))(rngs))

    for name, e in [("ORIG (3-pass, has inner forward-sim loop)", errs_orig),
                    ("NN 3-feat (|v0|,angle,k)", errs_3),
                    ("NN 4-feat (|v0|,angle,k,T)", errs_4)]:
        print(f"  {name}: mean={e.mean():.4f}m  p95={np.percentile(e,95):.4f}m  max={e.max():.4f}m")


def check_jit_vmap_batch():
    N = 32
    rng = jax.random.PRNGKey(4)
    rngs = jax.random.split(rng, N)
    ks = jax.random.uniform(jax.random.PRNGKey(5), (N,), minval=0.01, maxval=0.04)

    batched = jax.jit(jax.vmap(bln.sample_ball_launch_nn, in_axes=(0, 0, None)), static_argnums=(2,))
    p0s, v0s = batched(rngs, ks, "mixed")
    assert p0s.shape == (N, 3)
    assert v0s.shape == (N, 3)
    assert jnp.all(jnp.isfinite(v0s))
    print(f"[PASS] check_jit_vmap_batch (batch of {N})")


def main():
    check_landing_accuracy_comparison()
    check_jit_vmap_batch()


if __name__ == "__main__":
    main()
