"""Generates the training set for the drag-correction neural net.

2 versions with 3-feature version (|v0|, angle, k) and 4 feature version with T
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import train_ball_launch as bl

N_SAMPLES = 300_000
K_RANGE = (0.01, 0.04)  
SEED = 0


def _one_sample(rng: jax.Array):
    rng_origin, rng_target, rng_time, rng_k = jax.random.split(rng, 4)

    p0 = bl.sample_service_origin(rng_origin)
    target_xy = bl.sample_landing_target(rng_target, region="mixed")
    T = bl.sample_flight_time(rng_time, p0, target_xy)
    k = jax.random.uniform(rng_k, (), minval=K_RANGE[0], maxval=K_RANGE[1])

    v0 = bl._ballistic_inverse(p0, target_xy, T)
    v_final = bl.solve_launch_velocity(p0, target_xy, T, k)  # truth

    v0_h = jnp.linalg.norm(v0[:2])
    v0_mag = jnp.linalg.norm(v0)
    angle = jnp.arctan2(v0[2], v0_h)
    delta_v = jnp.linalg.norm(v_final[:2]) - v0_h

    return v0_mag, angle, k, T, delta_v


def main():
    rng = jax.random.PRNGKey(SEED)
    rngs = jax.random.split(rng, N_SAMPLES)
    batched = jax.jit(jax.vmap(_one_sample))
    v0_mag, angle, k, T, delta_v = batched(rngs)

    X4 = np.stack([np.asarray(v0_mag), np.asarray(angle), np.asarray(k), np.asarray(T)], axis=1)
    y = np.asarray(delta_v)

    print(f"generated {N_SAMPLES} samples")
    print(f"  |v0| in [{X4[:,0].min():.2f},{X4[:,0].max():.2f}] m/s")
    print(f"  angle in [{np.degrees(X4[:,1].min()):.1f},{np.degrees(X4[:,1].max()):.1f}] deg")
    print(f"  k in [{X4[:,2].min():.3f},{X4[:,2].max():.3f}]")
    print(f"  T in [{X4[:,3].min():.3f},{X4[:,3].max():.3f}]s")
    print(f"  delta_v in [{y.min():.3f},{y.max():.3f}] m/s, mean={y.mean():.3f}")

    np.savez("drag_dataset.npz", v0_mag=X4[:, 0], angle=X4[:, 1], k=X4[:, 2], T=X4[:, 3], delta_v=y)
    print("saved drag_dataset.npz")


if __name__ == "__main__":
    main()
