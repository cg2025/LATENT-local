"""

  mlp_3feat.npz (|v0|, angle, k)
  mlp_4feat.npz (|v0|, angle, k, T)

"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import train_ball_launch as bl

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_mlp(path: str) -> dict[str, jnp.ndarray]:
    d = np.load(path)
    return {k: jnp.array(d[k]) for k in d.files}


# loadede
_MLP_3FEAT = _load_mlp(os.path.join(_THIS_DIR, "mlp_3feat.npz"))
_MLP_4FEAT = _load_mlp(os.path.join(_THIS_DIR, "mlp_4feat.npz"))


def _mlp_predict(m: dict, x: jnp.ndarray) -> jnp.ndarray:
    """2-hidden-layer tanh MLP, standardized in/out. x  is a single
    feature vector (3,) or (4,); returns a scalar delta_v prediction."""
    xn = (x - m["x_mean"]) / m["x_std"]
    h = jnp.tanh(xn @ m["W1"] + m["b1"])
    h = jnp.tanh(h @ m["W2"] + m["b2"])
    yn = (h @ m["W3"] + m["b3"])[..., 0]
    return yn * m["y_std"] + m["y_mean"]


def _solve_launch_velocity_nn(
    p0: jnp.ndarray,
    target_xy: jnp.ndarray,
    T: float,
    k: float,
    model: dict,
    use_T: bool,
) -> jnp.ndarray:
    v0 = bl._ballistic_inverse(p0, target_xy, T)
    v0_h = jnp.linalg.norm(v0[:2])
    v0_mag = jnp.linalg.norm(v0)
    angle = jnp.arctan2(v0[2], v0_h)

    x = jnp.array([v0_mag, angle, k, T]) if use_T else jnp.array([v0_mag, angle, k])
    delta_v = _mlp_predict(model, x)

    scale = (v0_h + delta_v) / v0_h
    return v0.at[:2].multiply(scale)


def solve_launch_velocity_nn(p0, target_xy, T, k):
    """(|v0|, angle, k) -> delta_v. Default."""
    return _solve_launch_velocity_nn(p0, target_xy, T, k, _MLP_3FEAT, use_T=False)


def solve_launch_velocity_nn_4feat(p0, target_xy, T, k):
    """(|v0|, angle, k, T) -> delta_v."""
    return _solve_launch_velocity_nn(p0, target_xy, T, k, _MLP_4FEAT, use_T=True)


def sample_ball_launch_nn(rng: jax.Array, k: float, region: str = "mixed"):
    rng_origin, rng_target, rng_time = jax.random.split(rng, 3)
    p0 = bl.sample_service_origin(rng_origin)
    target_xy = bl.sample_landing_target(rng_target, region=region)
    T = bl.sample_flight_time(rng_time, p0, target_xy)
    v0 = solve_launch_velocity_nn(p0, target_xy, T, k)
    return p0, v0
