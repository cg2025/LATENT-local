"""Trains a small MLP on the dataset for (|v0|, angle, k) -> delta_v .

Also trains the 4-feature version (adds T) 

2 hidden layers, 64 units, tanh
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

HIDDEN = 64
N_STEPS = 4000
LR = 3e-3
BATCH = 4096
SEED = 0


def init_mlp(rng, in_dim, hidden, out_dim=1):
    k1, k2, k3 = jax.random.split(rng, 3)
    scale = lambda fan_in: 1.0 / jnp.sqrt(fan_in)
    params = {
        "W1": jax.random.normal(k1, (in_dim, hidden)) * scale(in_dim),
        "b1": jnp.zeros(hidden),
        "W2": jax.random.normal(k2, (hidden, hidden)) * scale(hidden),
        "b2": jnp.zeros(hidden),
        "W3": jax.random.normal(k3, (hidden, out_dim)) * scale(hidden),
        "b3": jnp.zeros(out_dim),
    }
    return params


def mlp_forward(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    return (h @ params["W3"] + params["b3"])[..., 0]


def loss_fn(params, x, y):
    pred = mlp_forward(params, x)
    return jnp.mean((pred - y) ** 2)


def train(X: np.ndarray, y: np.ndarray, seed: int = SEED, tag: str = ""):
    
    x_mean, x_std = X.mean(0), X.std(0)
    y_mean, y_std = y.mean(), y.std()
    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    n = len(y)
    idx = np.random.RandomState(1).permutation(n)
    split = int(0.9 * n)
    tr, te = idx[:split], idx[split:]
    Xtr, ytr = jnp.array(Xn[tr]), jnp.array(yn[tr])
    Xte, yte = jnp.array(Xn[te]), jnp.array(yn[te])

    rng = jax.random.PRNGKey(seed)
    params = init_mlp(rng, in_dim=X.shape[1], hidden=HIDDEN)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng_batch = jax.random.PRNGKey(seed + 1)
    for step_i in range(N_STEPS):
        rng_batch, sub = jax.random.split(rng_batch)
        b_idx = jax.random.randint(sub, (BATCH,), 0, len(ytr))
        params, opt_state, loss = step(params, opt_state, Xtr[b_idx], ytr[b_idx])
        if step_i % 1000 == 0:
            test_loss = float(loss_fn(params, Xte, yte))
            print(f"  [{tag}] step {step_i}: train_loss={float(loss):.4f} test_loss(norm)={test_loss:.4f}")

    
    pred_te_n = mlp_forward(params, Xte)
    pred_te = np.asarray(pred_te_n) * y_std + y_mean
    y_te_real = np.asarray(yte) * y_std + y_mean
    err = y_te_real - pred_te
    print(f"  [{tag}] held-out delta_v error: mean_abs={np.abs(err).mean():.4f} m/s, "
          f"max_abs={np.abs(err).max():.4f} m/s")

    return dict(params=params, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


def main():
    d = np.load("drag_dataset.npz")
    v0_mag, angle, k, T, y = d["v0_mag"], d["angle"], d["k"], d["T"], d["delta_v"]

    print("=== 3-feature model: (|v0|, angle, k) -> delta_v ===")
    X3 = np.stack([v0_mag, angle, k], axis=1)
    model3 = train(X3, y, tag="3-feat")

    print("\n=== 4-feature model: (|v0|, angle, k, T) -> delta_v ===")
    X4 = np.stack([v0_mag, angle, k, T], axis=1)
    model4 = train(X4, y, tag="4-feat")

    np.savez("mlp_3feat.npz",
             W1=model3["params"]["W1"], b1=model3["params"]["b1"],
             W2=model3["params"]["W2"], b2=model3["params"]["b2"],
             W3=model3["params"]["W3"], b3=model3["params"]["b3"],
             x_mean=model3["x_mean"], x_std=model3["x_std"],
             y_mean=model3["y_mean"], y_std=model3["y_std"])
    np.savez("mlp_4feat.npz",
             W1=model4["params"]["W1"], b1=model4["params"]["b1"],
             W2=model4["params"]["W2"], b2=model4["params"]["b2"],
             W3=model4["params"]["W3"], b3=model4["params"]["b3"],
             x_mean=model4["x_mean"], x_std=model4["x_std"],
             y_mean=model4["y_mean"], y_std=model4["y_std"])
    print("\nsaved mlp_3feat.npz and mlp_4feat.npz")


if __name__ == "__main__":
    main()
