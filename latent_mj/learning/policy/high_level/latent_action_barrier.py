"""Frozen decoder/prior wrapper + Latent Action Barrier (LAB).

a_full_t = [ D(s_t, mu_p_t + lambda * sigma_p_t * tanh(a_latent_t)),
                 a_correct_t ]

where D is the frozen decoder and (mu_p, sigma_p) = P(z | s_t) is the frozen
learnable prior, both trained in Section 3.2.2 (train_vae_distill.py) and
loaded here as fixed, non-trainable parameters.

 `a_correct_t` term (direct PD
targets for the 3 excluded right-wrist joints, Section 3.3.2) is produced
directly by the high-level policy's own output head. 

Will need to concatenating it into the full 29-dim actuator target with tennis-return env
that wraps this class.

no numeric value for lambda (the barrier scale) -  hyperparameter to sweep
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from latent_mj.learning.policy.vae.networks import VAEPolicy


def load_frozen_vae_params(ckpt_path: str | Path, vae: VAEPolicy):
    """Restore a VAEPolicy checkpoint saved by train_vae_distill.py.
    """
    checkpointer = ocp.StandardCheckpointer()
    dummy_proprio = jnp.zeros((1, vae.proprio_dim))
    dummy_dif = jnp.zeros((1, vae.dif_dim))
    rng = jax.random.PRNGKey(0)
    # init() builds the full module tree (encoder included) even though the
    # encoder is never used post-distillation -- neeed it  for  matching
    # pytree structure for orbax's restore().
    dummy_params = vae.init(rng, dummy_proprio, dummy_dif, rng)
    params = checkpointer.restore(str(ckpt_path), dummy_params)
    return params


@dataclasses.dataclass
class LABOutput:
    """Diagnostics from a single LAB decode call.

    a_body: decoded action for the active (non-wrist) joints, (batch, action_dim)
    z: the latent code actually fed to the decoder, (batch, latent_dim)
    mu_p: prior mean at s_t, (batch, latent_dim)
    sigma_p: prior std at s_t, (batch, latent_dim)
    tanh_sat: mean |tanh(a_latent)| across the batch and latent dims, in [0, 1].
        Values near 1.0 indicate the high-level policy is consistently
        saturating the barrier.
    """
    a_body: jnp.ndarray
    z: jnp.ndarray
    mu_p: jnp.ndarray
    sigma_p: jnp.ndarray
    tanh_sat: jnp.ndarray


class LatentActionBarrier:
    """Frozen decoder + learnable prior, exposed as a single decode() call.
    """

    def __init__(self, vae: VAEPolicy, params, lam: float):
        self.vae = vae
        self.params = params
        self.lam = lam
        self._decode_fn = jax.jit(self._decode_impl)  # returns tuple

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        proprio_dim: int,
        dif_dim: int,
        action_dim: int,
        latent_dim: int,
        lam: float,
        encoder_hidden: Tuple[int, ...] = (512, 256),
        decoder_hidden: Tuple[int, ...] = (512, 256, 128),
        prior_hidden: Tuple[int, ...] = (256, 128),
    ) -> "LatentActionBarrier":
        vae = VAEPolicy(
            proprio_dim=proprio_dim,
            dif_dim=dif_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            prior_hidden=prior_hidden,
        )
        params = load_frozen_vae_params(ckpt_path, vae)
        return cls(vae=vae, params=params, lam=lam)

    def _decode_impl(self, proprio: jnp.ndarray, a_latent_residual: jnp.ndarray):
        # Returns a tuple (not LABOutput) because jax.jit requires the
        # traced return value to be a registered pytree.
        frozen_params = jax.tree_util.tree_map(jax.lax.stop_gradient, self.params)
        mu_p, log_sigma_p = self.vae.apply(frozen_params, proprio, method=self.vae.prior_params)
        sigma_p = jnp.exp(log_sigma_p)

        squashed = jnp.tanh(a_latent_residual)
        z = mu_p + self.lam * sigma_p * squashed  # Eq. 4

        a_body = self.vae.apply(frozen_params, proprio, z, method=self.vae.decode)
        tanh_sat = jnp.mean(jnp.abs(squashed))

        return a_body, z, mu_p, sigma_p, tanh_sat

    def decode(self, proprio: jnp.ndarray, a_latent_residual: jnp.ndarray) -> LABOutput:
        """Eq. 4 decode. Batched: proprio (B, proprio_dim), a_latent_residual (B, latent_dim)."""
        a_body, z, mu_p, sigma_p, tanh_sat = self._decode_fn(proprio, a_latent_residual)
        return LABOutput(a_body=a_body, z=z, mu_p=mu_p, sigma_p=sigma_p, tanh_sat=tanh_sat)

    def decode_deterministic(self, proprio: jnp.ndarray) -> LABOutput:
        """a_latent_residual = 0 -> z = mu_p (prior mean)."""
        a_latent_residual = jnp.zeros((proprio.shape[0], self.vae.latent_dim))
        return self.decode(proprio, a_latent_residual)
