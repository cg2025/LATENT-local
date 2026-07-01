"""VAE networks for online distillation with variational bottleneck.

Architecture from LATENT paper:
  - Posterior encoder E(z|s_t, s̃_{t+1}) --> N(mu, sigma)
  - Decoder D(a|s_t, z) --> action
  - Learnable prior P(z|s_t) --> N(mu_p, sigma_p)

Variables:
  s_t = current state (151 dims)
  s̃_{t+1} = privileged state / motion target (572 dims)
  z = latent code (latent_dim dims, default 32)
  a = action (26 dims)
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Tuple


class PosteriorEncoder(nn.Module):
    """Encoding (state, motion_target) to (mu, log_sigma) of posterior q(z|s, s̃).

    Inputs:
        state: (batch, state_dim) which is the current policy observation
        motion_target: (batch, privileged_dim) which is the "privileged" state (motion reference info)
    Outputs:
        mu: (batch, latent_dim)
        log_sigma: (batch, latent_dim)
    """
    hidden_sizes: Sequence[int] = (512, 256)
    latent_dim: int = 32

    @nn.compact
    def __call__(self, state: jnp.ndarray, motion_target: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([state, motion_target], axis=-1)
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.silu(x)
        mu = nn.Dense(self.latent_dim)(x)
        log_sigma = nn.Dense(self.latent_dim)(x)
        log_sigma = jnp.clip(log_sigma, -5.0, 2.0)  # numerical stability
        return mu, log_sigma


class Decoder(nn.Module):
    """Decodes (state, latent_code) to action.

    Inputs:
        state: (batch, state_dim) which is current policy observation
        z: (batch, latent_dim) which is sampled latent code
    Outputs:
        action: (batch, action_dim)
    """
    hidden_sizes: Sequence[int] = (512, 256, 128)
    action_dim: int = 26

    @nn.compact
    def __call__(self, state: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, z], axis=-1)
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.silu(x)
        action = nn.Dense(self.action_dim)(x)
        return action


class LearnablePrior(nn.Module):
    """State-conditioned prior P(z|s) = N(mu_p(s), sigma_p(s)).

    We don't have fixed N(0,I) prior but rather we capture state-dependent action distributions
    (ex: lateral shuffle vs racket-swinging have different latent distributions).

    Inputs:
        state: (batch, state_dim)
    Outputs:
        mu_prior: (batch, latent_dim)
        log_sigma_prior: (batch, latent_dim)
    """
    hidden_sizes: Sequence[int] = (256, 128)
    latent_dim: int = 32

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = state
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.silu(x)
        mu_prior = nn.Dense(self.latent_dim)(x)
        log_sigma_prior = nn.Dense(self.latent_dim)(x)
        log_sigma_prior = jnp.clip(log_sigma_prior, -5.0, 2.0)
        return mu_prior, log_sigma_prior


class VAEPolicy(nn.Module):
    """Full VAE policy combining encoder, decoder, and prior.

    At training time: we encode (s, s̃) -> z, decode z -> a, compute KL with prior
    At inference time: we sample z from prior P(z|s), decode -> a
    """
    state_dim: int = 151
    privileged_dim: int = 572
    action_dim: int = 26
    latent_dim: int = 32
    encoder_hidden: Sequence[int] = (512, 256)
    decoder_hidden: Sequence[int] = (512, 256, 128)
    prior_hidden: Sequence[int] = (256, 128)

    def setup(self):
        self.encoder = PosteriorEncoder(
            hidden_sizes=self.encoder_hidden,
            latent_dim=self.latent_dim,
        )
        self.decoder = Decoder(
            hidden_sizes=self.decoder_hidden,
            action_dim=self.action_dim,
        )
        self.prior = LearnablePrior(
            hidden_sizes=self.prior_hidden,
            latent_dim=self.latent_dim,
        )

    def encode(self, state: jnp.ndarray, motion_target: jnp.ndarray):
        """Posterior encode (s, s̃) -> (mu, log_sigma)."""
        return self.encoder(state, motion_target)

    def decode(self, state: jnp.ndarray, z: jnp.ndarray):
        """Decode (s, z) -> action."""
        return self.decoder(state, z)

    def prior_params(self, state: jnp.ndarray):
        """Prior P(z|s) -> (mu_prior, log_sigma_prior)."""
        return self.prior(state)

    def __call__(self, state: jnp.ndarray, motion_target: jnp.ndarray, rng: jax.Array, deterministic: bool = False):
        """Full forward pass for training.

        We return action (reconstructed action), mu (posterior mean), log_sigma (posterior log std), mu_prior (prior mean), log_sigma_prior (prior log std)
        """
        mu, log_sigma = self.encoder(state, motion_target)
        mu_prior, log_sigma_prior = self.prior(state)

        if deterministic:
            z = mu
        else:
            eps = jax.random.normal(rng, shape=mu.shape)
            z = mu + jnp.exp(log_sigma) * eps

        action = self.decoder(state, z)
        return action, mu, log_sigma, mu_prior, log_sigma_prior

    def inference(self, state: jnp.ndarray, rng: jax.Array, deterministic: bool = False):
        """Inference-time forward pass: sample z from prior P(z|s) as action and use at deployment time when motion target is not available.
        """
        mu_prior, log_sigma_prior = self.prior(state)
        if deterministic:
            z = mu_prior
        else:
            eps = jax.random.normal(rng, shape=mu_prior.shape)
            z = mu_prior + jnp.exp(log_sigma_prior) * eps
        action = self.decoder(state, z)
        return action


def kl_divergence(mu_q, log_sigma_q, mu_p, log_sigma_p):
    """KL divergence between two diagonal Gaussians. KL(N(mu_q, sigma_q) || N(mu_p, sigma_p))
    """
    sigma_q = jnp.exp(log_sigma_q)
    sigma_p = jnp.exp(log_sigma_p)
    kl = (
        log_sigma_p - log_sigma_q
        + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
        - 0.5
    )
    return jnp.sum(kl, axis=-1).mean()


def vae_loss(action_pred, action_teacher, mu, log_sigma, mu_prior, log_sigma_prior,
             lambda_action=1.0, lambda_kl=0.01):
    """Total VAE loss = lambda_action * L_action + lambda_kl * L_KL.
    L_action: MSE between student decoded action and teacher action
    L_KL: KL divergence between posterior q(z|s,s̃) and prior p(z|s)
    """
    l_action = jnp.mean(jnp.sum((action_pred - action_teacher)**2, axis=-1))
    l_kl = kl_divergence(mu, log_sigma, mu_prior, log_sigma_prior)
    total = lambda_action * l_action + lambda_kl * l_kl
    return total, l_action, l_kl
