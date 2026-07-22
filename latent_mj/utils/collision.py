"""Local geom-collision helpers.

Reimplements ``geoms_colliding`` (formerly ``mujoco_playground._src.collision``,
removed in newer mujoco_playground) so the envs don't depend on a shifting
playground internal path. Semantics match the original: two geoms are
"colliding" if they share an active contact with penetration (dist < 0).
"""

from __future__ import annotations

import jax.numpy as jp


def geoms_colliding(state, geom1, geom2):
    """Whether ``geom1`` and ``geom2`` are in penetrating contact in ``state``.

    Args:
        state: an ``mjx.Data`` carrying a ``.contact`` with ``.geom`` of shape
            ``(ncon, 2)`` and ``.dist`` of shape ``(ncon,)``.
        geom1, geom2: integer geom ids (order-independent).

    Returns:
        Scalar boolean array.
    """
    contact = state.contact
    g = contact.geom
    pair = ((g[:, 0] == geom1) & (g[:, 1] == geom2)) | (
        (g[:, 0] == geom2) & (g[:, 1] == geom1)
    )
    return jp.any(pair & (contact.dist < 0))

