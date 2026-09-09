"""PPO fine-tuning that keeps a validated behaviour-cloning actor as an anchor.

The actor starts from a BC checkpoint that has passed the rendered-camera gate.
Plain PPO can gradually improve reward in the geometric training lane while
moving into actions that the camera lane never validated.  This algorithm adds
an explicit post-tanh duty rehearsal step on each PPO rollout, using a frozen
copy of that BC actor.  The actor still sees only the seven-value firmware ABI;
the anchor provides no privileged state or action at deployment time.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.algorithms import PPO


class BCAnchoredPPO(PPO):
    """PPO with a frozen direct-duty BC reference on its own rollout states."""

    def __init__(
        self,
        *args,
        bc_anchor_weight: float = 0.0,
        bc_anchor_batch_size: int = 8192,
        bc_anchor_epochs: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if bc_anchor_weight < 0.0:
            raise ValueError("bc_anchor_weight must be non-negative.")
        if bc_anchor_batch_size < 1 or bc_anchor_epochs < 1:
            raise ValueError("bc_anchor_batch_size and bc_anchor_epochs must be positive.")
        self.bc_anchor_weight = float(bc_anchor_weight)
        self.bc_anchor_batch_size = int(bc_anchor_batch_size)
        self.bc_anchor_epochs = int(bc_anchor_epochs)
        self._anchor_actor: torch.nn.Module | None = None
        self._anchor_normalizer: torch.nn.Module | None = None

    def set_actor_anchor(self) -> None:
        """Freeze the already-loaded BC actor as the deployable-duty target."""
        self._anchor_actor = copy.deepcopy(self.policy.actor).to(self.device).eval()
        self._anchor_normalizer = copy.deepcopy(self.policy.actor_obs_normalizer).to(self.device).eval()
        for module in (self._anchor_actor, self._anchor_normalizer):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def update(self) -> dict[str, float]:
        """Run ordinary PPO, then rehearse BC duties on this rollout's ABI states."""
        if self.bc_anchor_weight == 0.0:
            return super().update()
        if self._anchor_actor is None or self._anchor_normalizer is None:
            raise RuntimeError("BCAnchoredPPO requires set_actor_anchor() after the actor warm start.")
        if self.storage is None:
            raise RuntimeError("BCAnchoredPPO update called before rollout storage was initialized.")

        # Take the policy observation tensor before PPO clears rollout storage.
        # It is the frozen ABI only, never privileged track state.
        actor_observations = self.policy.get_actor_obs(self.storage.observations.flatten(0, 1)).detach().clone()
        with torch.no_grad():
            reference_duty = torch.tanh(self._anchor_actor(self._anchor_normalizer(actor_observations)))

        losses = super().update()
        sample_count = actor_observations.shape[0]
        cumulative_loss = 0.0
        update_count = 0
        for _ in range(self.bc_anchor_epochs):
            order = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.bc_anchor_batch_size):
                indices = order[start:start + self.bc_anchor_batch_size]
                current_duty = torch.tanh(
                    self.policy.actor(self.policy.actor_obs_normalizer(actor_observations[indices])),
                )
                anchor_loss = torch.nn.functional.mse_loss(current_duty, reference_duty[indices])
                self.optimizer.zero_grad()
                (self.bc_anchor_weight * anchor_loss).backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                cumulative_loss += anchor_loss.item()
                update_count += 1
        losses["bc_anchor"] = cumulative_loss / update_count
        return losses
