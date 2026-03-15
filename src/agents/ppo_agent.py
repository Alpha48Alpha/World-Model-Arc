"""Proximal Policy Optimisation (PPO) agent (Schulman et al. 2017).

Implements clipped PPO with:
  - Generalised Advantage Estimation (GAE).
  - Multiple epochs of mini-batch updates per rollout.
  - Entropy bonus for exploration.
  - Gradient norm clipping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.agents.base_agent import BaseAgent
from src.models.networks import ActorCritic
from src.models.memory import RolloutStorage


class PPOAgent(BaseAgent):
    """PPO agent for discrete action spaces.

    Parameters
    ----------
    obs_dim:
        Observation dimensionality.
    action_dim:
        Number of discrete actions.
    hidden_sizes:
        Hidden sizes for the shared ActorCritic backbone.
    lr:
        Adam learning rate.
    gamma:
        Discount factor.
    gae_lambda:
        GAE lambda parameter.
    clip_eps:
        PPO clipping epsilon.
    entropy_coef:
        Entropy regularisation coefficient.
    value_coef:
        Value function loss coefficient.
    n_epochs:
        Number of optimisation epochs per rollout.
    batch_size:
        Mini-batch size during PPO updates.
    rollout_steps:
        Number of environment steps collected per update.
    max_grad_norm:
        Gradient clipping threshold.
    device:
        Torch device string.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 512,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_dim, action_dim, device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm

        self.ac = ActorCritic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        self.rollout = RolloutStorage(
            obs_dim=obs_dim,
            max_steps=rollout_steps + 1,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, *, deterministic: bool = False
    ) -> int:
        obs_t = self.obs_to_tensor(obs)
        with torch.no_grad():
            logits, value = self.ac(obs_t)
            dist = Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        if not deterministic:
            self.rollout.add(
                obs=obs,
                action=int(action.item()),
                reward=0.0,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                done=False,
            )
        return int(action.item())

    def store_reward(self, reward: float, done: bool) -> None:
        """Overwrite the reward / done flag for the last collected step."""
        idx = self.rollout._ptr - 1
        self.rollout._rewards[idx] = reward
        self.rollout._dones[idx] = float(done)
        self.increment_steps()

    def update(
        self, last_obs: np.ndarray | None = None, **kwargs: Any
    ) -> dict[str, float]:  # type: ignore[override]
        """Run PPO update on current rollout buffer.

        Parameters
        ----------
        last_obs:
            Final observation (used to bootstrap the value when the
            episode has not terminated).
        """
        n = len(self.rollout)
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Bootstrap value
        if last_obs is not None:
            with torch.no_grad():
                _, last_val = self.ac(self.obs_to_tensor(last_obs))
            last_value = float(last_val.item())
        else:
            last_value = 0.0

        advantages, returns = self.rollout.compute_gae(
            gamma=self.gamma,
            lam=self.gae_lambda,
            last_value=last_value,
        )
        advantages_t = torch.from_numpy(advantages).to(self.device)
        returns_t = torch.from_numpy(returns).to(self.device)

        # Normalise advantages
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)

        data = self.rollout.as_tensors()
        old_log_probs = data["log_probs"]
        obs_batch = data["obs"]
        actions_batch = data["actions"]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.n_epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                mb_idx = idx[start: start + self.batch_size]
                mb_obs = obs_batch[mb_idx]
                mb_actions = actions_batch[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                logits, values = self.ac(mb_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                num_updates += 1

        self.rollout.reset()
        denom = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "ac": self.ac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ac.load_state_dict(state["ac"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._total_steps = state["total_steps"]
