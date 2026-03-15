"""REINFORCE policy-gradient agent (Williams 1992).

Uses a Monte-Carlo estimate of the policy gradient computed from full
episode returns.  Supports an optional value-function baseline to
reduce variance.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.agents.base_agent import BaseAgent
from src.models.networks import MLP
from src.models.memory import RolloutStorage


class REINFORCEAgent(BaseAgent):
    """REINFORCE with optional value-function baseline.

    Parameters
    ----------
    obs_dim:
        Observation dimensionality.
    action_dim:
        Number of discrete actions.
    hidden_sizes:
        MLP hidden sizes shared by policy and value networks.
    lr:
        Learning rate.
    gamma:
        Discount factor.
    entropy_coef:
        Entropy regularisation coefficient.
    use_baseline:
        If True, learn a separate value network as a variance-reduction
        baseline.
    max_episode_steps:
        Maximum trajectory length; used to pre-allocate rollout storage.
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
        entropy_coef: float = 0.01,
        use_baseline: bool = True,
        max_episode_steps: int = 500,
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_dim, action_dim, device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline

        # Policy network
        self.policy_net = MLP(obs_dim, action_dim, hidden_sizes).to(self.device)

        # Optional value-function baseline
        self.value_net: nn.Module | None = None
        if use_baseline:
            self.value_net = MLP(obs_dim, 1, hidden_sizes).to(self.device)
            params = list(self.policy_net.parameters()) + list(
                self.value_net.parameters()
            )
        else:
            params = list(self.policy_net.parameters())

        self.optimizer = optim.Adam(params, lr=lr)
        self.rollout = RolloutStorage(
            obs_dim=obs_dim,
            max_steps=max_episode_steps,
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
            logits = self.policy_net(obs_t)
            dist = Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            value = (
                self.value_net(obs_t).squeeze(-1)
                if self.value_net is not None
                else torch.tensor(0.0)
            )
        if not deterministic:
            self.rollout.add(
                obs=obs,
                action=int(action.item()),
                reward=0.0,  # updated later via store_reward
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                done=False,
            )
        return int(action.item())

    def store_reward(self, reward: float, done: bool) -> None:
        """Overwrite the reward and done flag for the last stored step."""
        idx = self.rollout._ptr - 1
        self.rollout._rewards[idx] = reward
        self.rollout._dones[idx] = float(done)
        self.increment_steps()

    def update(self, **kwargs: Any) -> dict[str, float]:  # type: ignore[override]
        """Compute REINFORCE gradient update over the current episode."""
        n = len(self.rollout)
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        data = self.rollout.as_tensors()
        returns = torch.from_numpy(
            self.rollout.compute_returns(self.gamma)
        ).to(self.device)

        # Compute current policy log-probs and entropy
        logits = self.policy_net(data["obs"])
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(data["actions"])
        entropy = dist.entropy().mean()

        # Baseline
        if self.value_net is not None:
            values = self.value_net(data["obs"]).squeeze(-1)
            advantages = returns - values.detach()
            value_loss = nn.functional.mse_loss(values, returns)
        else:
            advantages = returns
            value_loss = torch.tensor(0.0, device=self.device)

        # Normalise advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.rollout.reset()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
        }
        if self.value_net is not None:
            sd["value_net"] = self.value_net.state_dict()
        return sd

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        if self.value_net is not None and "value_net" in state:
            self.value_net.load_state_dict(state["value_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._total_steps = state["total_steps"]
