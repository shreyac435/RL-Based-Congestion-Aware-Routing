"""Tabular Q-Learning Agent for congestion-aware routing."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from agents.base import Agent


class QLearningAgent(Agent):
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        decay_episodes: int = 300,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_episodes = decay_episodes
        self.epsilon = epsilon_start
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Q-table: state -> np.ndarray of shape (degree,)
        self.q_table: Dict[Tuple[Any, ...], np.ndarray] = {}
        self.training: bool = True
        self.episode_count: int = 0

        # Episode stats
        self.ep_q_deltas: List[float] = []
        self.ep_rewards: List[float] = []
        self.history_rewards: List[float] = []
        self.history_q_deltas: List[float] = []

    def get_q_values(self, state: Tuple[Any, ...], num_actions: int) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(num_actions, dtype=float)
        return self.q_table[state]

    def choose_next_hop(self, env: Any, packet: Any, node: Any) -> int:
        nbrs = env.neighbors(node)
        k = len(nbrs)
        if k == 0:
            return 0
        if k == 1:
            return 0

        state = env.get_state(node, packet.dst)
        q_vals = self.get_q_values(state, k)

        # Masking:
        # (a) down links
        # (b) previous hop (unless it's the only choice)
        prev_hop = packet.path[-2] if len(packet.path) >= 2 else None
        valid_actions = []
        for idx, nbr in enumerate(nbrs):
            link = env.links.get((node, nbr))
            if link and not link.up:
                continue
            if nbr == prev_hop and k > 1:
                continue
            valid_actions.append(idx)

        # If all masked out, fall back to any up neighbor
        if not valid_actions:
            valid_actions = [
                idx for idx, nbr in enumerate(nbrs)
                if env.links.get((node, nbr)) and env.links[(node, nbr)].up
            ]
        if not valid_actions:
            valid_actions = list(range(k))

        # Epsilon-greedy
        if self.training and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(valid_actions))

        # Greedy choice among valid actions with random tie-breaking
        best_val = -np.inf
        best_acts = []
        for a in valid_actions:
            val = q_vals[a]
            if val > best_val:
                best_val = val
                best_acts = [a]
            elif np.isclose(val, best_val):
                best_acts.append(a)

        return int(self.rng.choice(best_acts))

    def observe(self, transition: Tuple[Any, int, float, Any, bool, Dict[str, Any]]) -> None:
        if not self.training:
            return

        state, action, reward, next_state, done, _ = transition
        k = len(state) - 2  # state is (node, dst, c1..ck)
        q_vals = self.get_q_values(state, k)
        old_val = q_vals[action]

        if done or next_state is None:
            target = reward
        else:
            k_next = len(next_state) - 2
            next_q = self.get_q_values(next_state, k_next)
            target = reward + self.gamma * np.max(next_q)

        q_delta = self.alpha * (target - old_val)
        q_vals[action] += q_delta

        self.ep_q_deltas.append(abs(q_delta))
        self.ep_rewards.append(reward)

    def begin_episode(self, training: bool) -> None:
        self.training = training
        self.ep_q_deltas = []
        self.ep_rewards = []
        if self.training:
            if self.episode_count < self.decay_episodes:
                # Exponential decay towards epsilon_end
                decay_rate = np.log(self.epsilon_end / self.epsilon_start) / float(self.decay_episodes)
                self.epsilon = float(self.epsilon_start * np.exp(decay_rate * self.episode_count))
            else:
                self.epsilon = self.epsilon_end
        else:
            self.epsilon = 0.0

    def end_episode(self) -> Dict[str, float]:
        mean_delta = float(np.mean(self.ep_q_deltas)) if self.ep_q_deltas else 0.0
        mean_reward = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0
        if self.training:
            self.history_q_deltas.append(mean_delta)
            self.history_rewards.append(mean_reward)
            self.episode_count += 1

        return {
            "epsilon": self.epsilon,
            "mean_q_delta": mean_delta,
            "mean_reward": mean_reward,
            "q_table_size": float(len(self.q_table)),
        }
