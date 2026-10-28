"""B2 Ablation: Q-Routing without congestion state."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from agents.qlearning import QLearningAgent


class QRoutingNoCongAgent(QLearningAgent):
    """Q-routing where state is only (node, dest), ignoring congestion bins."""

    def choose_next_hop(self, env: Any, packet: Any, node: Any) -> int:
        nbrs = env.neighbors(node)
        k = len(nbrs)
        if k <= 1:
            return 0

        state = (node, packet.dst)
        q_vals = self.get_q_values(state, k)

        prev_hop = packet.path[-2] if len(packet.path) >= 2 else None
        valid_actions = []
        for idx, nbr in enumerate(nbrs):
            link = env.links.get((node, nbr))
            if link and not link.up:
                continue
            if nbr == prev_hop and k > 1:
                continue
            valid_actions.append(idx)

        if not valid_actions:
            valid_actions = [
                idx for idx, nbr in enumerate(nbrs)
                if env.links.get((node, nbr)) and env.links[(node, nbr)].up
            ]
        if not valid_actions:
            valid_actions = list(range(k))

        if self.training and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(valid_actions))

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
        node, dst = state[0], state[1]
        s_nocong = (node, dst)
        k = len(state) - 2
        q_vals = self.get_q_values(s_nocong, k)
        old_val = q_vals[action]

        if done or next_state is None:
            target = reward
        else:
            next_node, next_dst = next_state[0], next_state[1]
            s_next_nocong = (next_node, next_dst)
            k_next = len(next_state) - 2
            next_q = self.get_q_values(s_next_nocong, k_next)
            target = reward + self.gamma * np.max(next_q)

        q_delta = self.alpha * (target - old_val)
        q_vals[action] += q_delta

        self.ep_q_deltas.append(abs(q_delta))
        self.ep_rewards.append(reward)
