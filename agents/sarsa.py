"""On-Policy SARSA Agent."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from agents.qlearning import QLearningAgent


class SARSAAgent(QLearningAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pending hop transitions awaiting next action choice: pkt_id -> (state, action, reward, next_state)
        self.pending_sarsa: Dict[int, Tuple[Tuple[Any, ...], int, float, Tuple[Any, ...]]] = {}

    def observe(self, transition: Tuple[Any, int, float, Any, bool, Dict[str, Any]]) -> None:
        if not self.training:
            return

        state, action, reward, next_state, done, info = transition
        pkt_id = info.get("pkt_id")

        if done or next_state is None:
            # Terminal update: Q(s, a) <- Q(s, a) + alpha * (r - Q(s, a))
            k = max(1, len(state) - 2)
            q_vals = self.get_q_values(state, k)
            act_idx = action if action < len(q_vals) else 0
            old_val = q_vals[act_idx]
            q_delta = self.alpha * (reward - old_val)
            q_vals[act_idx] += q_delta
            self.ep_q_deltas.append(abs(q_delta))
            self.ep_rewards.append(reward)
            if pkt_id in self.pending_sarsa:
                del self.pending_sarsa[pkt_id]
        else:
            # Store pending until next action is chosen at next node
            if pkt_id is not None:
                self.pending_sarsa[pkt_id] = (state, action, reward, next_state)

    def choose_next_hop(self, env: Any, packet: Any, node: Any) -> int:
        act = super().choose_next_hop(env, packet, node)

        # Complete previous pending SARSA update for this packet using chosen action act
        if self.training and packet.id in self.pending_sarsa:
            s_prev, a_prev, r_prev, s_next = self.pending_sarsa.pop(packet.id)
            k_prev = max(1, len(s_prev) - 2)
            k_next = max(1, len(s_next) - 2)
            q_prev = self.get_q_values(s_prev, k_prev)
            q_next = self.get_q_values(s_next, k_next)

            next_val = q_next[act] if act < len(q_next) else (np.max(q_next) if len(q_next) > 0 else 0.0)
            target = r_prev + self.gamma * next_val
            target_act = a_prev if a_prev < len(q_prev) else 0
            q_delta = self.alpha * (target - q_prev[target_act])
            q_prev[target_act] += q_delta

            self.ep_q_deltas.append(abs(q_delta))
            self.ep_rewards.append(r_prev)

        return act
