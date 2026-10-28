"""Agent abstract base interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class Agent(ABC):
    @abstractmethod
    def choose_next_hop(self, env: Any, packet: Any, node: Any) -> int:
        """Choose next-hop neighbor index in range [0, degree - 1]."""
        raise NotImplementedError

    def on_tick(self, env: Any) -> None:
        """Optional tick hook called at each simulation tick."""
        pass

    def observe(self, transition: Tuple[Any, int, float, Any, bool, Dict[str, Any]]) -> None:
        """Observe delayed transition: (state, action_idx, reward, next_state_or_None, done, info)."""
        pass

    def begin_episode(self, training: bool) -> None:
        """Hook called at the start of each episode."""
        pass

    def end_episode(self) -> Dict[str, float]:
        """Hook called at the end of each episode; returns episode summary dict."""
        return {}
