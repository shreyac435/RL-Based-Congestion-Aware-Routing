"""Discrete-time network simulation primitives."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx


@dataclass
class Packet:
    id: int
    src: Any
    dst: Any
    birth_tick: int
    ttl: int
    hops: int = 0
    current_node: Any = None
    path: List[Any] = field(default_factory=list)
    pending_transition: Optional[Dict[str, Any]] = None
    enqueue_tick: int = 0

    def __post_init__(self):
        if self.current_node is None:
            self.current_node = self.src
        if not self.path:
            self.path = [self.current_node]


class Link:
    def __init__(
        self,
        u: Any,
        v: Any,
        capacity: int = 2,
        buffer_size: int = 20,
        prop_delay: int = 1,
    ):
        self.u = u
        self.v = v
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.prop_delay = prop_delay
        self.queue: deque[Packet] = deque()
        self.pipe: deque[Tuple[int, Packet]] = deque()
        self.up: bool = True
        self.packets_sent_this_tick: int = 0
        self.total_sent: int = 0
        self.total_dropped: int = 0

    @property
    def queue_occupancy(self) -> float:
        if self.buffer_size == 0:
            return 1.0
        return len(self.queue) / float(self.buffer_size)

    def congestion_bin(self, n_bins: int = 3) -> int:
        if not self.up:
            return n_bins - 1
        q = self.queue_occupancy
        if n_bins == 2:
            return 0 if q < 0.5 else 1
        elif n_bins == 4:
            if q < 0.25:
                return 0
            elif q < 0.50:
                return 1
            elif q < 0.75:
                return 2
            return 3
        else:  # default 3 bins
            if q < 0.33:
                return 0
            elif q < 0.66:
                return 1
            return 2

    def transmit(self, current_tick: int) -> int:
        """Transmit up to capacity packets into propagation pipe."""
        self.packets_sent_this_tick = 0
        if not self.up:
            return 0
        while self.queue and self.packets_sent_this_tick < self.capacity:
            pkt = self.queue.popleft()
            arrival_tick = current_tick + self.prop_delay
            self.pipe.append((arrival_tick, pkt))
            self.packets_sent_this_tick += 1
            self.total_sent += 1
        return self.packets_sent_this_tick

    def purge_on_failure(self) -> List[Packet]:
        """Purge queued and in-flight packets on link failure."""
        dropped = list(self.queue)
        dropped.extend([pkt for _, pkt in self.pipe])
        self.queue.clear()
        self.pipe.clear()
        self.total_dropped += len(dropped)
        return dropped
