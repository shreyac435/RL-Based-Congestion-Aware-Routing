"""Traffic generation models: Poisson and Bursty On/Off flows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx


@dataclass
class Flow:
    src: Any
    dst: Any
    rate: float  # average packets per tick
    bursty: bool = False
    mean_on: float = 20.0
    mean_off: float = 30.0


class TrafficGenerator:
    def __init__(self, flows: List[Flow], seed: int):
        self.flows = flows
        self.rng = np.random.default_rng(seed)
        self.flow_states: List[Dict[str, Any]] = []
        for f in flows:
            if f.bursty:
                # Initialize state: in_on_state, remaining_ticks
                is_on = self.rng.random() < (f.mean_on / (f.mean_on + f.mean_off))
                duration = self._sample_geom(f.mean_on if is_on else f.mean_off)
                self.flow_states.append({"is_on": is_on, "remaining": duration})
            else:
                self.flow_states.append({})

    def _sample_geom(self, mean: float) -> int:
        p = 1.0 / max(1.0, mean)
        return int(self.rng.geometric(p))

    def generate_packets(self, current_tick: int, pkt_id_start: int, default_ttl: int) -> Tuple[List[Any], int]:
        from sim.network import Packet
        new_packets = []
        cur_id = pkt_id_start

        for idx, flow in enumerate(self.flows):
            if flow.bursty:
                state = self.flow_states[idx]
                state["remaining"] -= 1
                if state["remaining"] <= 0:
                    state["is_on"] = not state["is_on"]
                    mean = flow.mean_on if state["is_on"] else flow.mean_off
                    state["remaining"] = self._sample_geom(mean)

                if state["is_on"]:
                    # In ON period, rate is boosted so average equals flow.rate
                    duty_cycle = flow.mean_on / (flow.mean_on + flow.mean_off)
                    effective_rate = flow.rate / max(1e-4, duty_cycle)
                    k = int(self.rng.poisson(effective_rate))
                    for _ in range(k):
                        pkt = Packet(
                            id=cur_id,
                            src=flow.src,
                            dst=flow.dst,
                            birth_tick=current_tick,
                            ttl=default_ttl,
                        )
                        new_packets.append(pkt)
                        cur_id += 1
            else:
                k = int(self.rng.poisson(flow.rate))
                for _ in range(k):
                    pkt = Packet(
                        id=cur_id,
                        src=flow.src,
                        dst=flow.dst,
                        birth_tick=current_tick,
                        ttl=default_ttl,
                    )
                    new_packets.append(pkt)
                    cur_id += 1

        return new_packets, cur_id


def build_overlapping_flows(graph: nx.Graph, num_hotspots: int = 4, seed: int = 42) -> List[Tuple[Any, Any]]:
    """Generate flows such that Dijkstra paths overlap significantly on central links."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return []

    # Pick pairs with paths >= 2 hops to force routing decisions
    candidate_pairs = []
    for u in nodes:
        for v in nodes:
            if u != v:
                try:
                    path = nx.shortest_path(graph, u, v)
                    if len(path) >= 3:
                        candidate_pairs.append((u, v))
                except nx.NetworkXNoPath:
                    pass

    if not candidate_pairs:
        # Fallback to all distinct pairs
        candidate_pairs = [(u, v) for u in nodes for v in nodes if u != v]

    rng.shuffle(candidate_pairs)
    # Return a subset ensuring some overlap
    return candidate_pairs[:max(num_hotspots, len(nodes))]


def calibrate_load(
    graph: nx.Graph,
    pairs: List[Tuple[Any, Any]],
    target_rho: float,
    link_capacity: int = 2,
    bursty: bool = False,
) -> Tuple[List[Flow], float]:
    """Calculate per-flow rate to achieve target rho on the most loaded Dijkstra link."""
    # Count link usage per shortest path
    link_usage: Dict[Tuple[Any, Any], float] = {}
    for u, v in pairs:
        try:
            path = nx.shortest_path(graph, u, v)
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                link_usage[edge] = link_usage.get(edge, 0.0) + 1.0
        except nx.NetworkXNoPath:
            pass

    max_usage = max(link_usage.values()) if link_usage else 1.0
    # target_rho = (max_usage * base_rate) / link_capacity
    # base_rate = target_rho * link_capacity / max_usage
    base_rate = (target_rho * link_capacity) / max_usage

    flows = [
        Flow(src=u, dst=v, rate=base_rate, bursty=bursty)
        for u, v in pairs
    ]
    achieved_rho = (max_usage * base_rate) / link_capacity
    return flows, achieved_rho
