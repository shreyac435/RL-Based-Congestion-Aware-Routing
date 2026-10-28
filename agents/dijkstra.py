"""Dijkstra baseline (B1): static shortest path on hop count."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import networkx as nx

from agents.base import Agent


class DijkstraAgent(Agent):
    def __init__(self, graph: nx.Graph, detection_delay: int = 20):
        self.original_graph = graph
        self.detection_delay = detection_delay
        self.last_failure_tick: Optional[int] = None
        self.active_graph = graph.copy()
        self.routes: Dict[Any, Dict[Any, List[Any]]] = {}
        self._recompute_routes()

    def _recompute_routes(self) -> None:
        self.routes = {}
        for src in self.active_graph.nodes():
            self.routes[src] = {}
            for dst in self.active_graph.nodes():
                if src == dst:
                    self.routes[src][dst] = [src]
                else:
                    try:
                        self.routes[src][dst] = nx.shortest_path(self.active_graph, src, dst)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        self.routes[src][dst] = []

    def on_tick(self, env: Any) -> None:
        self.check_topology_update(env)

    def check_topology_update(self, env: Any) -> None:
        # Check if topology changed (links went down) and detection delay passed
        down_links = [
            (u, v) for (u, v), link in env.links.items() if not link.up
        ]
        if down_links:
            needs_update = False
            for u, v in down_links:
                if self.active_graph.has_edge(u, v):
                    needs_update = True
                    break
            if needs_update:
                if self.last_failure_tick is None:
                    self.last_failure_tick = env.current_tick
                elif env.current_tick - self.last_failure_tick >= self.detection_delay:
                    for u, v in down_links:
                        if self.active_graph.has_edge(u, v):
                            self.active_graph.remove_edge(u, v)
                    self._recompute_routes()
                    self.last_failure_tick = None
        else:
            if self.active_graph.number_of_edges() < self.original_graph.number_of_edges():
                self.active_graph = self.original_graph.copy()
                self._recompute_routes()
                self.last_failure_tick = None

    def choose_next_hop(self, env: Any, packet: Any, node: Any) -> int:
        nbrs = env.neighbors(node)
        if not nbrs:
            return 0

        self.check_topology_update(env)

        dst = packet.dst
        path = self.routes.get(node, {}).get(dst, [])
        if len(path) >= 2:
            next_node = path[1]
            if next_node in nbrs:
                return nbrs.index(next_node)

        # Fallback to first available up neighbor
        for i, nbr in enumerate(nbrs):
            if env.links.get((node, nbr), None) and env.links[(node, nbr)].up:
                return i
        return 0
