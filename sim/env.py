"""Gym-style Discrete-Time Routing Environment."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np

from sim.network import Link, Packet
from sim.traffic import TrafficGenerator, Flow, build_overlapping_flows, calibrate_load
from sim.metrics import EpisodeMetricsCollector


class RoutingEnv:
    def __init__(
        self,
        graph: nx.Graph,
        traffic_cfg: Optional[Dict[str, Any]] = None,
        sim_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.graph = graph
        self.traffic_cfg = traffic_cfg or {}
        self.sim_cfg = sim_cfg or {}
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        # Cache fixed sorted neighbors
        self.nodes = sorted(list(graph.nodes()))
        self._neighbors: Dict[Any, List[Any]] = {
            n: sorted(list(graph.neighbors(n))) for n in self.nodes
        }

        # Config parameters
        self.capacity = self.sim_cfg.get("capacity", 2)
        self.buffer_size = self.sim_cfg.get("buffer_size", 20)
        self.prop_delay = self.sim_cfg.get("prop_delay", 1)
        self.default_ttl = self.sim_cfg.get("default_ttl", min(64, 2 * len(self.nodes)))
        self.n_bins = self.sim_cfg.get("n_bins", 3)

        # Rewards
        self.p_cong = self.sim_cfg.get("p_cong", 2.0)
        self.p_drop = self.sim_cfg.get("p_drop", 30.0)
        self.r_deliv = self.sim_cfg.get("r_deliv", 10.0)
        self.p_loop = self.sim_cfg.get("p_loop", 5.0)

        # Build directed links
        self.links: Dict[Tuple[Any, Any], Link] = {}
        for u in self.nodes:
            for v in self._neighbors[u]:
                self.links[(u, v)] = Link(
                    u=u,
                    v=v,
                    capacity=self.capacity,
                    buffer_size=self.buffer_size,
                    prop_delay=self.prop_delay,
                )

        # Scheduled failures: tick -> list of ((u, v), is_recover)
        self.failure_schedule: Dict[int, List[Tuple[Tuple[Any, Any], bool]]] = {}

        # Runtime state
        self.current_tick: int = 0
        self.next_pkt_id: int = 1
        self.routing_backlog: Dict[Any, List[Packet]] = {n: [] for n in self.nodes}
        self.traffic_generator: Optional[TrafficGenerator] = None
        self.metrics_collector = EpisodeMetricsCollector(
            warmup_ticks=self.sim_cfg.get("warmup_ticks", 200)
        )
        self.reset(seed=self.base_seed)

    def neighbors(self, node: Any) -> List[Any]:
        return self._neighbors[node]

    def schedule_failure(self, tick: int, edge: Tuple[Any, Any], recover_tick: Optional[int] = None) -> None:
        u, v = edge
        if tick not in self.failure_schedule:
            self.failure_schedule[tick] = []
        self.failure_schedule[tick].append(((u, v), False))
        if recover_tick is not None:
            if recover_tick not in self.failure_schedule:
                self.failure_schedule[recover_tick] = []
            self.failure_schedule[recover_tick].append(((u, v), True))

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_tick = 0
        self.next_pkt_id = 1
        self.routing_backlog = {n: [] for n in self.nodes}

        # Reset all links
        for link in self.links.values():
            link.queue.clear()
            link.pipe.clear()
            link.up = True
            link.packets_sent_this_tick = 0
            link.total_sent = 0
            link.total_dropped = 0

        # Reset traffic generator
        flows = self.traffic_cfg.get("flows")
        if flows is None:
            # Generate default flows
            pairs = build_overlapping_flows(self.graph, num_hotspots=4, seed=int(self.rng.integers(0, 100000)))
            target_rho = self.traffic_cfg.get("target_rho", 0.7)
            bursty = self.traffic_cfg.get("bursty", False)
            flows, _ = calibrate_load(self.graph, pairs, target_rho, self.capacity, bursty)
        self.traffic_generator = TrafficGenerator(flows, seed=int(self.rng.integers(0, 100000)))
        self.metrics_collector.reset()

    def get_state(self, node: Any, dest: Any) -> Tuple[Any, ...]:
        bins = []
        for nbr in self._neighbors[node]:
            link = self.links[(node, nbr)]
            bins.append(link.congestion_bin(self.n_bins))
        return (node, dest, *bins)

    def _apply_failures(self) -> None:
        if self.current_tick in self.failure_schedule:
            for (u, v), recover in self.failure_schedule[self.current_tick]:
                for d1, d2 in [(u, v), (v, u)]:
                    if (d1, d2) in self.links:
                        link = self.links[(d1, d2)]
                        link.up = recover
                        if not recover:
                            purged = link.purge_on_failure()
                            for pkt in purged:
                                self.metrics_collector.record_drop(self.current_tick, "link_down")
                                # If packet has pending transition, terminal drop update
                                if pkt.pending_transition:
                                    s = pkt.pending_transition["state"]
                                    a = pkt.pending_transition["action"]
                                    h_delay = max(1, self.current_tick - pkt.pending_transition["enqueue_tick"])
                                    r = -h_delay - self.p_drop
                                    if pkt.pending_transition.get("high_cong"):
                                        r -= self.p_cong
                                    trans = (s, a, r, None, True, {"reason": "link_down", "pkt_id": pkt.id})
                                    if hasattr(self, "_active_agent") and self._active_agent:
                                        self._active_agent.observe(trans)

    def step(self, agent: Any) -> Dict[str, Any]:
        self._active_agent = agent
        # Apply failure events scheduled for current_tick
        self._apply_failures()
        if hasattr(agent, "on_tick"):
            agent.on_tick(self)

        # 1. INJECT
        if self.traffic_generator:
            new_pkts, self.next_pkt_id = self.traffic_generator.generate_packets(
                self.current_tick, self.next_pkt_id, self.default_ttl
            )
            for pkt in new_pkts:
                self.metrics_collector.record_injection(self.current_tick, 1)
                self.routing_backlog[pkt.src].append(pkt)

        # 2. ARRIVE
        for (u, v), link in self.links.items():
            ready_pkts = []
            while link.pipe and link.pipe[0][0] <= self.current_tick:
                _, pkt = link.pipe.popleft()
                ready_pkts.append(pkt)

            for pkt in ready_pkts:
                pkt.current_node = v
                pkt.hops += 1
                pkt.path.append(v)
                hop_delay = max(1, self.current_tick - pkt.enqueue_tick)

                # Check loop
                is_loop = (pkt.path.count(v) > 1)

                if v == pkt.dst:
                    # Delivered!
                    total_delay = self.current_tick - pkt.birth_tick
                    self.metrics_collector.record_delivery(self.current_tick, total_delay, pkt.hops)
                    r = -hop_delay + self.r_deliv
                    if pkt.pending_transition and pkt.pending_transition.get("high_cong"):
                        r -= self.p_cong
                    if is_loop:
                        r -= self.p_loop
                    if pkt.pending_transition:
                        s = pkt.pending_transition["state"]
                        a = pkt.pending_transition["action"]
                        agent.observe((s, a, r, None, True, {"reason": "delivered", "pkt_id": pkt.id}))
                        pkt.pending_transition = None
                else:
                    # Intermediate arrival
                    r = -hop_delay
                    if pkt.pending_transition and pkt.pending_transition.get("high_cong"):
                        r -= self.p_cong
                    if is_loop:
                        r -= self.p_loop
                    next_s = self.get_state(v, pkt.dst)
                    if pkt.pending_transition:
                        s = pkt.pending_transition["state"]
                        a = pkt.pending_transition["action"]
                        agent.observe((s, a, r, next_s, False, {"reason": "hop", "pkt_id": pkt.id}))
                        pkt.pending_transition = None
                    self.routing_backlog[v].append(pkt)

        # 3. ROUTE
        for node in self.nodes:
            pkts = self.routing_backlog[node]
            self.routing_backlog[node] = []
            for pkt in pkts:
                pkt.ttl -= 1
                if pkt.ttl <= 0:
                    self.metrics_collector.record_drop(self.current_tick, "ttl")
                    continue

                nbrs = self._neighbors[node]
                if not nbrs:
                    self.metrics_collector.record_drop(self.current_tick, "no_route")
                    continue

                act_idx = agent.choose_next_hop(self, pkt, node)
                chosen_nbr = nbrs[act_idx]
                link = self.links[(node, chosen_nbr)]

                s = self.get_state(node, pkt.dst)
                high_cong = (link.congestion_bin(self.n_bins) == (self.n_bins - 1))

                if not link.up:
                    self.metrics_collector.record_drop(self.current_tick, "link_down")
                    r = -1.0 - self.p_drop
                    agent.observe((s, act_idx, r, None, True, {"reason": "link_down", "pkt_id": pkt.id}))
                    continue

                if len(link.queue) >= link.buffer_size:
                    self.metrics_collector.record_drop(self.current_tick, "queue_full")
                    r = -1.0 - self.p_drop
                    agent.observe((s, act_idx, r, None, True, {"reason": "queue_full", "pkt_id": pkt.id}))
                    continue

                pkt.enqueue_tick = self.current_tick
                pkt.pending_transition = {
                    "state": s,
                    "action": act_idx,
                    "enqueue_tick": self.current_tick,
                    "high_cong": high_cong,
                }
                link.queue.append(pkt)

        # 4. TRANSMIT
        for (u, v), link in self.links.items():
            link.transmit(self.current_tick)

        # 5. LOG
        for (u, v), link in self.links.items():
            util = link.packets_sent_this_tick / float(max(1, link.capacity))
            self.metrics_collector.record_link_util(self.current_tick, (u, v), util)

        tick_stats = {
            "tick": self.current_tick,
            "in_flight": sum(len(l.queue) + len(l.pipe) for l in self.links.values()),
        }
        self.current_tick += 1
        return tick_stats

    def count_packets_in_flight(self) -> int:
        return sum(len(l.queue) + len(l.pipe) for l in self.links.values()) + sum(
            len(b) for b in self.routing_backlog.values()
        )

    def run(self, agent: Any, n_ticks: int, learn: bool = True) -> Dict[str, float]:
        agent.begin_episode(training=learn)
        for _ in range(n_ticks):
            self.step(agent)
        ep_stats = agent.end_episode()
        metrics = self.metrics_collector.compute(n_ticks)
        metrics.update(ep_stats)
        return metrics
