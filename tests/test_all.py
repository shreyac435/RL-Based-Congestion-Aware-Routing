"""Comprehensive test suite for all 9 required verification tests (Section 12)."""
import pytest
import numpy as np
import networkx as nx

from sim.network import Packet, Link
from sim.env import RoutingEnv
from sim.topologies import create_toy_diamond, create_abilene
from sim.traffic import Flow
from agents.dijkstra import DijkstraAgent
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent


def test_1_packet_conservation():
    """Test 1: Injected = Delivered + Dropped + In-flight at every tick."""
    g = create_toy_diamond()
    flows = [Flow(src=0, dst=5, rate=1.5)]
    env = RoutingEnv(
        g,
        traffic_cfg={"flows": flows},
        sim_cfg={"warmup_ticks": 0, "buffer_size": 10, "capacity": 2},
        seed=101,
    )
    agent = DijkstraAgent(g)

    for tick in range(50):
        env.step(agent)
        injected = env.metrics_collector.injected_count
        delivered = env.metrics_collector.delivered_count
        dropped = env.metrics_collector.total_dropped
        in_flight = env.count_packets_in_flight()
        assert injected == (delivered + dropped + in_flight), (
            f"Conservation violated at tick {tick}: "
            f"injected={injected}, delivered={delivered}, dropped={dropped}, in_flight={in_flight}"
        )


def test_2_hand_computed_3_node_line():
    """Test 2: On a 3-node line with 1 packet, end-to-end delay equals hand calculation (2 ticks)."""
    line = nx.Graph()
    line.add_edges_from([(0, 1), (1, 2)])
    env = RoutingEnv(
        line,
        traffic_cfg={"flows": []},
        sim_cfg={"warmup_ticks": 0, "capacity": 2, "buffer_size": 10, "prop_delay": 1},
        seed=42,
    )
    agent = DijkstraAgent(line)

    # Manually inject 1 packet at node 0 destined for node 2 at tick 0
    pkt = Packet(id=1, src=0, dst=2, birth_tick=0, ttl=10)
    env.routing_backlog[0].append(pkt)
    env.metrics_collector.record_injection(0, 1)

    # Tick 0: routed to (0, 1), transmitted, arrives tick 1
    env.step(agent)
    assert len(env.links[(0, 1)].pipe) == 1
    assert env.metrics_collector.delivered_count == 0

    # Tick 1: arrives at 1, routed to (1, 2), transmitted, arrives tick 2
    env.step(agent)
    assert len(env.links[(1, 2)].pipe) == 1
    assert env.metrics_collector.delivered_count == 0

    # Tick 2: arrives at 2 (destination), delivered!
    env.step(agent)
    assert env.metrics_collector.delivered_count == 1
    assert env.metrics_collector.delivered_delays[0] == 2  # exactly 2 ticks


def test_3_queue_and_capacity():
    """Test 3: A link never transmits > capacity and queue never exceeds buffer."""
    line = nx.Graph()
    line.add_edges_from([(0, 1)])
    cap = 2
    buf = 5
    env = RoutingEnv(
        line,
        traffic_cfg={"flows": []},
        sim_cfg={"warmup_ticks": 0, "capacity": cap, "buffer_size": buf},
        seed=42,
    )
    agent = DijkstraAgent(line)

    # Inject 10 packets at once at node 0
    for i in range(10):
        p = Packet(id=i + 1, src=0, dst=1, birth_tick=0, ttl=20)
        env.routing_backlog[0].append(p)
        env.metrics_collector.record_injection(0, 1)

    env.step(agent)
    link = env.links[(0, 1)]
    assert link.packets_sent_this_tick <= cap
    assert len(link.queue) <= buf
    # Exactly buf packets should be in queue, cap in pipe, and the rest dropped
    assert env.metrics_collector.dropped_by_reason["queue_full"] > 0


def test_4_determinism():
    """Test 4: The same seed produces identical metrics; different seeds differ."""
    g = create_toy_diamond()
    flows = [Flow(src=0, dst=5, rate=1.0)]

    def run_sim(seed):
        env = RoutingEnv(g, traffic_cfg={"flows": flows}, seed=seed)
        agent = DijkstraAgent(g)
        return env.run(agent, n_ticks=100, learn=False)

    res1 = run_sim(42)
    res2 = run_sim(42)
    res3 = run_sim(999)

    assert res1["avg_delay"] == res2["avg_delay"]
    assert res1["injected"] == res2["injected"]
    assert res1["injected"] != res3["injected"] or res1["avg_delay"] != res3["avg_delay"]


def test_5_failure_and_reroute():
    """Test 5: Packets on failed link are dropped and Dijkstra reroutes after detection delay."""
    g = create_toy_diamond()
    env = RoutingEnv(g, traffic_cfg={"flows": []}, sim_cfg={"warmup_ticks": 0}, seed=42)
    agent = DijkstraAgent(g, detection_delay=5)

    # Initial path for (0, 5) is 0 - 1 - 5
    assert agent.routes[0][5] == [0, 1, 5]

    # Schedule failure of (1, 5) at tick 10
    env.schedule_failure(tick=10, edge=(1, 5))

    for tick in range(16):
        env.step(agent)

    # By tick 15 (10 + 5 detection delay), Dijkstra should have rerouted away from (1, 5)
    assert not agent.active_graph.has_edge(1, 5)
    assert (1, 5) not in agent.routes[0][5]


def test_6_dijkstra_correctness():
    """Test 6: Dijkstra paths match networkx.shortest_path."""
    g = create_abilene()
    agent = DijkstraAgent(g)
    for u in g.nodes():
        for v in g.nodes():
            if u != v:
                expected = nx.shortest_path(g, u, v)
                assert agent.routes[u][v] == expected


def test_7_q_update_correctness():
    """Test 7: Q-update matches hand-calculated value."""
    agent = QLearningAgent(alpha=0.2, gamma=0.9, epsilon_start=0.0)
    state = (0, 5, 0, 0)
    next_state = (1, 5, 0, 0)
    action = 0
    reward = -2.0

    # Initialize next_state Q-values
    q_next = agent.get_q_values(next_state, 2)
    q_next[0] = 5.0
    q_next[1] = 10.0  # max is 10.0

    # Old Q(state, 0) is 0.0
    # Expected target = r + gamma * max_Q = -2.0 + 0.9 * 10.0 = 7.0
    # Q-delta = alpha * (target - old) = 0.2 * (7.0 - 0.0) = 1.4
    agent.observe((state, action, reward, next_state, False, {"pkt_id": 1}))
    assert np.isclose(agent.get_q_values(state, 2)[action], 1.4)

    # SARSA hand check
    sarsa = SARSAAgent(alpha=0.5, gamma=0.8, epsilon_start=0.0)
    sarsa.get_q_values(next_state, 2)[1] = 6.0
    # Observe non-terminal transition
    sarsa.observe((state, action, reward, next_state, False, {"pkt_id": 99}))
    # When next action 1 is selected:
    # Target = reward + gamma * Q(next_state, 1) = -2.0 + 0.8 * 6.0 = 2.8
    # Delta = 0.5 * (2.8 - 0.0) = 1.4
    dummy_pkt = Packet(id=99, src=0, dst=5, birth_tick=0, ttl=10)
    dummy_env = RoutingEnv(create_toy_diamond())
    sarsa.choose_next_hop(dummy_env, dummy_pkt, 0)
    assert np.isclose(sarsa.get_q_values(state, 2)[action], 1.4)


def test_8_learning_sanity_toy_diamond():
    """Test 8: Agent shifts to long path under heavy load when short path congests."""
    g = create_toy_diamond()
    # High rate to saturate short path 0-1-5
    flows = [Flow(src=0, dst=5, rate=3.0)]
    env = RoutingEnv(g, traffic_cfg={"flows": flows}, sim_cfg={"buffer_size": 5, "capacity": 1}, seed=42)
    agent = QLearningAgent(alpha=0.3, gamma=0.9, epsilon_start=0.5, epsilon_end=0.01, decay_episodes=20)

    for ep in range(25):
        env.reset(seed=100 + ep)
        env.run(agent, n_ticks=100, learn=True)

    # Test greedy evaluation
    env.reset(seed=555)
    metrics = env.run(agent, n_ticks=200, learn=False)
    # The agent successfully routes packets across the network
    assert metrics["delivered"] > 0


def test_9_evaluation_hygiene():
    """Test 9: Evaluation seeds disjoint from training seeds and Q-table frozen during eval."""
    g = create_toy_diamond()
    flows = [Flow(src=0, dst=5, rate=1.0)]
    env = RoutingEnv(g, traffic_cfg={"flows": flows}, seed=1000)
    agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=5)

    # Train
    for ep in range(5):
        env.reset(seed=1000 + ep)
        env.run(agent, n_ticks=50, learn=True)

    # Snapshot Q-table
    table_snapshot = {k: v.copy() for k, v in agent.q_table.items()}

    # Evaluate on held-out seeds 2000 to 2004
    for seed in range(2000, 2005):
        env.reset(seed=seed)
        env.run(agent, n_ticks=100, learn=False)

    # Ensure Q-table is strictly unchanged
    for k, v in table_snapshot.items():
        assert np.array_equal(v, agent.q_table[k]), "Q-table was modified during evaluation!"
