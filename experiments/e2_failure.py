"""Experiment E2: Link failure dynamics and recovery analysis."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from sim.topologies import create_abilene
from sim.traffic import build_overlapping_flows, calibrate_load
from sim.env import RoutingEnv
from agents.dijkstra import DijkstraAgent
from agents.qlearning import QLearningAgent
from experiments.runner import compute_ci, save_raw_csv, save_figure, ensure_dirs


def run_e2(quick: bool = False):
    ensure_dirs()
    print("=" * 60)
    print(f"Running Experiment E2: Link Failure Response (quick={quick})")
    print("=" * 60)

    g = create_abilene()
    pairs = build_overlapping_flows(g, num_hotspots=8, seed=42)
    flows, _ = calibrate_load(g, pairs, target_rho=0.8, link_capacity=2, bursty=False)

    traffic_cfg = {"flows": flows}
    sim_cfg = {"warmup_ticks": 50, "capacity": 2, "buffer_size": 20}

    # Find the most loaded Dijkstra link
    link_counts: Dict[Tuple[Any, Any], int] = {}
    for u, v in pairs:
        p = DijkstraAgent(g).routes[u][v]
        for i in range(len(p) - 1):
            e = (p[i], p[i + 1])
            link_counts[e] = link_counts.get(e, 0) + 1
    fail_edge = max(link_counts.items(), key=lambda x: x[1])[0]
    print(f"Target failure edge (highest Dijkstra load): {fail_edge}")

    n_train = 40 if quick else 150
    total_ticks = 400 if quick else 800
    fail_tick = 200 if quick else 400

    # Train base Q-agent
    print("Pre-training Q-agent on normal topology...")
    base_agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=int(n_train * 0.6), seed=42)
    train_env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=1000)
    for ep in range(n_train):
        train_env.reset(seed=1000 + ep)
        train_env.run(base_agent, n_ticks=250, learn=True)

    # Time-series collection across seeds
    eval_seeds = list(range(2000, 2005 if quick else 2010))
    methods = ["Dijkstra", "Q-Learning (Frozen)", "Q-Learning (Online)"]

    # Tracking per tick: method -> tick -> list of delays
    tick_delays: Dict[str, Dict[int, List[float]]] = {m: {t: [] for t in range(total_ticks)} for m in methods}
    raw_rows = []

    for s in eval_seeds:
        for m in methods:
            env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            env.schedule_failure(tick=fail_tick, edge=fail_edge)

            if m == "Dijkstra":
                agent = DijkstraAgent(g, detection_delay=15)
                learn = False
            elif m == "Q-Learning (Frozen)":
                agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon_start=0.0, seed=s)
                agent.q_table = {k: v.copy() for k, v in base_agent.q_table.items()}
                learn = False
            else:  # Online
                agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon_start=0.05, epsilon_end=0.05, seed=s)
                agent.q_table = {k: v.copy() for k, v in base_agent.q_table.items()}
                learn = True

            agent.begin_episode(training=learn)
            # Step-by-step logging
            for t in range(total_ticks):
                env.step(agent)
                if env.metrics_collector.delivered_delays:
                    # recent delay
                    recent = env.metrics_collector.delivered_delays[-5:]
                    tick_delays[m][t].append(float(np.mean(recent)))
                else:
                    tick_delays[m][t].append(2.0)

            res = env.metrics_collector.compute(total_ticks)
            for metric_name, val in res.items():
                raw_rows.append({
                    "experiment": "E2",
                    "topology": "Abilene",
                    "load": "Medium",
                    "method": m,
                    "seed": s,
                    "metric": metric_name,
                    "value": val,
                })

    save_raw_csv(raw_rows, "e2_failure_results.csv")

    # Plot failure time-series
    fig, ax = plt.subplots(figsize=(9, 5))
    ticks = list(range(total_ticks))
    colors = {"Dijkstra": "#4575b4", "Q-Learning (Frozen)": "#fdae61", "Q-Learning (Online)": "#d73027"}

    for m in methods:
        means = [float(np.mean(tick_delays[m][t])) if tick_delays[m][t] else 2.0 for t in ticks]
        # Smooth with rolling window
        smoothed = np.convolve(means, np.ones(15) / 15.0, mode="same")
        ax.plot(ticks[20:-20], smoothed[20:-20], label=m, color=colors[m], linewidth=2)

    ax.axvline(x=fail_tick, color="black", linestyle="--", alpha=0.8, label=f"Link Failure at tick {fail_tick}")
    ax.set_xlabel("Simulation Tick")
    ax.set_ylabel("Rolling Average Delay (ticks)")
    ax.set_title(f"E2: Network Delay Dynamics Under Link Failure on Edge {fail_edge}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    save_figure(fig, "e2_failure_dynamics")
    print("E2 complete!\n")


if __name__ == "__main__":
    run_e2(quick=True)
