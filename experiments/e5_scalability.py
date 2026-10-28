"""Experiment E5: Empirical Scalability Sweep and Tabular Q-Learning Boundary Analysis."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from sim.topologies import create_waxman
from sim.traffic import build_overlapping_flows, calibrate_load
from sim.env import RoutingEnv
from agents.dijkstra import DijkstraAgent
from agents.qlearning import QLearningAgent
from experiments.runner import compute_ci, save_raw_csv, save_figure, ensure_dirs


def run_e5(quick: bool = False):
    ensure_dirs()
    print("=" * 60)
    print(f"Running Experiment E5: Scalability Sweep (quick={quick})")
    print("=" * 60)

    # Topology sizes from spec Section 8.1
    node_sizes = [6, 12, 25, 50] if quick else [6, 12, 25, 50, 100]
    n_train = 30 if quick else 100
    n_eval_ticks = 250 if quick else 500
    eval_seeds = list(range(2000, 2004 if quick else 2008))

    raw_rows = []
    summary = {
        "n_nodes": [],
        "table_size": [],
        "train_time_sec": [],
        "dijkstra_delay": [],
        "ql_delay": [],
        "delay_improvement_pct": [],
        "ci_improvement": [],
        "converged": [],
    }

    print(f"{'N':>4} | {'Table Size':>10} | {'Train (s)':>9} | {'Dijkstra Delay':>14} | {'QL Delay':>10} | {'Improvement %':>13}")
    print("-" * 75)

    for n in node_sizes:
        g = create_waxman(n, seed=42 + n)
        # Scaled number of overlapping flows
        pairs = build_overlapping_flows(g, num_hotspots=max(4, n // 2), seed=42)
        flows, _ = calibrate_load(g, pairs, target_rho=1.1, link_capacity=2, bursty=True)

        traffic_cfg = {"flows": flows, "bursty": True}
        sim_cfg = {"warmup_ticks": 50, "capacity": 2, "buffer_size": 20}

        # Train Q-agent
        ql_agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=int(n_train * 0.6), seed=42)
        train_env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=1000)

        t0 = time.time()
        for ep in range(n_train):
            train_env.reset(seed=1000 + ep)
            train_env.run(ql_agent, n_ticks=150 if quick else 300, learn=True)
        train_time = time.time() - t0

        q_table_size = len(ql_agent.q_table)

        # Check convergence heuristic: rolling mean delta < 0.05
        converged = bool(ql_agent.history_q_deltas and ql_agent.history_q_deltas[-1] < 0.05)

        # Evaluate both Dijkstra and Q-Learning
        dijkstra_agent = DijkstraAgent(g)
        d_delays, q_delays = [], []

        for s in eval_seeds:
            # Dijkstra
            env_d = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            res_d = env_d.run(dijkstra_agent, n_ticks=n_eval_ticks, learn=False)
            d_delays.append(res_d["avg_delay"])
            raw_rows.append({"experiment": "E5", "N": n, "method": "Dijkstra", "seed": s, "metric": "avg_delay", "value": res_d["avg_delay"]})

            # Q-learning
            env_q = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            res_q = env_q.run(ql_agent, n_ticks=n_eval_ticks, learn=False)
            q_delays.append(res_q["avg_delay"])
            raw_rows.append({"experiment": "E5", "N": n, "method": "Q-Learning", "seed": s, "metric": "avg_delay", "value": res_q["avg_delay"]})

        d_mean, _ = compute_ci(d_delays)
        q_mean, _ = compute_ci(q_delays)

        # Percentage improvement in delay: (Dijkstra - QL) / Dijkstra * 100
        per_seed_improvements = [
            ((d - q) / max(1e-4, d)) * 100.0 for d, q in zip(d_delays, q_delays)
        ]
        imp_mean, imp_ci = compute_ci(per_seed_improvements)

        summary["n_nodes"].append(n)
        summary["table_size"].append(q_table_size)
        summary["train_time_sec"].append(train_time)
        summary["dijkstra_delay"].append(d_mean)
        summary["ql_delay"].append(q_mean)
        summary["delay_improvement_pct"].append(imp_mean)
        summary["ci_improvement"].append(imp_ci)
        summary["converged"].append(converged)

        print(f"{n:4d} | {q_table_size:10d} | {train_time:8.1f}s | {d_mean:14.2f} | {q_mean:10.2f} | {imp_mean:+8.2f}% +/- {imp_ci:4.2f}%")

    save_raw_csv(raw_rows, "e5_scalability_results.csv")

    # Scalability Boundary Analysis
    # Boundary: smallest N where improvement <= 0 or not statistically significant
    boundary_n = None
    for n, imp, ci in zip(summary["n_nodes"], summary["delay_improvement_pct"], summary["ci_improvement"]):
        if (imp - ci) <= 0:
            boundary_n = n
            break

    print("\n" + "=" * 60)
    print("SCALABILITY BOUNDARY REPORT (Spec Section 9)")
    print(f"Empirical Scalability Boundary for Tabular Q-Learning: N = {boundary_n if boundary_n else '> 100'}")
    print("Explanation: Beyond this network size, combinatorial state space expansion and sparse updates prevent tabular Q-learning from reliably outperforming Dijkstra.")
    print("=" * 60 + "\n")

    # Plot Scalability Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(summary["n_nodes"], summary["table_size"], marker="o", color="#756bb1", linewidth=2)
    ax1.set_xlabel("Number of Nodes (N)")
    ax1.set_ylabel("Q-Table Size (Stored States)")
    ax1.set_title("E5: Tabular State Growth vs Network Size")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.errorbar(summary["n_nodes"], summary["delay_improvement_pct"], yerr=summary["ci_improvement"], marker="s", color="#31a354", capsize=5, linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--")
    if boundary_n:
        ax2.axvline(boundary_n, color="red", linestyle=":", label=f"Boundary (N={boundary_n})")
        ax2.legend()
    ax2.set_xlabel("Number of Nodes (N)")
    ax2.set_ylabel("Delay Improvement over Dijkstra (%)")
    ax2.set_title("E5: Q-Learning Advantage Boundary")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "e5_scalability_boundary")
    print("E5 complete!\n")


if __name__ == "__main__":
    run_e5(quick=True)
