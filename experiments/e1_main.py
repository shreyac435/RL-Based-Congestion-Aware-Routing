"""Experiment E1: Q-Learning vs Dijkstra across Low, Medium, and Bursty-High loads."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import time
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

from sim.topologies import create_abilene
from sim.traffic import build_overlapping_flows, calibrate_load
from sim.env import RoutingEnv
from agents.dijkstra import DijkstraAgent
from agents.qlearning import QLearningAgent
from experiments.runner import compute_ci, save_raw_csv, save_figure, ensure_dirs


def run_e1(quick: bool = False):
    ensure_dirs()
    print("=" * 60)
    print(f"Running Experiment E1: Main Performance Benchmark (quick={quick})")
    print("=" * 60)

    g = create_abilene()
    pairs = build_overlapping_flows(g, num_hotspots=8, seed=42)

    scenarios = [
        {"load": "Low", "rho": 0.4, "bursty": False},
        {"load": "Medium", "rho": 0.7, "bursty": False},
        {"load": "Bursty-High", "rho": 1.1, "bursty": True},
    ]

    n_train = 50 if quick else 200
    n_eval_ticks = 400 if quick else 800
    eval_seeds = list(range(2000, 2005 if quick else 2010))

    raw_rows = []
    summary_data = {
        "load": [],
        "metric": [],
        "dijkstra_mean": [],
        "dijkstra_ci": [],
        "ql_mean": [],
        "ql_ci": [],
    }

    metrics_to_track = ["avg_delay", "p95_delay", "loss_pct", "throughput", "load_balance", "avg_hops"]

    for sc in scenarios:
        load_name = sc["load"]
        target_rho = sc["rho"]
        bursty = sc["bursty"]
        flows, achieved_rho = calibrate_load(g, pairs, target_rho, link_capacity=2, bursty=bursty)
        print(f"\n--- Scenario: {load_name} (Target rho: {target_rho}, Achieved rho: {achieved_rho:.2f}) ---")

        traffic_cfg = {"flows": flows, "bursty": bursty}
        sim_cfg = {"warmup_ticks": 100 if quick else 200, "capacity": 2, "buffer_size": 20}

        # 1. Train Q-learning agent
        print(f"Training Q-learning agent ({n_train} episodes)...")
        ql_agent = QLearningAgent(
            alpha=0.2,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_episodes=int(n_train * 0.6),
            seed=42,
        )

        train_env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=1000)
        t0 = time.time()
        for ep in range(n_train):
            train_env.reset(seed=1000 + ep)
            train_env.run(ql_agent, n_ticks=300 if quick else 500, learn=True)
        train_time = time.time() - t0
        print(f"Training completed in {train_time:.1f}s. Q-table size: {len(ql_agent.q_table)} states.")

        # Save trained high-load agent for demo
        if load_name == "Bursty-High":
            with open("results/models/qlearning_abilene_high.pkl", "wb") as f:
                pickle.dump(ql_agent, f)

        # 2. Evaluate both agents on exact same seeds
        dijkstra_results: Dict[str, List[float]] = {m: [] for m in metrics_to_track}
        ql_results: Dict[str, List[float]] = {m: [] for m in metrics_to_track}

        dijkstra_agent = DijkstraAgent(g)

        for s in eval_seeds:
            # Dijkstra eval
            env_d = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            res_d = env_d.run(dijkstra_agent, n_ticks=n_eval_ticks, learn=False)
            for m in metrics_to_track:
                val = res_d[m]
                dijkstra_results[m].append(val)
                raw_rows.append({
                    "experiment": "E1",
                    "topology": "Abilene",
                    "load": load_name,
                    "method": "Dijkstra",
                    "seed": s,
                    "metric": m,
                    "value": val,
                })

            # Q-learning eval
            env_q = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            res_q = env_q.run(ql_agent, n_ticks=n_eval_ticks, learn=False)
            for m in metrics_to_track:
                val = res_q[m]
                ql_results[m].append(val)
                raw_rows.append({
                    "experiment": "E1",
                    "topology": "Abilene",
                    "load": load_name,
                    "method": "Q-Learning",
                    "seed": s,
                    "metric": m,
                    "value": val,
                })

        # Summarize
        for m in metrics_to_track:
            d_mean, d_ci = compute_ci(dijkstra_results[m])
            q_mean, q_ci = compute_ci(ql_results[m])
            summary_data["load"].append(load_name)
            summary_data["metric"].append(m)
            summary_data["dijkstra_mean"].append(d_mean)
            summary_data["dijkstra_ci"].append(d_ci)
            summary_data["ql_mean"].append(q_mean)
            summary_data["ql_ci"].append(q_ci)

            print(f"  {m:14s} | Dijkstra: {d_mean:6.2f} +/- {d_ci:4.2f} | Q-Learning: {q_mean:6.2f} +/- {q_ci:4.2f}")

    save_raw_csv(raw_rows, "e1_results.csv")

    # Generate Figures
    _plot_e1_figures(summary_data)
    print("E1 complete!\n")


def _plot_e1_figures(summary: Dict[str, List[Any]]):
    loads = ["Low", "Medium", "Bursty-High"]
    x = np.arange(len(loads))
    width = 0.35

    def get_vals(metric_name: str):
        d_means, d_cis, q_means, q_cis = [], [], [], []
        for l in loads:
            idx = [i for i, (ld, m) in enumerate(zip(summary["load"], summary["metric"])) if ld == l and m == metric_name][0]
            d_means.append(summary["dijkstra_mean"][idx])
            d_cis.append(summary["dijkstra_ci"][idx])
            q_means.append(summary["ql_mean"][idx])
            q_cis.append(summary["ql_ci"][idx])
        return d_means, d_cis, q_means, q_cis

    # 1. Average Delay
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dm, dc, qm, qc = get_vals("avg_delay")
    ax.bar(x - width/2, dm, width, yerr=dc, capsize=5, label="Dijkstra (Static)", color="#4575b4")
    ax.bar(x + width/2, qm, width, yerr=qc, capsize=5, label="Q-Learning (Adaptive)", color="#d73027")
    ax.set_ylabel("Average Packet Delay (ticks)")
    ax.set_title("E1: Average Packet Delay by Traffic Load (Abilene Topology)")
    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(fig, "e1_delay_by_load")

    # 2. Packet Loss %
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dm, dc, qm, qc = get_vals("loss_pct")
    ax.bar(x - width/2, dm, width, yerr=dc, capsize=5, label="Dijkstra (Static)", color="#4575b4")
    ax.bar(x + width/2, qm, width, yerr=qc, capsize=5, label="Q-Learning (Adaptive)", color="#d73027")
    ax.set_ylabel("Packet Loss (%)")
    ax.set_title("E1: Packet Loss Rate by Traffic Load")
    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(fig, "e1_loss_by_load")

    # 3. Throughput
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dm, dc, qm, qc = get_vals("throughput")
    ax.bar(x - width/2, dm, width, yerr=dc, capsize=5, label="Dijkstra (Static)", color="#4575b4")
    ax.bar(x + width/2, qm, width, yerr=qc, capsize=5, label="Q-Learning (Adaptive)", color="#d73027")
    ax.set_ylabel("Throughput (delivered packets / tick)")
    ax.set_title("E1: System Throughput by Traffic Load")
    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(fig, "e1_throughput_by_load")


if __name__ == "__main__":
    run_e1(quick=True)
