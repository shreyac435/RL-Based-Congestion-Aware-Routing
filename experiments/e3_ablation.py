"""Experiment E3: Ablation Study (Q-Learning vs SARSA vs Q-Routing Without Congestion)."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

from sim.topologies import create_abilene
from sim.traffic import build_overlapping_flows, calibrate_load
from sim.env import RoutingEnv
from agents.dijkstra import DijkstraAgent
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.qrouting_nocong import QRoutingNoCongAgent
from experiments.runner import compute_ci, save_raw_csv, save_figure, ensure_dirs


def run_e3(quick: bool = False):
    ensure_dirs()
    print("=" * 60)
    print(f"Running Experiment E3: Ablation Study (quick={quick})")
    print("=" * 60)

    g = create_abilene()
    pairs = build_overlapping_flows(g, num_hotspots=8, seed=42)
    flows, _ = calibrate_load(g, pairs, target_rho=1.1, link_capacity=2, bursty=True)

    traffic_cfg = {"flows": flows, "bursty": True}
    sim_cfg = {"warmup_ticks": 100 if quick else 200, "capacity": 2, "buffer_size": 20}

    n_train = 40 if quick else 150
    n_eval_ticks = 400 if quick else 800
    eval_seeds = list(range(2000, 2005 if quick else 2010))

    methods = [
        ("Dijkstra", lambda: DijkstraAgent(g)),
        ("Q-Learning (Full)", lambda: QLearningAgent(alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=int(n_train * 0.6), seed=42)),
        ("SARSA", lambda: SARSAAgent(alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=int(n_train * 0.6), seed=42)),
        ("Q-Routing (No Congestion)", lambda: QRoutingNoCongAgent(alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=int(n_train * 0.6), seed=42)),
    ]

    raw_rows = []
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for name, agent_builder in methods:
        print(f"\nTraining and evaluating: {name}...")
        agent = agent_builder()
        if name != "Dijkstra":
            train_env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=1000)
            for ep in range(n_train):
                train_env.reset(seed=1000 + ep)
                train_env.run(agent, n_ticks=300 if quick else 500, learn=True)

        res_dict: Dict[str, List[float]] = {"avg_delay": [], "loss_pct": [], "throughput": [], "load_balance": []}
        for s in eval_seeds:
            env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            r = env.run(agent, n_ticks=n_eval_ticks, learn=False)
            for m in res_dict.keys():
                res_dict[m].append(r[m])
                raw_rows.append({
                    "experiment": "E3",
                    "topology": "Abilene",
                    "load": "Bursty-High",
                    "method": name,
                    "seed": s,
                    "metric": m,
                    "value": r[m],
                })

        summary[name] = {m: compute_ci(res_dict[m]) for m in res_dict.keys()}
        print(f"  {name:26s} | Avg Delay: {summary[name]['avg_delay'][0]:.2f} +/- {summary[name]['avg_delay'][1]:.2f} | Loss: {summary[name]['loss_pct'][0]:.2f}%")

    save_raw_csv(raw_rows, "e3_ablation_results.csv")

    # Plot Ablation Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    names = [m[0] for m in methods]
    x = np.arange(len(names))

    delays = [summary[n]["avg_delay"][0] for n in names]
    delay_errs = [summary[n]["avg_delay"][1] for n in names]
    ax1.bar(x, delays, yerr=delay_errs, capsize=5, color=["#4575b4", "#d73027", "#fc8d59", "#91bfdb"])
    ax1.set_ylabel("Average Delay (ticks)")
    ax1.set_title("E3: Average Delay under Bursty-High Load")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    losses = [summary[n]["loss_pct"][0] for n in names]
    loss_errs = [summary[n]["loss_pct"][1] for n in names]
    ax2.bar(x, losses, yerr=loss_errs, capsize=5, color=["#4575b4", "#d73027", "#fc8d59", "#91bfdb"])
    ax2.set_ylabel("Packet Loss (%)")
    ax2.set_title("E3: Packet Loss Rate under Bursty-High Load")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "e3_ablation_comparison")
    print("E3 complete!\n")


if __name__ == "__main__":
    run_e3(quick=True)
