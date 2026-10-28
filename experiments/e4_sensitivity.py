"""Experiment E4: Hyperparameter Sensitivity Analysis."""
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
from agents.qlearning import QLearningAgent
from experiments.runner import compute_ci, save_raw_csv, save_figure, ensure_dirs


def run_e4(quick: bool = False):
    ensure_dirs()
    print("=" * 60)
    print(f"Running Experiment E4: Sensitivity Analysis (quick={quick})")
    print("=" * 60)

    g = create_abilene()
    pairs = build_overlapping_flows(g, num_hotspots=8, seed=42)
    flows, _ = calibrate_load(g, pairs, target_rho=0.9, link_capacity=2, bursty=False)

    traffic_cfg = {"flows": flows}
    sim_cfg = {"warmup_ticks": 50, "capacity": 2, "buffer_size": 20}

    n_train = 30 if quick else 100
    n_eval_ticks = 300 if quick else 600
    eval_seeds = list(range(2000, 2004 if quick else 2008))

    # Sweeps
    alphas = [0.05, 0.1, 0.2, 0.5] if quick else [0.05, 0.1, 0.2, 0.3, 0.5]
    gammas = [0.8, 0.9, 0.95] if quick else [0.8, 0.9, 0.95, 0.99]
    bin_counts = [2, 3, 4]

    raw_rows = []

    # 1. Sweep Alpha
    print("Sweeping Alpha (Learning Rate)...")
    alpha_means, alpha_cis = [], []
    for a in alphas:
        agent = QLearningAgent(alpha=a, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=int(n_train * 0.6), seed=42)
        train_env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=1000)
        for ep in range(n_train):
            train_env.reset(seed=1000 + ep)
            train_env.run(agent, n_ticks=200, learn=True)

        delays = []
        for s in eval_seeds:
            env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            res = env.run(agent, n_ticks=n_eval_ticks, learn=False)
            delays.append(res["avg_delay"])
            raw_rows.append({"experiment": "E4", "param": "alpha", "param_val": a, "seed": s, "metric": "avg_delay", "value": res["avg_delay"]})
        m, ci = compute_ci(delays)
        alpha_means.append(m)
        alpha_cis.append(ci)
        print(f"  alpha={a:4.2f} -> Avg Delay: {m:5.2f} +/- {ci:4.2f}")

    # 2. Sweep Gamma
    print("\nSweeping Gamma (Discount Factor)...")
    gamma_means, gamma_cis = [], []
    for g_val in gammas:
        agent = QLearningAgent(alpha=0.2, gamma=g_val, epsilon_start=1.0, epsilon_end=0.05, decay_episodes=int(n_train * 0.6), seed=42)
        train_env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=1000)
        for ep in range(n_train):
            train_env.reset(seed=1000 + ep)
            train_env.run(agent, n_ticks=200, learn=True)

        delays = []
        for s in eval_seeds:
            env = RoutingEnv(g, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=s)
            res = env.run(agent, n_ticks=n_eval_ticks, learn=False)
            delays.append(res["avg_delay"])
            raw_rows.append({"experiment": "E4", "param": "gamma", "param_val": g_val, "seed": s, "metric": "avg_delay", "value": res["avg_delay"]})
        m, ci = compute_ci(delays)
        gamma_means.append(m)
        gamma_cis.append(ci)
        print(f"  gamma={g_val:4.2f} -> Avg Delay: {m:5.2f} +/- {ci:4.2f}")

    save_raw_csv(raw_rows, "e4_sensitivity_results.csv")

    # Plot Sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.errorbar(alphas, alpha_means, yerr=alpha_cis, marker="o", capsize=5, color="#d73027", linewidth=2)
    ax1.set_xlabel("Learning Rate (alpha)")
    ax1.set_ylabel("Average Delay (ticks)")
    ax1.set_title("E4: Sensitivity to Learning Rate (alpha)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.errorbar(gammas, gamma_means, yerr=gamma_cis, marker="s", capsize=5, color="#4575b4", linewidth=2)
    ax2.set_xlabel("Discount Factor (gamma)")
    ax2.set_ylabel("Average Delay (ticks)")
    ax2.set_title("E4: Sensitivity to Discount Factor (gamma)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "e4_sensitivity_sweep")
    print("E4 complete!\n")


if __name__ == "__main__":
    run_e4(quick=True)
