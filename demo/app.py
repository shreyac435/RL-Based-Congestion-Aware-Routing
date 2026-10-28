"""Interactive Streamlit Demo for Congestion-Aware Adaptive Routing (Section 13)."""
from __future__ import annotations

import os
import pickle
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.topologies import create_toy_diamond, create_abilene
from sim.traffic import build_overlapping_flows, calibrate_load
from sim.env import RoutingEnv
from agents.dijkstra import DijkstraAgent
from agents.qlearning import QLearningAgent

st.set_page_config(page_title="RL Adaptive Routing Demo", layout="wide")

st.title("RL-Based Congestion-Aware Adaptive Routing (Q-Learning vs Dijkstra)")
st.markdown("Computer Networks Course Project • Interactive Topology & Performance Visualizer")

tab_demo, tab_results = st.tabs(["🔴 Live Simulation & Link Failure", "📊 Benchmark Results & Scalability"])

with tab_demo:
    # Sidebar controls
    st.sidebar.header("Simulation Settings")
    topo_choice = st.sidebar.selectbox("Network Topology", ["Abilene (12 nodes, 15 links)", "Toy Diamond (6 nodes)"])
    load_choice = st.sidebar.selectbox("Offered Load", ["Bursty-High (rho ~ 1.1)", "Medium (rho ~ 0.7)", "Low (rho ~ 0.4)"])
    seed = st.sidebar.number_input("RNG Seed", value=2024, min_value=1, max_value=99999)
    n_ticks_to_run = st.sidebar.slider("Simulation Ticks", min_value=50, max_value=500, value=150, step=25)

    # Setup topology
    if "Abilene" in topo_choice:
        graph = create_abilene()
        pos = nx.spring_layout(graph, seed=42)
        pairs = build_overlapping_flows(graph, num_hotspots=6, seed=42)
    else:
        graph = create_toy_diamond()
        pos = {0: (0, 0.5), 1: (0.3, 1), 2: (0.2, 0), 3: (0.5, 0), 4: (0.8, 0), 5: (1.0, 0.5)}
        pairs = [(0, 5)]

    rho = 1.1 if "High" in load_choice else (0.7 if "Medium" in load_choice else 0.4)
    bursty = ("High" in load_choice)
    flows, _ = calibrate_load(graph, pairs, target_rho=rho, link_capacity=2, bursty=bursty)
    traffic_cfg = {"flows": flows, "bursty": bursty}
    sim_cfg = {"warmup_ticks": 20, "capacity": 2, "buffer_size": 20}

    # Initialize environments
    if "env_d" not in st.session_state or st.sidebar.button("Reset Simulation"):
        st.session_state.env_d = RoutingEnv(graph, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=seed)
        st.session_state.env_q = RoutingEnv(graph, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=seed)
        st.session_state.agent_d = DijkstraAgent(graph, detection_delay=15)

        # Load or train Q-agent
        model_path = "results/models/qlearning_abilene_high.pkl"
        if os.path.exists(model_path) and "Abilene" in topo_choice:
            with open(model_path, "rb") as f:
                st.session_state.agent_q = pickle.load(f)
            st.session_state.agent_q.training = False
            st.session_state.agent_q.epsilon = 0.0
        else:
            ql = QLearningAgent(alpha=0.2, gamma=0.95, epsilon_start=0.5, epsilon_end=0.02, decay_episodes=30, seed=seed)
            train_env = RoutingEnv(graph, traffic_cfg=traffic_cfg, sim_cfg=sim_cfg, seed=100)
            for ep in range(35):
                train_env.reset(seed=100 + ep)
                train_env.run(ql, n_ticks=150, learn=True)
            ql.training = False
            ql.epsilon = 0.0
            st.session_state.agent_q = ql

        st.session_state.failed_edge = None
        st.session_state.tick_history_d = []
        st.session_state.tick_history_q = []

    # Failure Injection UI
    st.sidebar.subheader("Fault Injection (Kill Link)")
    all_edges = list(graph.edges())
    edge_str_list = [f"{u} - {v}" for u, v in all_edges]
    selected_edge_str = st.sidebar.selectbox("Choose Edge to Fail", edge_str_list)
    sel_u, sel_v = selected_edge_str.split(" - ")
    # Cast to int for Toy Diamond
    if "Toy" in topo_choice:
        sel_u, sel_v = int(sel_u), int(sel_v)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("💥 Kill Link"):
        st.session_state.failed_edge = (sel_u, sel_v)
        st.session_state.env_d.schedule_failure(st.session_state.env_d.current_tick, (sel_u, sel_v))
        st.session_state.env_q.schedule_failure(st.session_state.env_q.current_tick, (sel_u, sel_v))
        st.sidebar.warning(f"Link {sel_u}-{sel_v} killed!")

    if c2.button("💚 Restore"):
        if st.session_state.failed_edge:
            u, v = st.session_state.failed_edge
            for d1, d2 in [(u, v), (v, u)]:
                if (d1, d2) in st.session_state.env_d.links:
                    st.session_state.env_d.links[(d1, d2)].up = True
                if (d1, d2) in st.session_state.env_q.links:
                    st.session_state.env_q.links[(d1, d2)].up = True
            st.session_state.failed_edge = None
            st.sidebar.success("Link restored!")

    # Step simulation button
    if st.button(f"▶️ Run / Advance {n_ticks_to_run} Ticks", type="primary"):
        with st.spinner("Advancing discrete-time simulation..."):
            for _ in range(n_ticks_to_run):
                st.session_state.env_d.step(st.session_state.agent_d)
                st.session_state.env_q.step(st.session_state.agent_q)

    # Compute current metrics
    res_d = st.session_state.env_d.metrics_collector.compute(max(1, st.session_state.env_d.current_tick))
    res_q = st.session_state.env_q.metrics_collector.compute(max(1, st.session_state.env_q.current_tick))

    # Helper to plot network graph
    def render_topology_fig(env: RoutingEnv, title: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        edge_colors = []
        for u, v in graph.edges():
            link1 = env.links.get((u, v))
            link2 = env.links.get((v, u))
            if (link1 and not link1.up) or (link2 and not link2.up):
                edge_colors.append("black")  # failed
            else:
                occ = max(link1.queue_occupancy if link1 else 0, link2.queue_occupancy if link2 else 0)
                # Color map from green (0.0) -> orange (0.5) -> red (1.0)
                if occ < 0.33:
                    edge_colors.append("#2ca02c")  # Green
                elif occ < 0.66:
                    edge_colors.append("#ff7f0e")  # Orange
                else:
                    edge_colors.append("#d62728")  # Red

        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#1f77b4", node_size=350)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=3.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        return fig

    # Main side-by-side comparison view
    col_d, col_q = st.columns(2)

    with col_d:
        st.subheader("🔵 Dijkstra (Static Shortest Path)")
        fig_d = render_topology_fig(st.session_state.env_d, f"Dijkstra State (Tick {st.session_state.env_d.current_tick})")
        st.pyplot(fig_d)
        plt.close(fig_d)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Delay", f"{res_d['avg_delay']:.1f} ticks")
        m2.metric("Packet Loss", f"{res_d['loss_pct']:.1f}%")
        m3.metric("Throughput", f"{res_d['throughput']:.2f} pkt/t")
        m4.metric("Fairness", f"{res_d['load_balance']:.2f}")

    with col_q:
        st.subheader("🔴 Q-Learning (Adaptive Congestion-Aware)")
        fig_q = render_topology_fig(st.session_state.env_q, f"Q-Learning State (Tick {st.session_state.env_q.current_tick})")
        st.pyplot(fig_q)
        plt.close(fig_q)

        q1, q2, q3, q4 = st.columns(4)
        delay_diff = res_q['avg_delay'] - res_d['avg_delay']
        loss_diff = res_q['loss_pct'] - res_d['loss_pct']
        thru_diff = res_q['throughput'] - res_d['throughput']

        q1.metric("Avg Delay", f"{res_q['avg_delay']:.1f} ticks", delta=f"{delay_diff:+.1f}", delta_color="inverse")
        q2.metric("Packet Loss", f"{res_q['loss_pct']:.1f}%", delta=f"{loss_diff:+.1f}%", delta_color="inverse")
        q3.metric("Throughput", f"{res_q['throughput']:.2f} pkt/t", delta=f"{thru_diff:+.2f}")
        q4.metric("Fairness", f"{res_q['load_balance']:.2f}")

    st.info("💡 **Color Legend**: 🟢 Green = Low Buffer Occupancy (<33%) | 🟠 Orange = Medium (33-66%) | 🔴 Red = Congested (>66%) | ⚫ Black = Link Down")


with tab_results:
    st.header("Benchmark Figures & Empirical Verification")
    st.markdown("All figures below are generated directly from reproducible code runs with 95% confidence intervals.")

    res_fig_dir = "results/figures"
    fig_files = [
        ("E1: Main Performance by Traffic Load", "e1_delay_by_load.png", "e1_loss_by_load.png"),
        ("E1: System Throughput Comparison", "e1_throughput_by_load.png", None),
        ("E2: Link Failure Dynamics Over Time", "e2_failure_dynamics.png", None),
        ("E3: Ablations (SARSA & Q-Routing without Congestion)", "e3_ablation_comparison.png", None),
        ("E4: Hyperparameter Sensitivity", "e4_sensitivity_sweep.png", None),
        ("E5: Scalability Sweep & Tabular Boundary", "e5_scalability_boundary.png", None),
    ]

    for title, f1, f2 in fig_files:
        st.subheader(title)
        p1 = os.path.join(res_fig_dir, f1)
        if f2:
            p2 = os.path.join(res_fig_dir, f2)
            c1, c2 = st.columns(2)
            if os.path.exists(p1):
                c1.image(p1, caption=f1)
            if os.path.exists(p2):
                c2.image(p2, caption=f2)
        else:
            if os.path.exists(p1):
                st.image(p1, caption=f1, use_container_width=True)
            else:
                st.info(f"Figure `{f1}` will appear once `python run_all.py` completes.")
