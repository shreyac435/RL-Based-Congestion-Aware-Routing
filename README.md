# RL-Based Congestion-Aware Adaptive Routing Using Q-Learning

Computer Networks Final Course Project: High-performance, reproducible comparison between Reinforcement Learning (Tabular Q-Learning) and Static Shortest-Path Routing (Dijkstra) under diverse traffic loads, link failures, and scale topologies.

---

## 🚀 Quick Start

### 1. Requirements & Setup
Ensure Python 3.11+ is installed. Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run All Tests and Experiments (One-Command Reproducibility)
Execute the complete test suite and regenerate all experiment figures (`results/figures/*.png` & `*.pdf`) and raw CSV logs:
```bash
python run_all.py --quick
```
*(For full extensive evaluation across all seeds and topology sizes, run `python run_all.py --full`)*

### 3. Launch Interactive Streamlit Demo
Experience real-time traffic visualization, side-by-side comparison, and fault injection:
```bash
streamlit run demo/app.py
```

---

## 📁 Repository Structure
```
├── PROJECT_SPEC.md        # Single authoritative specification document
├── DECISIONS.md           # Log of design decisions and mathematical parameters
├── requirements.txt       # Locked dependencies
├── run_all.py             # Master script: runs tests and all E1-E5 experiments
├── sim/                   # Discrete-time packet-level network simulator
│   ├── network.py         # Packet, Link, queues, transmission, and propagation pipes
│   ├── topologies.py      # Toy Diamond (6 nodes), Abilene (12 nodes), Waxman random generator
│   ├── traffic.py         # Poisson and bursty On/Off traffic generators + load calibration
│   ├── env.py             # Gym-style discrete-time environment interface
│   └── metrics.py         # Statistical delay, loss %, throughput, and Jain's fairness index
├── agents/                # Routing policies
│   ├── base.py            # Agent interface
│   ├── dijkstra.py        # Static shortest path with convergence detection delay
│   ├── qlearning.py       # Tabular Q-Learning with delayed per-packet transitions
│   ├── sarsa.py           # On-policy SARSA agent
│   └── qrouting_nocong.py # B2 ablation agent (without congestion features)
├── experiments/           # Benchmark scripts
│   ├── runner.py          # Confidence intervals (95% CI) and CSV/Figure export
│   ├── e1_main.py         # E1: Low, Medium, and Bursty-High load on Abilene
│   ├── e2_failure.py      # E2: Dynamic link failure and recovery response
│   ├── e3_ablation.py     # E3: SARSA vs Q-Learning vs No-Congestion
│   ├── e4_sensitivity.py  # E4: Alpha, Gamma, and binning sensitivity sweeps
│   └── e5_scalability.py  # E5: Empirical scalability boundary (Waxman N in 6..100)
├── demo/
│   └── app.py             # Streamlit visualizer with live kill-link button
├── tests/
│   └── test_all.py        # Pytest suite covering all 9 required verification tests
├── results/
│   ├── raw/               # Tidy CSV logs
│   ├── figures/           # Publication-ready PNG and PDF figures
│   └── summary.md         # Generated benchmark summary
└── report/
    └── project_report.md  # Complete final course project report
```

---

## 🔬 Experimental Findings Summary

1. **E1 (Main Load Benchmark)**:
   - At **Low Load** ($\rho \approx 0.4$), Q-Learning and Dijkstra perform identically because shortest paths have zero queueing delay.
   - At **Bursty-High Load** ($\rho \approx 1.1$), Q-Learning achieves significant reductions in both average and 95th-percentile packet delay and lowers buffer-overflow packet loss by dynamically routing around bottleneck links.
2. **E2 (Fault Tolerance)**:
   - Following link failure on the highest-utilized edge, Dijkstra drops 100% of packets routed across that edge during its detection window (15 ticks).
   - Q-Learning perceives queue saturation and drops immediately, updating state values and detouring traffic within 2–3 ticks.
3. **E3 (Ablation Analysis)**:
   - State representation with 3 link congestion bins is critical; omitting congestion features (state = `(node, dest)`) degrades performance back towards static routing.
4. **E5 (Tabular Scalability Boundary)**:
   - The empirical scalability limit of tabular Q-learning is observed around **$N \approx 25$ nodes**. Beyond this, state space explosion ($N \times |D| \times 3^k$) requires exponential training samples to achieve convergence, motivating deep reinforcement learning (DQN) for large-scale SDNs.
