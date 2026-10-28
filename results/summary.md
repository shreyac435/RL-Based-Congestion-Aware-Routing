# Project Benchmark Summary & Findings

Generated automatically from experimental runs.

## 1. Key Results Overview
- **E1 Main Benchmark**: Under bursty-high load on the 12-node Abilene topology, Q-Learning achieves significant delay reduction and lower packet loss compared to Dijkstra by dynamically balancing traffic onto idle alternate paths.
- **E2 Link Failure**: When the most critical link fails, Dijkstra suffers an immediate packet drop spike during its 15-tick detection convergence window. Q-Learning adapts dynamically, rapidly routing around the failed edge.
- **E3 Ablations**: Both the congestion bin feature representation and off-policy Q-learning show distinct performance advantages over no-congestion state (B2) and static shortest path.
- **E4 Sensitivity**: Performance is robust for alpha in [0.1, 0.3] and gamma in [0.9, 0.95]. 3 congestion bins provide the ideal balance between sample complexity and state distinction.
- **E5 Scalability Boundary**: As network size $N$ increases, tabular state growth scales combinatorially ($N \times |D| \times 3^k$). Tabular Q-learning is highly effective for $N \le 25$, but at large scales ($N \ge 50$), deep function approximation (DQN) is required.

## 2. Generated Deliverables
- Figures stored in `results/figures/` (PNG and PDF formats).
- Raw CSV logs stored in `results/raw/`.
- Interactive demo ready via `streamlit run demo/app.py`.
