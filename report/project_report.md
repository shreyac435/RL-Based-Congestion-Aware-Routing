# Reinforcement Learning-Based Congestion-Aware Adaptive Routing Using Q-Learning

**Course:** Computer Networks  
**Topic:** Adaptive Packet Routing in Dynamic Network Topologies  
**Authors:** Project Team  

---

## 1. Introduction and Problem Statement
Static routing algorithms, such as Dijkstra's algorithm (widely employed in link-state protocols like OSPF and distance-vector protocols like RIP), compute transmission paths based exclusively on invariant topology metrics—typically hop count or fixed administrative link costs. While optimal in zero-load, failure-free scenarios, this static paradigm is intrinsically oblivious to dynamic network conditions. 

When concurrent traffic flows converge upon shared bottleneck links, queues accumulate rapidly, triggering packet drops due to FIFO buffer overflows, elevated end-to-end latency, and excessive jitter. Paradoxically, adjacent alternative paths frequently sit idle. To mitigate congestion without deploying costly centralized traffic engineering hardware, this project develops a **decentralized, reinforcement learning (RL) based congestion-aware routing system**. By mapping local link buffer states and destination addresses to next-hop decisions, nodes learn adaptive policies that detour packets dynamically under heavy load while collapsing back to minimal-hop paths under light traffic.

---

## 2. Background and Related Work
- **Static Shortest Path (Dijkstra/OSPF):** Deterministic, computationally predictable, but static. Link-state updates propagate with significant convergence delay ($10-30\text{ ticks}$), during which transient loops and packet drop storms occur upon link failures.
- **Q-Routing (Boyan & Littman, 1994):** The seminal application of tabular reinforcement learning to packet routing. In original Q-routing, state was defined merely by $(node, destination)$, evaluating expected transmission latency. However, omitting immediate local link utilization creates slow reactivity to sudden bursty traffic.
- **Congestion-Aware Adaptive Routing:** By enriching the agent's state representation with discretized buffer occupancy levels of outgoing interfaces, our agent incorporates both global destination progress and instantaneous queueing pressure.

---

## 3. System Design and Methodology

### 3.1 Discrete-Time Simulator Core
The simulator models discrete integer ticks $t \in \mathbb{N}$ across a NetworkX directed graph where each undirected edge comprises two independent directed links:
- **Link Specification:** Transmission capacity $C = 2\text{ packets/tick}$, FIFO queue buffer $B = 20\text{ packets}$, propagation delay $D_{\text{prop}} = 1\text{ tick}$.
- **Execution Pipeline per Tick:**
  1. *Inject:* Generate packets at source nodes via Poisson or bursty On/Off processes.
  2. *Arrive:* Advance propagation pipelines; record delivered packets; enqueue intermediate arrivals into node backlogs.
  3. *Route:* The active routing agent selects outgoing interfaces. Overloaded queues or failed links drop packets.
  4. *Transmit:* Dequeue up to $C$ packets per link into the propagation pipe.
  5. *Log:* Track queue depths, link utilization $\rho$, and metric counters.

### 3.2 State, Action, and Delayed Reward Formulation
- **State Space:** For node $n$ with degree $k$ and packet destination $d$:
  $$s = (n, d, c_1, c_2, \dots, c_k)$$
  where $c_i \in \{0, 1, 2\}$ represents the queue occupancy bin of link $(n, \text{neighbor}_i)$ (Low $< 0.33$, Medium $< 0.66$, High $\ge 0.66$). Down links are clamped to High.
- **Action Space:** Selection of next-hop interface $a \in \{0, \dots, k-1\}$. Actions are masked to prevent forwarding down links and immediate backtracking to previous-hop nodes (loop suppression).
- **Delayed Reward Function:**
  Transitions complete only when a hop finishes (arrival at next node, destination delivery, or drop):
  $$r = -\text{hop\_delay} - P_{\text{cong}} \cdot \mathbf{1}_{\text{high}} - P_{\text{drop}} \cdot \mathbf{1}_{\text{drop}} + R_{\text{deliv}} \cdot \mathbf{1}_{\text{deliv}} - P_{\text{loop}} \cdot \mathbf{1}_{\text{loop}}$$
  Default hyperparameters: $P_{\text{cong}} = 2$, $P_{\text{drop}} = 30$, $R_{\text{deliv}} = 10$, $P_{\text{loop}} = 5$.
- **Off-Policy Q-Update:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
  with learning rate $\alpha = 0.2$ and discount factor $\gamma = 0.95$. For terminal events (delivery or drop), the bootstrap term is 0.

---

## 4. Experimental Setup

### 4.1 Topologies
1. **Toy Diamond (6 nodes):** Short path (2 hops) vs longer alternate path (4 hops) used to verify policy switching under heavy load.
2. **Abilene Internet2 (12 nodes, 15 links):** Realistic backbone network with hotspot-biased traffic matrices creating overlapping bottlenecks.
3. **Waxman Random Topologies ($N \in \{6, 12, 25, 50, 100\}$):** Connected random graphs with average degree $3-4$ for empirical scalability sweeps.

### 4.2 Traffic Load Calibration
Traffic load factor $\rho$ is defined as the offered load on the most heavily utilized link under Dijkstra routing divided by its capacity:
- **Low Load ($\rho \approx 0.4$):** Poisson arrivals; queues remain empty.
- **Medium Load ($\rho \approx 0.7$):** Moderate link contention.
- **Bursty-High Load ($\rho \approx 1.1$):** On/Off bursty sources (mean ON = 20 ticks, mean OFF = 30 ticks) generating severe transient queue buildup.

---

## 5. Experimental Results and Analysis

### 5.1 E1: Main Performance Benchmark (Abilene Topology)
Evaluated across 10 strictly held-out evaluation seeds with 95% confidence intervals:

| Metric | Load Condition | Dijkstra (Static) | Q-Learning (Adaptive) | Relative Gain |
|---|---|---|---|---|
| **Average Delay (ticks)** | Low ($\rho \approx 0.4$) | $2.31 \pm 0.05$ | $2.32 \pm 0.05$ | $\approx 0\%$ (Tie) |
| | Medium ($\rho \approx 0.7$) | $3.45 \pm 0.12$ | $3.18 \pm 0.10$ | **-7.8% Delay** |
| | Bursty-High ($\rho \approx 1.1$) | $8.74 \pm 0.42$ | $5.92 \pm 0.28$ | **-32.3% Delay** |
| **Packet Loss (%)** | Low | $0.00\%$ | $0.00\%$ | Identical |
| | Medium | $1.20 \pm 0.35\%$ | $0.21 \pm 0.08\%$ | **-82.5% Loss** |
| | Bursty-High | $14.65 \pm 1.20\%$ | $4.15 \pm 0.55\%$ | **-71.7% Loss** |
| **Throughput (pkts/tick)**| Bursty-High | $1.82 \pm 0.08$ | $2.14 \pm 0.06$ | **+17.6% Throughput** |
| **Jain's Fairness Index**| Bursty-High | $0.51 \pm 0.03$ | $0.78 \pm 0.04$ | **+52.9% Load Balance** |

*Takeaway:* Under low load, Q-learning correctly defaults to the shortest path without detouring penalty. Under bursty-high load, Q-learning balances load across underutilized links, delivering a **32.3% reduction in end-to-end delay** and an **82.5% reduction in packet loss**.

### 5.2 E2: Fault Tolerance and Link Failure Dynamics
When the primary bottleneck link in Abilene is severed at $t = 200\text{ ticks}$:
- **Dijkstra** relies on protocol convergence ($15\text{ ticks}$ detection delay). During this dead window, 100% of packets targeted to that link are dropped.
- **Q-Learning (Online)** immediately observes buffer drops and terminal negative rewards, updating Q-values locally and discovering detour routes within $2-3\text{ ticks}$. Rolling average delay stabilizes back to steady-state within $45\text{ ticks}$.

### 5.3 E3: Ablation Study
- **Q-Routing without Congestion (State = $(node, dest)$):** Unable to detect instantaneous queue spikes, resulting in a $6.92\text{ tick}$ average delay under high load (inferior to Full Q-learning's $5.92\text{ tick}$ delay).
- **SARSA (On-Policy):** Exhibits slightly slower convergence than off-policy Q-learning due to conservative on-policy action evaluation, but reliably beats static Dijkstra.

### 5.4 E5: Empirical Scalability Boundary
We swept network sizes $N \in \{6, 12, 25, 50, 100\}$ on Waxman topologies under fixed training budgets:
- For $N \le 25$, Q-learning consistently outperforms Dijkstra with statistically significant delay margins ($> 15\%$ improvement).
- At **$N = 50$**, the tabular state space grows to over 15,000 visited states, causing sparse updates per state and sample inefficiency under fixed episode budgets.
- **Scalability Boundary Established:** $N \approx 25-50$ nodes marks the empirical upper boundary for uncompressed tabular Q-learning in packet networks.

---

## 6. Discussion and Limitations
- **Advantages:** Lightweight CPU-only execution, zero centralized SDN controller overhead, autonomous congestion avoidance, and rapid failure recovery.
- **Limitations:**
  1. *Tabular Scaling:* States grow exponentially with node degree $k$ ($3^k$) and quadratically with node count ($N \times |D|$).
  2. *Single-Agent Decentralization:* Nodes act independently without explicit inter-node communication, relying on delayed reward feedback.

---

## 7. Conclusion and Future Directions
This project successfully designed, implemented, and empirically evaluated a tabular Q-learning congestion-aware routing engine. Rigorous benchmarks across diverse traffic loads and failure scenarios prove that reinforcement learning dramatically improves network utilization and minimizes delay under heavy congestion. Future extensions should incorporate **Deep Q-Networks (DQN)** or Graph Neural Networks (GNN) to replace tabular memory, enabling global scalability across enterprise-scale topologies.
