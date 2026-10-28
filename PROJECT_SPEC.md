# RL-Based Congestion-Aware Routing Using Q-Learning: Master Project Specification

*Computer Networks course project. Build specification and agent instructions, written to be used as the single source of truth by an AI coding agent (Antigravity) and the human team.*

## 0. How to use this document

**For the human team:** save this file as `PROJECT_SPEC.md` in the repo root. Start the agent with the kickoff prompt in Section 17. Review at the gates in Section 14. Do not let the agent skip ahead of a gate.

**For the agent (read carefully):**

1. This document overrides your defaults. If something here conflicts with a habit of yours, follow this document.
2. Work milestone by milestone (Section 14). Before each milestone, write a short implementation plan, then wait for approval. After each milestone, run the tests and show the acceptance evidence.
3. Never invent results. Every number, plot and claim must come from code in this repo that can be re-run. No hard-coded or fabricated data.
4. Do not add features outside the scope tiers in Section 2. Do not add neural networks, GPUs, PyTorch, TensorFlow, Mininet or Ryu.
5. When a design detail is genuinely ambiguous, choose the simplest option that satisfies this spec, record the choice in `DECISIONS.md`, and continue. Ask only if the choice would change the experimental conclusions.
6. Reproducibility is mandatory: every random process takes an explicit seed, and `python run_all.py` must regenerate every result and figure from scratch.

## 1. Project definition and traceability

**Title:** Reinforcement Learning-Based Congestion-Aware Adaptive Routing Using Q-Learning.

**Problem:** Static protocols (RIP, OSPF) pick paths using hop count or fixed link cost. When many flows share the same shortest path, its queues build, delay, jitter and loss rise, while longer alternative paths sit idle. The goal is routing that adapts to congestion as it develops, without a centralised traffic model or expensive hardware.

**Headline contribution:** an empirical scalability boundary for GPU-free tabular Q-learning routing, meaning the topology size at which it stops being practical, plus a direct comparison against static shortest-path routing under several loads and link failures. Do not write that no prior study has done this. Phrase it as "we empirically characterise" and leave the novelty claim to a literature check by the humans.

### Traceability matrix (every required item maps to a deliverable)

| Requirement from the project brief | Where it is delivered | Evidence |
| --- | --- | --- |
| Model topology as a simulated environment | `sim/` package | Unit tests, topology plots |
| Q-learning agent learning next-hop from link utilisation and queueing delay | `agents/qlearning.py` | Learning curves, Q-table stats |
| Compare vs Dijkstra on delay, loss, throughput, load balance | `experiments/e1_main.py` | Table and bar charts with 95% CI |
| Low, medium and bursty high-load traffic | `sim/traffic.py` plus configs | E1 results |
| Link-failure scenarios | `experiments/e2_failure.py` | Time-series plot, recovery time |
| Lightweight: tabular, CPU-only, reproducible, easy to demo | Whole repo | `run_all.py`, `requirements.txt`, no GPU libs |
| Q-values converge in reasonable episodes | Convergence logging | Convergence plot |
| Comparison graphs in a working demo | `demo/app.py` | Live demo, runs offline |
| Scope note: tabular suits small and medium topologies | `experiments/e5_scalability.py` | Scalability figure and boundary table |
| Ablations (secondary) | `experiments/e3_ablation.py`, `e4_sensitivity.py` | Ablation tables |

## 2. Locked decisions and scope tiers

**Locked decisions**

- Language: Python 3.11+. Libraries allowed: `networkx`, `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `pytest`, `streamlit`, `tqdm`. Nothing else without a `DECISIONS.md` entry.
- Simulator: custom discrete-time simulator in pure Python/NumPy on top of NetworkX graphs.
- Algorithm: tabular Q-learning (primary), SARSA (secondary). Per-hop next-hop selection in the style of Q-routing (Boyan and Littman, 1994).
- Baselines: B1 Dijkstra on hop count (the required baseline). B2 Q-routing without congestion state (ablation, shows congestion features matter). B3 ECMP (stretch only).
- CPU only, laptop friendly. A full `run_all.py` in quick mode must finish in under 30 minutes, and under about 3 hours in full mode.

**Tier 1 (must ship):** simulator, Q-learning agent, Dijkstra baseline, low, medium and bursty loads, link-failure test, comparison graphs, demo, report.

**Tier 2 (should ship):** SARSA comparison, Q-routing ablation, hyperparameter sensitivity, scalability sweep at 5 topology sizes.

**Tier 3 (stretch, only after Tier 1 and 2 pass their gates):** ECMP baseline, congestion-level prediction output, paper-style draft.

**Freeze rule:** no new features once Milestone M7 starts.

## 3. Repository layout

```
project/
  PROJECT_SPEC.md        this file
  DECISIONS.md           every ambiguous choice the agent made
  README.md              setup, one-command reproduce, demo instructions
  requirements.txt       pinned versions
  run_all.py             regenerates everything (flags: --quick, --full)
  sim/
    network.py           links, queues, packets, tick loop
    traffic.py           Poisson and on/off burst generators
    topologies.py        toy, abilene, waxman generator
    env.py               Gym-style wrapper (Section 5)
    metrics.py           all metric definitions (Section 10)
  agents/
    base.py              Agent interface
    qlearning.py
    sarsa.py
    qrouting_nocong.py   ablation B2
    dijkstra.py          B1
    ecmp.py              B3 (stretch)
  experiments/
    configs/*.yaml       one file per experiment
    e1_main.py  e2_failure.py  e3_ablation.py
    e4_sensitivity.py  e5_scalability.py
    runner.py            seeding, parallel runs, CSV logging
  results/raw/           CSV, one row per (experiment, method, seed, metric)
  results/figures/       PNG and PDF figures
  demo/app.py            Streamlit demo
  tests/                 pytest suite
  report/                outline, figures, final report
```

## 4. Simulator specification

### 4.1 Model

Time advances in integer **ticks**. The network is a NetworkX graph. Each undirected edge becomes two independent **directed links**. Every directed link has:

- `capacity`: packets it can transmit per tick (default 2)
- `buffer`: output queue size in packets (default 20)
- `prop_delay`: ticks of propagation (default 1)
- a FIFO queue (`collections.deque`) and an `up` flag

A **packet** has: `id`, `src`, `dst`, `birth_tick`, `ttl` (default 2 times the node count, capped at 64), `hops`, and a log of the hop it is currently on.

### 4.2 Tick order (must be exactly this, and documented in code)

1. **Inject:** the traffic generator creates new packets at their source nodes.
2. **Arrive:** packets whose propagation finished this tick arrive at the next node. If the node is the destination, mark delivered and record delay. Otherwise they need a routing decision.
3. **Route:** for every packet needing a decision, the active agent picks a next-hop link. If the chosen link is down or its queue is full, drop the packet (reason: `queue_full` or `link_down`). If TTL is exhausted, drop (reason: `ttl`). Otherwise append to the link queue and record `enqueue_tick`.
4. **Transmit:** each up link dequeues up to `capacity` packets from its head. Each goes into a propagation pipe and arrives after `prop_delay` ticks.
5. **Log:** record per-link queue length, utilisation (packets sent divided by capacity), and counters.

**Hop delay** for a packet on a link = (arrival tick at next node) minus (enqueue tick). This is queueing plus transmission plus propagation delay and is the quantity the reward uses.

### 4.3 Failure handling

A failure event at tick `t_f` sets `up=False` on both directions of an edge. Packets queued on it and packets in its propagation pipe are dropped (reason: `link_down`). Optionally a recovery event restores the link.

### 4.4 Performance requirement

Per-tick cost must scale with packets in flight, not with the square of nodes. Use per-link deques and avoid rebuilding graph structures each tick. Cache the neighbour order per node. Profile at N=100 and report ticks per second in `README.md`.

## 5. Environment API (freeze this at M1)

Gym-style, but implemented by hand.

```python
class RoutingEnv:
    def __init__(self, graph, traffic_cfg, sim_cfg, seed): ...
    def reset(self, seed=None) -> None
    def step(self, agent) -> dict      # advances one tick, returns tick stats
    def run(self, agent, n_ticks, learn: bool) -> dict   # full episode, returns episode metrics
    def get_state(self, node, dest) -> tuple   # (node, dest, bin_1..bin_k)
    def neighbors(self, node) -> list          # fixed, sorted order
    def schedule_failure(self, tick, edge, recover_tick=None) -> None

class Agent:  # agents/base.py
    def choose_next_hop(self, env, packet, node) -> int       # returns neighbour node id
    def observe(self, transition) -> None                     # learning hook, no-op for baselines
    def begin_episode(self, training: bool) -> None
    def end_episode(self) -> dict                             # e.g. epsilon, q-delta, table size
```

A `transition` is `(state, action_idx, reward, next_state_or_None, done, info)`.

## 6. Reinforcement learning specification

### 6.1 Learning protocol (delayed, per-packet)

Each packet carries its own pending transition. When a packet is routed at node `n`, the agent stores `(s, a)` on the packet. When the hop completes (packet arrives at the next node, is delivered, or is dropped), the reward is computed and the Q-update is applied. This is the standard Q-routing structure: the update waits for the hop outcome and bootstraps from the next node's Q-values.

### 6.2 State

`s = (node, dest, c_1, ..., c_k)` where `k` is the node's degree and `c_i` is the congestion bin of outgoing link `i` in sorted-neighbour order.

Congestion bin from queue occupancy `q = queue_len / buffer`: **low** if q < 0.33, **medium** if q < 0.66, **high** otherwise. Three bins, configurable but default 3. Down links are treated as high and masked from actions.

Store Q-values lazily: `dict[state] -> np.ndarray(k)` initialised to zeros (optimistic initial values are a tunable). The Q-table size metric is the number of stored states.

### 6.3 Action

Choose a neighbour index in `0..k-1`. Mask: (a) down links, (b) the previous hop, unless it is the only option. If the destination is a direct neighbour, still let the agent choose (do not hard-code the answer).

### 6.4 Reward

For a completed hop:

```
r = -hop_delay
    - P_cong   if next link (at decision time) was in the high bin
    - P_drop   if the packet was dropped on this hop (terminal)
    + R_deliv  if the packet reached its destination on this hop (terminal)
    - P_loop   if the packet revisits a node it already visited
```

Defaults (all in YAML, tunable): `P_cong=2`, `P_drop=30`, `R_deliv=10`, `P_loop=5`. Document the intuition: delay dominates, drops are much worse than any queueing delay, delivery gives a positive terminal signal.

### 6.5 Update rules

- **Q-learning (off-policy):** `Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`. For terminal transitions the bootstrap term is 0.
- **SARSA (on-policy):** same, but use `Q(s',a')` for the action `a'` actually chosen at the next node. Implement by deferring the previous packet's update until the next-hop decision is made.

Defaults: `alpha=0.2`, `gamma=0.95`. Exploration: epsilon-greedy with exponential decay from 1.0 to 0.05 over the first 60% of training episodes, then fixed at 0.05. Tuning ranges: alpha in {0.05, 0.1, 0.2, 0.3, 0.5}, gamma in {0.8, 0.9, 0.95, 0.99}, final epsilon in {0.01, 0.05, 0.1}.

### 6.6 Training and evaluation protocol (strict)

- **Training:** `n_train_episodes` (default 500 for the 12-node case) of `episode_ticks=500` ticks each, using training seeds 1000 to 1999, with the traffic seed varying per episode. Learning is on, epsilon decays.
- **Evaluation:** `epsilon=0`, Q-table frozen (learning off), 1000 ticks, first 200 ticks discarded as warm-up, **10 held-out seeds** (2000 to 2009) that were never used in training. Dijkstra is evaluated on exactly the same seeds and traffic traces. Different methods must see identical packet arrivals.
- Always train a separate agent per scenario type and per topology unless an experiment states otherwise.

### 6.7 Convergence criterion

Track per episode: mean reward per packet, mean absolute Q-update (the Q-delta), table size. **Converged** at the first episode where the 50-episode rolling mean reward changes by less than 2% over the next 100 episodes and the rolling mean Q-delta is below 0.05. Log `episodes_to_converge` or `None` if the budget runs out.

## 7. Baselines

- **B1 Dijkstra (required):** shortest path by hop count (all weights 1), precomputed next-hop table, recomputed on a topology change after a configurable detection delay (default 20 ticks, to mimic routing-protocol convergence). It is deterministic and ignores load.
- **B2 Q-routing without congestion:** the same agent with state `(node, dest)` only. Shows the value of congestion features.
- **B3 ECMP (stretch):** split equal-cost shortest paths by flow hash.

## 8. Topologies and traffic

### 8.1 Topologies

1. **Toy diamond (6 nodes)** for debugging: a short path and a longer parallel path, so the correct policy is obvious (use the long path when the short one saturates).
2. **Abilene (12 nodes, 15 links)**, hard-coded. Edge list (verify once against a public source and note it in `DECISIONS.md`): ATLAM5-ATLAng, ATLAng-HSTNng, ATLAng-IPLSng, ATLAng-WASHng, CHINng-IPLSng, CHINng-NYCMng, DNVRng-KSCYng, DNVRng-SNVAng, DNVRng-STTLng, HSTNng-KSCYng, HSTNng-LOSAng, IPLSng-KSCYng, LOSAng-SNVAng, NYCMng-WASHng, SNVAng-STTLng.
3. **Scalability family:** connected Waxman graphs with N in {6, 12, 25, 50, 100}, generated from a seed, targeting average degree about 3 to 4, uniform link parameters.

### 8.2 Traffic

- **Flow definition:** a flow is a (src, dst) pair with a packet rate. Choose flow sets so that several Dijkstra shortest paths **overlap on shared links**. Otherwise there is no congestion to avoid and the comparison is meaningless. For Abilene, use a hotspot-biased traffic matrix (a few heavy pairs crossing the middle of the network plus light background pairs). Save the flow matrix in the config.
- **Low and medium:** Poisson arrivals per flow.
- **High (bursty):** on/off sources, with exponentially distributed ON and OFF durations (default mean 20 ticks ON, 30 ticks OFF) and Poisson arrivals during ON.
- **Load calibration (important):** define load factor `rho` as the offered load on the most-loaded link under Dijkstra routing divided by its capacity. Calibrate the rates automatically with a helper function so that `rho` is about **0.4 (low), 0.7 (medium), 1.1 (high, bursty, average 0.8 with peak above 1.2)**. Print the achieved `rho` per scenario.
- For scalability, scale the number of flows proportionally to N and recalibrate so that `rho` for the medium and high settings stays comparable across sizes.

## 9. Experiments

Every experiment uses at least 10 evaluation seeds, reports mean and 95% confidence interval (t-distribution), and writes raw CSV before any plotting.

| ID | Question | Setup | Output |
| --- | --- | --- | --- |
| E0 | Does the simulator behave correctly and does the agent learn anything? | Toy diamond, Dijkstra vs Q-learning | Sanity plot: Q-learning shifts to the long path under load |
| E1 | Does Q-learning beat Dijkstra? | Abilene, low, medium, bursty-high | Table and bar charts for all metrics |
| E2 | How does each method handle a link failure? | Abilene, medium and high load, fail the link with the highest Dijkstra load at tick 500 of 1000 | Delay and loss over time, recovery time, loss spike |
| E3 | Which parts matter? | Q-learning vs SARSA vs Q-routing without congestion | Ablation table |
| E4 | How sensitive is it? | Sweep alpha, gamma, final epsilon, reward penalties, bin count (2, 3, 4) one at a time | Sensitivity plots |
| E5 | Where does tabular Q-learning break down? | Waxman N in {6, 12, 25, 50, 100}, fixed training budget (2000 episodes) | Scalability figure and boundary table |

**E2 details:** evaluate two Q-modes: *frozen* (no learning) and *online* (learning on, epsilon 0.05). Recovery time is the number of ticks after failure until the 50-tick rolling average delay returns to within 10% of the pre-failure average (or `None`). Loss spike is the peak 50-tick loss rate after failure.

**E5 boundary definition (fixed in advance, do not move afterwards):** the boundary is the smallest N where either (a) the agent fails the convergence criterion within the episode budget, or (b) its mean delay under high load is not statistically better than Dijkstra. Report both boundaries separately. Plot N against: episodes to converge, Q-table size (states visited), training wall-clock time, and the percentage delay improvement over Dijkstra with CI. At large N use fewer seeds only if time forces it (minimum 5) and state it explicitly.

**Expected-result honesty:** the hypothesis is that Q-learning wins under high and bursty load and after failures, and roughly ties at low load. Report whatever actually happens. A tie or loss at medium load is an acceptable and publishable result.

## 10. Metric definitions (implement once in `sim/metrics.py`)

- **Average delay:** mean end-to-end delay of delivered packets (ticks), after warm-up.
- **P95 delay:** 95th percentile of the same.
- **Packet loss %:** dropped divided by injected, after warm-up. Also break down by drop reason.
- **Throughput:** delivered packets per tick.
- **Load balance:** Jain's fairness index over mean link utilisation, `(sum x)^2 / (n * sum x^2)`, over all up directed links. Also record maximum link utilisation.
- **Convergence:** episodes to converge (Section 6.7).
- **Cost:** training wall-clock seconds, evaluation ticks per second, Q-table states and approximate memory.
- **Average hop count** (to show the price of detouring).

## 11. Output formats

- One tidy CSV per experiment in `results/raw/`: columns `experiment, topology, load, method, seed, metric, value` plus any experiment-specific parameters.
- Figures: consistent style, labelled axes with units, error bars as 95% CI, legend, saved as PNG (300 dpi) and PDF. Filenames match experiment IDs, for example `e1_delay_by_load.png`.
- A machine-written `results/summary.md` listing the key numbers for the report.

## 12. Testing and validation (must pass before experiments)

Write `pytest` tests for:

1. **Packet conservation:** injected = delivered + dropped + in flight, every tick.
2. **Hand-computed case:** on a 3-node line with known capacity and one packet, the end-to-end delay equals the value computed by hand.
3. **Queue and capacity:** a link never transmits more than `capacity` per tick, and the queue never exceeds `buffer`.
4. **Determinism:** the same seed produces identical metrics; different seeds differ.
5. **Failure:** packets on a failed link are dropped, and Dijkstra reroutes after the detection delay.
6. **Dijkstra correctness:** its paths match `networkx.shortest_path`.
7. **Q-update:** a single update matches a hand-calculated value for both Q-learning and SARSA.
8. **Learning sanity:** on the toy diamond, after training the agent prefers the long path under heavy load and the short path under light load.
9. **Evaluation hygiene:** evaluation seeds are disjoint from training seeds, and learning is off during evaluation (Q-table checksum unchanged).

## 13. Demo specification (`demo/app.py`, Streamlit)

- Runs fully offline: `streamlit run demo/app.py`.
- Sidebar: choose topology (toy or Abilene), load level (low, medium, bursty), seed, and playback speed.
- Main view: two topology plots side by side, Dijkstra on the left and Q-learning on the right, using the same traffic seed. Link colours shift from green to red with queue occupancy. Below each, live counters for average delay, loss % and throughput, and a rolling line chart.
- A **Kill link** button (or dropdown plus button) triggers a failure in both panels at the current tick, and a **Restore** button heals it.
- A tab with the pre-computed result figures from `results/figures/` (E1, E2, E5).
- Use a pre-trained Q-table file (`results/models/*.npz` or `.pkl`) so the demo starts instantly. Provide a fallback replay mode from saved logs if live simulation is too slow.

## 14. Milestones, acceptance criteria and review gates

The agent must stop at every gate and present evidence.

| # | Milestone | Acceptance criteria |
| --- | --- | --- |
| M1 | Scaffold and simulator core | Repo layout, `requirements.txt`, link/queue/packet/tick loop, toy topology, API frozen. Tests 1 to 4 pass. |
| M2 | Traffic, topologies, failures | Poisson and on/off traffic, Abilene, Waxman generator, load calibration helper, failure events. Tests 5 and 6 pass. |
| M3 | Dijkstra and metrics | B1 agent, `metrics.py`, runner with seeds and CSV logging. E0 baseline run on the toy graph. |
| M4 | Q-learning agent | Delayed per-packet updates, masking, TTL and loop handling, epsilon schedule, convergence logging. Tests 7 to 9 pass. E0 shows the learned detour. |
| M5 | Main experiments | E1 and E2 complete with CIs and figures. **Tier 1 results frozen here.** |
| M6 | Tier 2 experiments | SARSA, B2 ablation, sensitivity (E3, E4), scalability sweep (E5) with the boundary table. |
| M7 | Demo | Streamlit demo meeting Section 13. No new features after this starts. |
| M8 | Reproducibility and report | `run_all.py --quick` and `--full` work from a clean clone, README complete, report draft and slides outline generated from real results. |

**Human review gates:** after M1 (API and simulator semantics), M4 (reward and state design), M5 (are the results believable?), and M8 (final).

## 15. Engineering standards

- Type hints and docstrings on public functions. Small modules, no god files.
- Configs in YAML, no magic numbers in code. Every experiment is one YAML file plus one script.
- Pin versions in `requirements.txt`. Use `random.Random(seed)` and `numpy.random.default_rng(seed)`, never global RNG state.
- Log with `logging`, show progress with `tqdm`. Parallelise across seeds with `multiprocessing` and a `--workers` flag, default 1.
- Commit-style hygiene: after each milestone, summarise what changed and which tests pass.
- Never modify evaluation code to make results look better. If results disappoint, change the agent or the reward, retrain, and record the change in `DECISIONS.md`.

## 16. Risks and prescribed fallbacks

| Risk | Detection | Fallback (apply in this order) |
| --- | --- | --- |
| Agent does not learn | Flat reward curve, or it loses badly to Dijkstra on the toy graph | Debug on the toy graph. Reduce to delay-only reward. Use 2 bins. Lower alpha. Increase episodes. Check the update timing. |
| Routing loops | High TTL drops, rising hop counts | Stronger loop penalty, previous-hop masking, smaller TTL. |
| Ties or loses at medium load | E1 | Expected. Report honestly. Check that the traffic actually causes congestion under Dijkstra (calibration). |
| Result looks too good | Large gains at low load | Check identical traces, epsilon 0, frozen Q, held-out seeds, and that the loss accounting is the same for both methods. |
| Scalability sweep too slow | Wall-clock | Cap episodes and use fewer seeds at large N (minimum 5), profile the hot loop, use NumPy for bin computation. |
| Q-table explodes at large N | Table size metric | This is a finding. Record it as part of the boundary analysis rather than hiding it. |
| No congestion appears in the scenario | Max link utilisation under Dijkstra is below 0.6 at high load | Recalibrate flows. This is a scenario-design bug, not an agent bug. |

## 17. Kickoff prompt for the agent (paste this first)

> Read `PROJECT_SPEC.md` completely. It is the authoritative specification for this project. Do not write code yet. First reply with: (1) a summary of the project in your own words, (2) your understanding of the tick order in Section 4.2 and the learning protocol in Section 6.1, (3) the list of files you will create in Milestone M1, (4) any ambiguities you found, each with a proposed default. Then wait for my approval. Work one milestone at a time, stop at each review gate, and never fabricate results. Log every judgement call in `DECISIONS.md`.

## 18. Report outline (for M8)

1. Introduction and problem statement. 2. Background: OSPF/RIP limits, Q-routing, related work (the humans add a proper literature search, including DQN-based SDN routing). 3. System design: simulator, state, action, reward, algorithm. 4. Experimental setup: topologies, traffic, calibration, protocol, metrics. 5. Results: E1 to E5 with figures and tables. 6. Discussion: where it wins, where it does not, scalability boundary. 7. Limitations: simulation only, tabular state growth, no packet-level protocol realism, single-agent decentralised approximation. 8. Future work: function approximation (DQN), multi-agent, real SDN testbed. 9. Conclusion. Appendix: hyperparameters, reproduction steps.

## 19. Definition of done

- [ ] `python run_all.py --quick` and `--full` regenerate every figure and table from a clean clone
- [ ] All 9 test groups pass
- [ ] Q-learning vs Dijkstra shown under low, medium and bursty-high load, with 95% CIs
- [ ] Link-failure recovery result shown for both methods
- [ ] Ablations and sensitivity done (Tier 2)
- [ ] Scalability figure with an explicitly defined boundary and a boundary table
- [ ] Streamlit demo runs offline with side-by-side comparison and a kill-link button
- [ ] README, `DECISIONS.md`, and report draft complete, every number traceable to a CSV
- [ ] Every team member can explain the Q-update, the state design, and the reward design without notes
