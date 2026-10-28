# Design Decisions and Defaults

This document records architectural decisions, defaults, and parameters for the RL-Based Congestion-Aware Routing project.

## 1. Network & Simulator Modeling
- **Link Data Structure**: Each undirected edge between $u$ and $v$ is modeled as two independent directed links $(u, v)$ and $(v, u)$. Each link has a FIFO queue (`collections.deque`) capped at `buffer` (default 20), a transmission capacity (default 2 pkts/tick), and propagation delay (default 1 tick).
- **Propagation Pipe**: Implemented as a FIFO deque of `(arrival_tick, packet)` items per link. When ticks advance, expired packets are transferred to the target node's arrival buffer.
- **Link Failure Handling**: At the failure tick $t_f$, the link is marked `up = False`. Any packets currently waiting in the link's FIFO buffer or in-flight in its propagation pipe are immediately dropped with reason `link_down`.

## 2. State & Action Representation
- **State Tuple**: $s = (n, \text{dst}, c_1, c_2, \dots, c_k)$, where $n$ is current node, $\text{dst}$ is destination, and $c_i \in \{0, 1, 2\}$ is the congestion level (0: Low $< 0.33$, 1: Medium $< 0.66$, 2: High $\ge 0.66$) of outgoing link $i$ to sorted neighbor $i$.
- **Action Selection**: Index into sorted neighbors. If a link is down, it is masked. The immediate previous hop is masked unless no other path exists.

## 3. Rewards & Q-Updates
- **Reward**: $r = -\text{hop\_delay} - P_{\text{cong}} \cdot \mathbf{1}_{\text{high}} - P_{\text{drop}} \cdot \mathbf{1}_{\text{drop}} + R_{\text{deliv}} \cdot \mathbf{1}_{\text{deliv}} - P_{\text{loop}} \cdot \mathbf{1}_{\text{loop}}$, with defaults $P_{\text{cong}}=2, P_{\text{drop}}=30, R_{\text{deliv}}=10, P_{\text{loop}}=5$.
- **Q-learning Update**: Off-policy bootstrap with $\gamma=0.95, \alpha=0.2$. Terminal transition (delivery or drop) uses next value 0.
- **SARSA Update**: On-policy bootstrap using the actual action chosen at the next hop.

## 4. Topologies
- **Abilene**: Hard-coded 12 nodes, 15 bidirectional edges verified from standard Internet2 topology.
- **Toy Diamond**: 6 nodes, 7 bidirectional edges with a 2-hop short path and 3-hop alternate path.
- **Waxman Family**: Random Waxman graphs generated with reproducible seeds for $N \in \{6, 12, 25, 50, 100\}$.
