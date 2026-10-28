"""Network topology generators."""
from __future__ import annotations

from typing import Dict, List, Tuple
import networkx as nx
import numpy as np


def create_toy_diamond() -> nx.Graph:
    """Create a 6-node toy diamond graph for debugging and sanity tests.
    
    Nodes: 0 to 5.
    Source: 0, Destination: 5.
    Short path: 0 - 1 - 5 (2 hops)
    Long path:  0 - 2 - 3 - 4 - 5 (4 hops)
    Cross link: 1 - 3
    """
    g = nx.Graph()
    g.add_nodes_from(range(6))
    edges = [
        (0, 1), (1, 5),          # Short path (2 hops)
        (0, 2), (2, 3), (3, 4), (4, 5),  # Long path (4 hops)
        (1, 3),                  # Cross link for alternative routing
    ]
    g.add_edges_from(edges)
    return g


def create_abilene() -> nx.Graph:
    """Create the 12-node, 15-link Abilene Internet2 topology.
    
    Hardcoded verified edge list from spec Section 8.1.
    """
    g = nx.Graph()
    nodes = [
        "ATLAM5", "ATLAng", "CHINng", "DNVRng", "HSTNng", "IPLSng",
        "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng"
    ]
    g.add_nodes_from(nodes)
    edges = [
        ("ATLAM5", "ATLAng"),
        ("ATLAng", "HSTNng"),
        ("ATLAng", "IPLSng"),
        ("ATLAng", "WASHng"),
        ("CHINng", "IPLSng"),
        ("CHINng", "NYCMng"),
        ("DNVRng", "KSCYng"),
        ("DNVRng", "SNVAng"),
        ("DNVRng", "STTLng"),
        ("HSTNng", "KSCYng"),
        ("HSTNng", "LOSAng"),
        ("IPLSng", "KSCYng"),
        ("LOSAng", "SNVAng"),
        ("NYCMng", "WASHng"),
        ("SNVAng", "STTLng"),
    ]
    g.add_edges_from(edges)
    return g


def create_waxman(n: int, seed: int, alpha: float = 0.5, beta: float = 0.4) -> nx.Graph:
    """Create a connected Waxman random graph with N nodes targeting avg degree 3-4."""
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        # Generate random node coordinates in [0, 1] x [0, 1]
        coords = rng.uniform(0.0, 1.0, size=(n, 2))
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = dist_matrix[j, i] = d
        l_max = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1.0

        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                prob = alpha * np.exp(-d / (beta * l_max))
                if rng.random() < prob:
                    g.add_edge(i, j)

        if nx.is_connected(g):
            # Check avg degree is reasonable (at least 2)
            degrees = [deg for _, deg in g.degree()]
            if np.mean(degrees) >= 2.0:
                return g

    # Fallback to connected cycle + random shortcuts if random waxman failed to connect
    g = nx.cycle_graph(n)
    for i in range(n):
        target = int(rng.integers(0, n))
        if target != i and not g.has_edge(i, target):
            g.add_edge(i, target)
    return g
