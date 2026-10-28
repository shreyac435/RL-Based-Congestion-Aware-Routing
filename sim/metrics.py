"""Metric calculation utilities according to Section 10."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def jains_fairness_index(values: List[float]) -> float:
    """Calculate Jain's fairness index: (sum x)^2 / (n * sum x^2)."""
    if not values:
        return 1.0
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 1.0
    sum_x = np.sum(arr)
    sum_sq = np.sum(arr ** 2)
    n = len(arr)
    if sum_sq == 0:
        return 1.0
    return float((sum_x ** 2) / (n * sum_sq))


class EpisodeMetricsCollector:
    def __init__(self, warmup_ticks: int = 200):
        self.warmup_ticks = warmup_ticks
        self.reset()

    def reset(self):
        # Store tuples of (tick, data)
        self.deliveries: List[Tuple[int, int, int]] = []  # (tick, delay, hops)
        self.injections: List[Tuple[int, int]] = []        # (tick, count)
        self.drops: List[Tuple[int, str]] = []             # (tick, reason)
        self.link_utils: Dict[Any, List[Tuple[int, float]]] = {}

    def record_injection(self, tick: int, count: int = 1):
        self.injections.append((tick, count))

    def record_delivery(self, tick: int, delay: int, hops: int):
        self.deliveries.append((tick, delay, hops))

    def record_drop(self, tick: int, reason: str):
        self.drops.append((tick, reason))

    def record_link_util(self, tick: int, link_id: Any, util: float):
        if link_id not in self.link_utils:
            self.link_utils[link_id] = []
        self.link_utils[link_id].append((tick, util))

    @property
    def injected_count(self) -> int:
        return sum(c for _, c in self.injections)

    @property
    def delivered_count(self) -> int:
        return len(self.deliveries)

    @property
    def total_dropped(self) -> int:
        return len(self.drops)

    @property
    def delivered_delays(self) -> List[int]:
        return [d for _, d, _ in self.deliveries]

    @property
    def dropped_by_reason(self) -> Dict[str, int]:
        counts = {"queue_full": 0, "link_down": 0, "ttl": 0, "no_route": 0}
        for _, r in self.drops:
            counts[r] = counts.get(r, 0) + 1
        return counts

    def compute(self, n_ticks: int) -> Dict[str, float]:
        warmup = self.warmup_ticks if n_ticks > self.warmup_ticks else int(0.2 * n_ticks)
        eval_ticks = max(1, n_ticks - warmup)

        eval_delivs = [d for t, d, _ in self.deliveries if t >= warmup]
        eval_hops = [h for t, _, h in self.deliveries if t >= warmup]
        eval_injected = sum(c for t, c in self.injections if t >= warmup)
        eval_drops = [r for t, r in self.drops if t >= warmup]

        avg_delay = float(np.mean(eval_delivs)) if eval_delivs else 0.0
        p95_delay = float(np.percentile(eval_delivs, 95)) if eval_delivs else 0.0
        avg_hops = float(np.mean(eval_hops)) if eval_hops else 0.0

        total_pkts = len(eval_delivs) + len(eval_drops)
        loss_pct = (len(eval_drops) / total_pkts * 100.0) if total_pkts > 0 else 0.0
        throughput = len(eval_delivs) / float(eval_ticks)

        mean_link_utils = []
        for utils in self.link_utils.values():
            filtered = [u for t, u in utils if t >= warmup]
            if filtered:
                mean_link_utils.append(float(np.mean(filtered)))

        fairness = jains_fairness_index(mean_link_utils)
        max_util = float(np.max(mean_link_utils)) if mean_link_utils else 0.0

        drop_reasons = {"queue_full": 0, "link_down": 0, "ttl": 0, "no_route": 0}
        for r in eval_drops:
            drop_reasons[r] = drop_reasons.get(r, 0) + 1

        return {
            "avg_delay": avg_delay,
            "p95_delay": p95_delay,
            "avg_hops": avg_hops,
            "loss_pct": loss_pct,
            "throughput": throughput,
            "load_balance": fairness,
            "max_link_util": max_util,
            "injected": float(eval_injected if eval_injected > 0 else len(eval_delivs) + len(eval_drops)),
            "delivered": float(len(eval_delivs)),
            "dropped": float(len(eval_drops)),
            "dropped_queue_full": float(drop_reasons["queue_full"]),
            "dropped_link_down": float(drop_reasons["link_down"]),
            "dropped_ttl": float(drop_reasons["ttl"]),
        }
