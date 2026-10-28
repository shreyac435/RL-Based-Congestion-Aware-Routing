"""Experiment execution, statistical analysis, and logging framework."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def compute_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute sample mean and 95% confidence interval margin using t-distribution."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    h = float(se * stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, h


def ensure_dirs():
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)


def save_raw_csv(rows: List[Dict[str, Any]], filename: str):
    """Save raw results to results/raw/<filename>.csv."""
    ensure_dirs()
    path = os.path.join("results/raw", filename)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved raw results: {path} ({len(df)} rows)")


def save_figure(fig: plt.Figure, name: str):
    """Save figure in both PNG (300 dpi) and PDF formats."""
    ensure_dirs()
    png_path = os.path.join("results/figures", f"{name}.png")
    pdf_path = os.path.join("results/figures", f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {png_path} and {pdf_path}")
