# core.py
"""
Core utilities for the RYE Analyzer.

Kept (backward compatible):
  - safe_float
  - compute_rye
  - compute_rye_from_df
  - rolling_series
  - summarize_series

Added (optional upgrades):
  - load_table, normalize_columns, guess_numeric
  - PRESETS (AI / Biology / Robotics)
  - compute_rye_stepwise (adds RYE_step, RYE_cum)
  - stability_zones, resilience_index
  - compute_resilience_index (added into summarize_series output)
  - clip_outliers, smooth_series, safe_div
  - append_rye_columns (returns enriched DataFrame)
  - batch_analyze (multi-file summaries)
  - plot_timeseries, plot_rye, plot_compare  (matplotlib, one figure per plot)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Original functions (kept)
# ---------------------------

def safe_float(x):
    """Convert safely to float; return NaN if it fails."""
    try:
        return float(x)
    except Exception:
        return np.nan


def compute_rye(repair: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """
    Compute RYE (Repair Yield per Energy) elementwise.
    Formula: RYE = repair / energy
    Replaces zeros or NaNs in energy to avoid divide-by-zero.
    """
    repair = np.asarray(repair, dtype=float)
    energy = np.asarray(energy, dtype=float)
    energy = np.where((energy == 0) | np.isnan(energy), np.nan, energy)
    return repair / energy


def compute_rye_from_df(df: pd.DataFrame, repair_col: str, energy_col: str) -> np.ndarray:
    """Compute RYE directly from a DataFrame using column names."""
    r = df[repair_col].apply(safe_float).to_numpy()
    e = df[energy_col].apply(safe_float).to_numpy()
    return compute_rye(r, e)


def rolling_series(arr: np.ndarray, window: int) -> np.ndarray:
    """Return rolling mean with a given window size."""
    s = pd.Series(arr, dtype=float)
    if window <= 1:
        return s.to_numpy()
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


def summarize_series(arr: np.ndarray) -> dict:
    """Compute simple stats for a numeric array, plus resilience."""
    s = pd.Series(arr, dtype=float).dropna()
    if len(s) == 0:
        return {"mean": 0, "median": 0, "max": 0, "min": 0, "count": 0, "resilience": 0}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "max": float(s.max()),
        "min": float(s.min()),
        "count": int(s.count()),
        "resilience": compute_resilience_index(s.to_numpy()),
    }


# ---------------------------
# New: general helpers
# ---------------------------

def safe_div(num: Union[float, np.ndarray], den: Union[float, np.ndarray], eps: float = 1e-12):
    """Numerically safe division."""
    return np.asarray(num, dtype=float) / (np.asarray(den, dtype=float) + eps)


def clip_outliers(arr: Iterable[float], lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """Clip values to percentile band to reduce the impact of outliers."""
    s = pd.Series(arr, dtype=float).dropna()
    if s.empty:
        return np.asarray(arr, dtype=float)
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return np.clip(np.asarray(arr, dtype=float), lo, hi)


def smooth_series(arr: Iterable[float], window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    return rolling_series(np.asarray(arr, dtype=float), max(1, int(window)))


# ---------------------------
# New: IO helpers
# ---------------------------

def load_table(file) -> pd.DataFrame:
    """
    Read CSV / TSV / XLSX into a DataFrame.
    `file` may be a file-like object (Streamlit uploader) or a path string.
    """
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith(".tsv"):
        df = pd.read_csv(file, sep="\t")
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Use CSV/TSV/XLS/XLSX.")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim, lowercase, and snake_case headers for easier matching."""
    return df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})


def guess_numeric(df: pd.DataFrame) -> List[str]:
    """Return likely numeric columns for mapping UI."""
    return df.select_dtypes(include="number").columns.tolist()


# ---------------------------
# New: Vocabulary presets (extendable)
# ---------------------------

PRESETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "AI": {
        "repair_label": "Performance",
        "energy_label": "Energy",
        "tooltips": {
            "Performance": "Measured improvement (accuracy, loss reduction, stability, etc.).",
            "Energy": "Correction effort: steps, tokens, compute, or cost."
        }
    },
    "Biology": {
        "repair_label": "Recovery",
        "energy_label": "Effort",
        "tooltips": {
            "Recovery": "Improvement toward baseline or stability (e.g., function restored).",
            "Effort": "Metabolic cost, time, or resources invested."
        }
    },
    "Robotics": {
        "repair_label": "Task success",
        "energy_label": "Power",
        "tooltips": {
            "Task success": "Improvement toward nominal behavior (success rate, error reduction, uptime, MTTR).",
            "Power": "Electrical/actuation energy, battery draw, or equivalent effort units."
        }
    },
    # Examples to extend later:
    # "Economics": {"repair_label":"Productivity","energy_label":"Cost","tooltips": {...}}
    # "Medicine":  {"repair_label":"Healing","energy_label":"Treatment effort","tooltips": {...}}
}


# ---------------------------
# New: Stepwise and cumulative RYE on a DataFrame
# ---------------------------

def compute_rye_stepwise(
    df: pd.DataFrame,
    repair_col: str,
    energy_col: str,
    out_step: str = "RYE_step",
    out_cum: str = "RYE_cum"
) -> pd.DataFrame:
    """
    Adds two columns to `df`:
      - RYE_step: Δrepair / Δenergy between consecutive rows
      - RYE_cum : (repair - repair0) / (energy - energy0)
    """
    eps = 1e-12
    r = pd.to_numeric(df[repair_col], errors="coerce")
    e = pd.to_numeric(df[energy_col], errors="coerce")

    dR = r.diff()
    dE = e.diff().clip(lower=eps)
    df[out_step] = dR / dE

    R0 = r.iloc[0]
    E0 = e.iloc[0]
    num = r - R0
    den = (e - E0).replace(0, eps)
    df[out_cum] = num / den
    return df


# ---------------------------
# New: Stability and resilience helpers
# ---------------------------

def stability_zones(series: pd.Series, lower: float, upper: float) -> List[Tuple[int, int]]:
    """
    Return index spans where `series` stays within [lower, upper].
    Example: [(10, 25), (40, 57)]
    """
    inside = series.between(lower, upper)
    spans, start = [], None
    for i, ok in enumerate(inside):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(series) - 1))
    return spans


def resilience_index(series: pd.Series, target: float, pct: float = 0.95) -> Optional[int]:
    """
    First index where series reaches pct * target (e.g., 95% of baseline).
    Returns None if never reached.
    """
    thresh = target * pct
    hits = series[series >= thresh]
    return int(hits.index[0]) if len(hits) else None


def compute_resilience_index(arr: Iterable[float]) -> float:
    """
    Simple stability/resilience index for an array.
    1.0 = perfectly stable; lower values indicate more volatility relative to mean level.
    """
    arr = np.asarray(arr, dtype=float)
    s = pd.Series(arr).dropna()
    if len(s) == 0:
        return 0.0
    mean = s.mean()
    std = s.std()
    if mean == 0:
        return 0.0
    return float(max(0.0, 1.0 - (std / (abs(mean) + 1e-8))))


# ---------------------------
# New: Enrichment helpers
# ---------------------------

def append_rye_columns(df: pd.DataFrame, repair_col: str, energy_col: str,
                       out_step: str = "RYE_step", out_cum: str = "RYE_cum") -> pd.DataFrame:
    """
    Returns a copy of df with RYE_step and RYE_cum added (non-destructive).
    """
    d = df.copy()
    return compute_rye_stepwise(d, repair_col, energy_col, out_step=out_step, out_cum=out_cum)


# ---------------------------
# New: Batch comparison
# ---------------------------

def batch_analyze(files, repair_col: str, energy_col: str) -> pd.DataFrame:
    """
    Compute summary metrics across many uploaded files.
    Returns a DataFrame with per-file final RYE_cum and mean RYE_step.
    """
    rows = []
    for f in files:
        df = normalize_columns(load_table(f))
        df = compute_rye_stepwise(df, repair_col, energy_col)
        rows.append({
            "file": getattr(f, "name", "dataset"),
            "final_RYE_cum": float(df["RYE_cum"].iloc[-1]) if len(df) else np.nan,
            "mean_RYE_step": float(pd.to_numeric(df["RYE_step"], errors="coerce").mean()),
            "rows": int(len(df)),
        })
    return pd.DataFrame(rows)


# ---------------------------
# New: Lightweight plots (matplotlib; one fig per plot; no seaborn)
# ---------------------------

def plot_timeseries(df: pd.DataFrame, cols: List[str]):
    fig, ax = plt.subplots()
    df[cols].plot(ax=ax)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("Time series")
    fig.tight_layout()
    return fig


def plot_rye(df: pd.DataFrame, step_col: str = "RYE_step", cum_col: str = "RYE_cum"):
    fig, ax = plt.subplots()
    df[[step_col, cum_col]].plot(ax=ax)
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.set_title("RYE (step & cumulative)")
    fig.tight_layout()
    return fig


def plot_compare(dfs: List[pd.DataFrame], labels: List[str], cum_col: str = "RYE_cum"):
    fig, ax = plt.subplots()
    for d, l in zip(dfs, labels):
        ax.plot(d.index, d[cum_col], label=l)
    ax.set_xlabel("Index")
    ax.set_ylabel(cum_col)
    ax.set_title("Comparison")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_compare(dfs: List[pd.DataFrame], labels: List[str], cum_col: str = "RYE_cum"):
    fig, ax = plt.subplots()
    for d, l in zip(dfs, labels):
        ax.plot(d.index, d[cum_col], label=l)
    ax.set_xlabel("Index")
    ax.set_ylabel(cum_col)
    ax.set_title("Comparison")
    ax.legend()
    fig.tight_layout()
    return figun 
