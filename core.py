# core.py
"""
Core utilities for the RYE Analyzer.

This module keeps the original public functions:
  - safe_float
  - compute_rye
  - compute_rye_from_df
  - rolling_series
  - summarize_series

and adds new, fully optional helpers:
  - load_table, normalize_columns, guess_numeric
  - PRESETS (AI/Biology vocabulary)
  - compute_rye_stepwise (adds RYE_step, RYE_cum to a DataFrame)
  - stability_zones, resilience_index
  - batch_analyze
  - plot_timeseries, plot_rye, plot_compare  (matplotlib, one fig per plot)

Nothing existing should break; your app can adopt the new pieces incrementally.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


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
    Compute RYE (Repair Yield per Energy).
    Formula: RYE = repair / energy
    Replaces zeros or NaNs in energy to avoid divide-by-zero.
    """
    repair = np.asarray(repair, dtype=float)
    energy = np.asarray(energy, dtype=float)
    energy = np.where((energy == 0) | np.isnan(energy), np.nan, energy)
    return repair / energy


def compute_rye_from_df(df: pd.DataFrame, repair_col: str, energy_col: str) -> np.ndarray:
    """Compute RYE directly from a dataframe using column names."""
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
    """Compute simple stats for a numeric array."""
    s = pd.Series(arr, dtype=float).dropna()
    if len(s) == 0:
        return {"mean": 0, "median": 0, "max": 0, "min": 0, "count": 0}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "max": float(s.max()),
        "min": float(s.min()),
        "count": int(s.count()),
    }


# ---------------------------
# New: IO helpers
# ---------------------------

def load_table(file) -> pd.DataFrame:
    """
    Read CSV / TSV / XLSX into a DataFrame.
    `file` may be a file-like object from Streamlit uploader or a path.
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
# New: Vocabulary presets
# ---------------------------

PRESETS = {
    "AI": {
        "repair_label": "Repair",
        "energy_label": "Energy",
        "tooltips": {
            "Repair": "Measured improvement (accuracy, loss reduction, stability, etc.).",
            "Energy": "Correction effort: steps, tokens, compute, or cost."
        }
    },
    "Biology": {
        "repair_label": "Recovery",
        "energy_label": "Resource input",
        "tooltips": {
            "Recovery": "Improvement toward baseline or stability (e.g., function restored).",
            "Resource input": "Metabolic cost, time, or resources invested."
        }
    }
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
    Safe against divide-by-zero via small epsilon.
    """
    eps = 1e-12
    dR = pd.to_numeric(df[repair_col], errors="coerce").diff()
    dE = pd.to_numeric(df[energy_col], errors="coerce").diff().clip(lower=eps)
    df[out_step] = dR / dE

    R0 = pd.to_numeric(df[repair_col], errors="coerce").iloc[0]
    E0 = pd.to_numeric(df[energy_col], errors="coerce").iloc[0]
    num = pd.to_numeric(df[repair_col], errors="coerce") - R0
    den = (pd.to_numeric(df[energy_col], errors="coerce") - E0).replace(0, eps)
    df[out_cum] = num / den
    return df


# ---------------------------
# New: Stability and resilience helpers
# ---------------------------

def stability_zones(series: pd.Series, lower: float, upper: float) -> List[Tuple[int, int]]:
    """
    Return index spans where `series` stays within [lower, upper].
    Example output: [(10, 25), (40, 57)]
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
            "mean_RYE_step": float(pd.to_numeric(df["RYE_step"], errors="coerce").mean())
        })
    return pd.DataFrame(rows)


# ---------------------------
# New: Lightweight plots (matplotlib, no seaborn, one fig per plot)
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
