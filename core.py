# core.py
# Core utilities for RYE Analyzer:
# - file loading (CSV/TSV/XLS/XLSX)
# - column normalization
# - robust numeric coercion
# - RYE computation (Δperformance / energy)
# - rolling helpers & summaries (incl. resilience)
# - optional analytics: regimes, correlation, noise floor, bootstrap bands
# Exposes: load_table, normalize_columns, safe_float, compute_rye_from_df,
#          rolling_series, summarize_series, PRESETS,
#          detect_regimes, energy_delta_performance_correlation,
#          estimate_noise_floor, bootstrap_rolling_mean

from __future__ import annotations
import io
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ------------------------------
# PRESETS
# ------------------------------
# If you keep a separate presets.py (your long list), we re-export it.
try:
    from presets import PRESETS  # your full dict lives here
except Exception:
    # Minimal fallback so the app still runs if presets.py is missing
    @dataclass(frozen=True)
    class Preset:
        name: str
        time: List[str]
        performance: List[str]
        energy: List[str]
        domain: Optional[str] = None
        default_rolling: int = 10
        tooltips: Optional[Dict[str, str]] = None

    def _kw(*items):  # small helper
        return list(dict.fromkeys([s.strip() for s in items if s]))

    PRESETS: Dict[str, Preset] = {
        "AI": Preset(
            "AI",
            time=_kw("step", "iteration", "epoch", "t", "time"),
            performance=_kw("accuracy", "acc", "f1", "reward", "score", "coherence", "loss_inv"),
            energy=_kw("tokens", "compute", "energy", "cost", "gradient_updates"),
            domain="ai",
            default_rolling=10,
            tooltips={"coherence": "Higher is better", "loss_inv": "1/loss style metric"},
        ),
        "Biology": Preset(
            "Biology",
            time=_kw("time", "t", "hours", "days", "samples"),
            performance=_kw("viability", "function", "yield", "recovery", "signal", "growth", "fitness"),
            energy=_kw("dose", "stressor", "input", "energy", "treatment", "drug", "radiation"),
            domain="bio",
            default_rolling=10,
        ),
        "Robotics": Preset(
            "Robotics",
            time=_kw("t", "time", "cycle", "episode"),
            performance=_kw("task_success", "score", "stability", "tracking_inv", "uptime", "mean_reward"),
            energy=_kw("power", "torque_int", "battery_used", "energy", "effort", "cpu_load"),
            domain="robot",
            default_rolling=10,
        ),
    }

# ------------------------------
# File IO
# ------------------------------
def load_table(src) -> pd.DataFrame:
    """
    Read CSV/TSV/XLS/XLSX from a path or a Streamlit UploadedFile/BytesIO.
    Returns a DataFrame (may be empty).
    """
    # Stream/bytes?
    if hasattr(src, "read") and not isinstance(src, (str, bytes)):
        data = src.read()
        name = getattr(src, "name", "upload")
        buf = io.BytesIO(data)
        if name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(buf)
        # default: CSV/TSV sniff
        text = data.decode("utf-8", errors="replace")
        sep = "\t" if "\t" in text and "," not in text.splitlines()[0] else ","
        return pd.read_csv(io.StringIO(text), sep=sep)

    # Path-like
    path = str(src)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    # CSV/TSV
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snake-case headers, strip whitespace, collapse punctuation.
    """
    def norm(c: str) -> str:
        s = c.strip().lower()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "col"
    df = df.copy()
    df.columns = [norm(str(c)) for c in df.columns]
    return df

# ------------------------------
# Numerics
# ------------------------------
def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        # handle strings like "1,234.5" or "1 234"
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return float("nan")

def _coerce_numeric(series: Iterable) -> np.ndarray:
    arr = np.array([safe_float(v) for v in series], dtype=float)
    return arr

# ------------------------------
# RYE core
# ------------------------------
def compute_rye_from_df(
    df: pd.DataFrame,
    repair_col: str,
    energy_col: str,
    time_col: Optional[str] = None,
) -> np.ndarray:
    """
    RYE per step = max(Δperformance, 0) / max(energy, eps)
    (Protects against negative or zero energy and NaNs.)
    """
    perf = _coerce_numeric(df[repair_col])
    energy = _coerce_numeric(df[energy_col])

    # Δperformance (step-to-step improvement). If decreasing, treat negative
    # deltas as zero repair (no yield) — this matches the "repair" intuition.
    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)
    dperf = np.maximum(dperf, 0.0)

    eps = 1e-9
    denom = np.where(np.isfinite(energy) & (energy > 0), energy, eps)
    rye = dperf / denom
    rye = np.where(np.isfinite(rye), rye, 0.0)
    return rye

def rolling_series(series: Sequence[float], window: int) -> np.ndarray:
    """
    Simple moving average with edge-handling.
    """
    s = pd.Series(series, dtype=float)
    if window <= 1:
        return s.values
    return s.rolling(window=window, min_periods=1, center=False).mean().values

def summarize_series(series: Sequence[float]) -> Dict[str, float]:
    """
    Mean/median/min/max/count/std and a bounded 'resilience' in [0,1].
    Resilience ~ 1 - CV (clipped 0..1), with small-sample protection.
    """
    a = np.array(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0.0, "std": 0.0, "resilience": 0.0}
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a))
    cv = std / (abs(mean) + 1e-9)
    resilience = float(np.clip(1.0 - cv, 0.0, 1.0))
    return {
        "mean": mean,
        "median": float(np.nanmedian(a)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "count": float(a.size),
        "std": std,
        "resilience": resilience,
    }

# ------------------------------
# Optional analytics (safe fallbacks)
# ------------------------------
def detect_regimes(series: Sequence[float], min_len: int = 5, gap: float = 0.05) -> List[Dict[str, Union[int, str]]]:
    """
    Heuristic regime detector: segments where rolling mean stays within a band.
    Returns list of {'start': i0, 'end': i1, 'label': text}.
    """
    x = np.array(series, dtype=float)
    if x.size == 0:
        return []
    roll = rolling_series(x, max(3, min_len))
    regimes = []
    s = 0
    for i in range(1, len(roll)):
        if abs(roll[i] - roll[i-1]) > gap:
            if i - 1 - s + 1 >= min_len:
                mean_seg = float(np.nanmean(x[s:i]))
                regimes.append({"start": int(s), "end": int(i - 1), "label": f"mean≈{mean_seg:.3f}"})
            s = i
    if len(roll) - s >= min_len:
        mean_seg = float(np.nanmean(x[s:]))
        regimes.append({"start": int(s), "end": int(len(roll) - 1), "label": f"mean≈{mean_seg:.3f}"})
    return regimes

def energy_delta_performance_correlation(
    df: pd.DataFrame,
    perf_col: str,
    energy_col: str
) -> Dict[str, float]:
    """
    Pearson & Spearman correlation between energy and Δperformance.
    """
    from scipy.stats import pearsonr, spearmanr  # noqa
    perf = _coerce_numeric(df[perf_col])
    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)

    energy = _coerce_numeric(df[energy_col])
    m = np.isfinite(dperf) & np.isfinite(energy)
    if m.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}
    try:
        pr = float(pearsonr(energy[m], dperf[m]).statistic)
    except Exception:
        pr = float("nan")
    try:
        sr = float(spearmanr(energy[m], dperf[m]).statistic)
    except Exception:
        sr = float("nan")
    return {"pearson": pr, "spearman": sr}

def estimate_noise_floor(series: Sequence[float]) -> Dict[str, float]:
    """
    Very simple noise floor: std of first differences and IQR.
    """
    a = np.array(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return {"diff_std": float("nan"), "iqr": float("nan")}
    d = np.diff(a)
    diff_std = float(np.nanstd(d))
    q1, q3 = np.nanpercentile(a, [25, 75])
    return {"diff_std": diff_std, "iqr": float(q3 - q1)}

def bootstrap_rolling_mean(
    series: Sequence[float],
    window: int,
    n_boot: int = 100,
    q_low: float = 0.10,
    q_mid: float = 0.50,
    q_high: float = 0.90
) -> Dict[str, List[float]]:
    """
    Bootstrap envelopes of the rolling mean (low/mid/high quantiles).
    """
    x = np.array(series, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if x.size == 0:
        return {"low": [], "mid": [], "high": []}

    rolls = []
    rng = np.random.default_rng(12345)
    n = len(x)
    for _ in range(max(10, n_boot)):
        idx = rng.integers(0, n, size=n)  # bootstrap sample with replacement
        rs = rolling_series(x[idx], max(1, window))
        # Pad/truncate to original length
        if len(rs) < n:
            rs = np.pad(rs, (0, n - len(rs)), constant_values=rs[-1] if len(rs) > 0 else 0.0)
        elif len(rs) > n:
            rs = rs[:n]
        rolls.append(rs)

    R = np.vstack(rolls)
    low = np.nanquantile(R, q_low, axis=0)
    mid = np.nanquantile(R, q_mid, axis=0)
    high = np.nanquantile(R, q_high, axis=0)
    return {"low": low.tolist(), "mid": mid.tolist(), "high": high.tolist()}

# Explicit export surface for the app
__all__ = [
    "load_table",
    "normalize_columns",
    "safe_float",
    "compute_rye_from_df",
    "rolling_series",
    "summarize_series",
    "PRESETS",
    # optional analytics
    "detect_regimes",
    "energy_delta_performance_correlation",
    "estimate_noise_floor",
    "bootstrap_rolling_mean",
]
