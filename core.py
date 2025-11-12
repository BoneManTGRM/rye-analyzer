# core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple, Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Preset:
    time: str
    domain: str
    performance: str
    energy: str
    rolling_window: int = 10

# Built-in presets (expandable)
PRESETS: Mapping[str, Preset] = {
    "AI":       Preset(time="step", domain="domain", performance="accuracy",   energy="energy", rolling_window=10),
    "Biology":  Preset(time="time", domain="domain", performance="performance",energy="energy", rolling_window=10),
    "Robotics": Preset(time="t",    domain="domain", performance="score",      energy="power",  rolling_window=10),
}

# ---------- IO ----------
def parse_csv(file) -> pd.DataFrame:
    """Read CSV/TSV; infer delimiter from extension."""
    name = (getattr(file, "name", "") or "").lower()
    sep = "\t" if name.endswith(".tsv") else ","
    df = pd.read_csv(file, sep=sep)
    return df

# ---------- RYE core ----------
def compute_rye_series(perf: np.ndarray, energy: np.ndarray, smooth: str = "mean", window: int = 1) -> np.ndarray:
    """
    RYE_t = Δperf_t / Δenergy_t, with optional pre-smoothing of perf and energy.
    smooth: 'mean' | 'median' | 'none'
    """
    perf = np.asarray(perf, dtype=float)
    energy = np.asarray(energy, dtype=float)

    if window and window > 1:
        if smooth == "mean":
            perf = _rolling(perf, window, how="mean")
            energy = _rolling(energy, window, how="mean")
        elif smooth == "median":
            perf = _rolling(perf, window, how="median")

    d_perf = np.diff(perf, prepend=perf[0])
    d_energy = np.diff(energy, prepend=energy[0])
    d_energy[d_energy == 0] = np.nan

    rye = d_perf / d_energy
    rye = np.nan_to_num(rye, nan=0.0, posinf=0.0, neginf=0.0)
    return rye

def rolling_series(x: np.ndarray, window: int, how: str = "mean") -> np.ndarray:
    return _rolling(np.asarray(x, dtype=float), window, how=how)

def _rolling(x: np.ndarray, window: int, how: str = "mean") -> np.ndarray:
    s = pd.Series(x, dtype=float)
    if window <= 1:
        return s.to_numpy()
    if how == "median":
        out = s.rolling(window=window, min_periods=1, center=False).median()
    else:
        out = s.rolling(window=window, min_periods=1, center=False).mean()
    return out.to_numpy()

def summarize_rye(rye: np.ndarray, resilience_window: int = 10) -> Dict[str, float]:
    arr = np.asarray(rye, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "count": 0, "resilience": 0.0, "auc": 0.0}
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    vmax = float(np.max(arr))
    vmin = float(np.min(arr))

    # simple resilience: 1 - normalized rolling std
    rs = pd.Series(arr, dtype=float).rolling(window=max(2, resilience_window), min_periods=2).std()
    rstd = float(np.nanmean(rs)) if rs.size else 0.0
    resilience = max(0.0, min(1.0, 1.0 - rstd))

    # area under (RYE >= 0) curve (rough proxy)
    auc = float(np.trapz(np.clip(arr, 0, None)))

    return {"mean": mean, "median": median, "max": vmax, "min": vmin, "count": n, "resilience": resilience, "auc": auc}

def target_share(rye: np.ndarray, threshold: float) -> float:
    """Fraction of points with RYE >= threshold."""
    arr = np.asarray(rye, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= threshold))

def cumulative_rye(rye: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(rye, dtype=float))

# ---------- Domain helpers ----------
def build_plot_series_from_df(df: pd.DataFrame, cfg: Preset, smooth: str, smooth_window: int) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    if cfg.domain not in df.columns:
        return out
    for dom, grp in df.groupby(cfg.domain):
        rye = compute_rye_series(
            perf=grp[cfg.performance].astype(float).to_numpy(),
            energy=grp[cfg.energy].astype(float).to_numpy(),
            smooth=smooth, window=smooth_window
        )
        roll = rolling_series(rye, cfg.rolling_window)
        out[str(dom)] = {"RYE": list(rye), "RYE rolling": list(roll)}
    return out

# ---------- Outliers ----------
def apply_outlier_policy(df: pd.DataFrame, cols: List[str], policy: str, z: float = 3.0, p_low: float = 1.0, p_high: float = 99.0) -> pd.DataFrame:
    """Return a copy with outliers handled."""
    out = df.copy()
    if policy == "none":
        return out

    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype(float)
        if policy == "zscore":
            mu, sd = s.mean(), s.std(ddof=0)
            mask = (sd > 0) & (np.abs((s - mu) / sd) > z)
            out.loc[mask, c] = np.nan
        elif policy == "clip_pct":
            lo, hi = np.percentile(s.dropna(), [p_low, p_high])
            out[c] = s.clip(lower=lo, upper=hi)
    return out

# ---------- Demo data ----------
def make_demo(n: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(n)
    domain = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    base = np.cumsum(rng.normal(0.02, 0.015, size=n)) + 0.5
    bumps = (np.sin(np.linspace(0, 6, n)) + 1) * 0.05
    performance = base + bumps + rng.normal(0, 0.01, size=n)
    energy = np.cumsum(np.clip(rng.normal(1.2, 0.2, size=n), 0.6, 1.8))
    return pd.DataFrame({"time": time, "domain": domain, "performance": performance, "energy": energy})
