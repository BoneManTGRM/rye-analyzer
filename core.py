# core.py
# Core utilities for RYE Analyzer:
# - file loading (CSV/TSV/XLS/XLSX + optional Parquet/Feather/JSON/NetCDF/Xarray)
# - column normalization + robust auto-inference (aliases + preset hints)
# - robust numeric coercion
# - RYE computation (Δperformance / energy) with guards
# - rolling helpers & summaries (incl. resilience)
# - optional analytics: regimes, correlation, noise floor, bootstrap bands
# Exposes: load_table, normalize_columns, safe_float, compute_rye_from_df,
#          rolling_series, summarize_series, PRESETS,
#          detect_regimes, energy_delta_performance_correlation,
#          estimate_noise_floor, bootstrap_rolling_mean,
#          infer_columns, COLUMN_ALIASES

from __future__ import annotations
import io
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ------------------------------
# PRESETS (imports your long list if present)
# ------------------------------
try:
    # If you keep a large presets.py, we use it.
    from presets import PRESETS  # type: ignore
except Exception:
    # Minimal + expanded fallback so the app still runs if presets.py is missing
    @dataclass(frozen=True)
    class Preset:
        name: str
        time: List[str]
        performance: List[str]
        energy: List[str]
        domain: Optional[str] = None
        default_rolling: int = 10
        tooltips: Optional[Dict[str, str]] = None

    def _kw(*items) -> List[str]:
        # de-duplicate, keep order, strip whitespace
        return list(dict.fromkeys([s.strip() for s in items if s]))

    PRESETS: Dict[str, Preset] = {
        # --- Your existing core trio ---
        "AI": Preset(
            "AI",
            time=_kw("step", "iteration", "epoch", "t", "time"),
            performance=_kw("accuracy", "acc", "f1", "reward", "score", "coherence", "loss_inv", "bleu", "rouge"),
            energy=_kw("tokens", "compute", "energy", "cost", "gradient_updates", "lr", "batch_tokens"),
            domain="ai",
            default_rolling=10,
            tooltips={
                "coherence": "Higher is better.",
                "loss_inv": "1/loss style metric; increase implies better.",
            },
        ),
        "Biology": Preset(
            "Biology",
            time=_kw("time", "t", "hours", "days", "samples"),
            performance=_kw("viability", "function", "yield", "recovery", "signal", "od", "growth", "fitness"),
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

        # --- Marine / field science additions (new) ---
        "Marine Biology": Preset(
            "Marine Biology",
            time=_kw("time", "t", "date", "day", "doy", "timestamp", "sample_time"),
            performance=_kw(
                "survival", "growth", "calcification", "recruitment", "photosynthesis",
                "chlorophyll", "chl", "coverage", "abundance", "diversity", "shannon", "richness"
            ),
            energy=_kw(
                "dose", "nutrients", "nitrate", "phosphate", "silicate", "light", "par",
                "effort", "treatment", "temperature", "temp", "pco2", "salinity", "stress"
            ),
            domain="marine",
            default_rolling=10,
            tooltips={
                "chl": "Chlorophyll-a proxy for primary productivity.",
                "par": "Photosynthetically Active Radiation.",
            },
        ),
        "Fisheries": Preset(
            "Fisheries",
            time=_kw("time", "t", "date", "trip", "haul", "set", "tow"),
            performance=_kw("cpue", "yield", "biomass", "catch_rate", "survival", "recruitment"),
            energy=_kw("effort", "soak_time", "net_hours", "trawl_hours", "fuel", "cost"),
            domain="fisheries",
            default_rolling=10,
        ),
        "Coral Reef Monitoring": Preset(
            "Coral Reef Monitoring",
            time=_kw("time", "t", "date", "survey", "dive"),
            performance=_kw("live_coral_cover", "juvenile_density", "calcification", "photosynthesis", "recovery"),
            energy=_kw("intervention", "outplanting", "nursery_cost", "effort", "par", "dose"),
            domain="reef",
            default_rolling=10,
        ),
        "Oceanography/CTD": Preset(
            "Oceanography/CTD",
            time=_kw("time", "t", "cast", "profile", "date"),
            performance=_kw("signal", "stability", "coherence", "recovery", "oxygen", "chlorophyll", "fluorescence"),
            energy=_kw("pump_power", "ship_time", "fuel", "cost", "cast_depth", "niskin_trips"),
            domain="ctd",
            default_rolling=10,
        ),
        "Aquaculture": Preset(
            "Aquaculture",
            time=_kw("time", "t", "day", "date", "batch"),
            performance=_kw("growth_rate", "survival", "feed_conversion_inv", "yield", "biomass", "health_score"),
            energy=_kw("feed", "aeration_power", "oxygenation", "water_exchange", "temperature", "dose", "cost"),
            domain="aq",
            default_rolling=10,
            tooltips={"feed_conversion_inv": "Inverse FCR; higher is better efficiency."},
        ),
    }

# ------------------------------
# Column aliases (used by auto-inference helpers)
# ------------------------------
# These expand the names we recognize for each semantic role, including marine terms.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "time": [
        "time", "t", "timestamp", "date", "datetime", "step", "iteration", "epoch",
        "hours", "days", "sample_time", "survey", "cast", "profile", "dive", "trip", "haul", "set", "tow", "batch"
    ],
    "performance": [
        "performance", "repair", "accuracy", "acc", "f1", "reward", "score", "coherence",
        "loss_inv", "bleu", "rouge",
        # bio/marine
        "viability", "function", "yield", "recovery", "signal", "od", "growth", "fitness",
        "survival", "calcification", "recruitment", "photosynthesis", "chlorophyll", "chl",
        "coverage", "abundance", "diversity", "shannon", "richness",
        "cpue", "biomass", "catch_rate", "juvenile_density", "live_coral_cover",
        "oxygen", "fluorescence",
        "growth_rate", "feed_conversion_inv", "health_score", "soh", "capacity_retained",
    ],
    "energy": [
        "energy", "effort", "cost", "compute", "tokens", "gradient_updates", "lr", "batch_tokens",
        # experimental/control inputs & stressors
        "dose", "stressor", "input", "treatment", "drug", "radiation",
        # marine/field drivers
        "par", "light", "temperature", "temp", "pco2", "salinity", "nutrients", "nitrate",
        "phosphate", "silicate", "feed", "aeration_power", "oxygenation", "water_exchange",
        "pump_power", "ship_time", "fuel", "cast_depth", "niskin_trips", "soak_time", "net_hours", "trawl_hours",
        # ops
        "power", "cpu_load", "torque_int", "battery_used",
    ],
    "domain": [
        "domain", "group", "condition", "treatment_group", "species", "site", "station", "reef", "habitat"
    ],
}

# ------------------------------
# File IO
# ------------------------------
def _read_text_table(text: str) -> pd.DataFrame:
    # Guess sep: prefer tab if present and first line lacks commas
    first = text.splitlines()[0] if text.splitlines() else ""
    sep = "\t" if ("\t" in text and "," not in first) else ","
    return pd.read_csv(io.StringIO(text), sep=sep)

def load_table(src) -> pd.DataFrame:
    """
    Read CSV/TSV/XLS/XLSX (always) from a path or a Streamlit UploadedFile/BytesIO.
    Also tries, if libs are available: Parquet, Feather, JSON records, NetCDF via xarray.
    Returns a DataFrame (may be empty).
    """
    # Stream/bytes?
    if hasattr(src, "read") and not isinstance(src, (str, bytes)):
        data = src.read()
        name = getattr(src, "name", "upload")
        lower = name.lower()
        buf = io.BytesIO(data)

        if lower.endswith((".xls", ".xlsx")):
            return pd.read_excel(buf)

        # Optional: Parquet/Feather/JSON
        try:
            if lower.endswith(".parquet"):
                return pd.read_parquet(buf)  # requires pyarrow/fastparquet
            if lower.endswith(".feather"):
                return pd.read_feather(buf)  # requires pyarrow
            if lower.endswith(".json"):
                # Try JSON Lines then fallback to records
                try:
                    return pd.read_json(buf, lines=True)
                except Exception:
                    buf.seek(0)
                    return pd.read_json(buf)
        except Exception:
            pass

        # Optional: NetCDF/Xarray
        try:
            if lower.endswith((".nc", ".netcdf")):
                import xarray as xr  # type: ignore
                ds = xr.open_dataset(buf)
                # Flatten dataset variables into a tidy DataFrame if reasonable
                df = ds.to_dataframe().reset_index()
                return df
        except Exception:
            pass

        # Default: CSV/TSV sniff
        text = data.decode("utf-8", errors="replace")
        return _read_text_table(text)

    # Path-like
    path = str(src)
    lower = path.lower()

    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(path)

    # Optional: Parquet/Feather/JSON
    try:
        if lower.endswith(".parquet"):
            return pd.read_parquet(path)
        if lower.endswith(".feather"):
            return pd.read_feather(path)
        if lower.endswith(".json"):
            try:
                return pd.read_json(path, lines=True)
            except Exception:
                return pd.read_json(path)
        if lower.endswith((".nc", ".netcdf")):
            import xarray as xr  # type: ignore
            ds = xr.open_dataset(path)
            return ds.to_dataframe().reset_index()
    except Exception:
        # If optional formats fail, fall through to CSV/TSV
        pass

    # CSV/TSV default
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snake-case headers, strip whitespace, collapse punctuation.
    """
    def norm(c: str) -> str:
        s = str(c)
        s = s.strip().lower()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "col"
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    return df

# ------------------------------
# Column auto-inference (optional helper for your UI or pipelines)
# ------------------------------
def _pick_from_aliases(df_cols: List[str], candidates: List[str]) -> Optional[str]:
    """
    Return the first matching column name from df_cols that equals ANY alias,
    preferring exact matches; otherwise try 'contains' fuzzy match.
    """
    cols = [c.lower() for c in df_cols]
    # exact
    for a in candidates:
        a = a.lower()
        if a in cols:
            return df_cols[cols.index(a)]
    # contains fuzzy
    for a in candidates:
        a = a.lower()
        for idx, c in enumerate(cols):
            if a in c:
                return df_cols[idx]
    return None

def infer_columns(
    df: pd.DataFrame,
    preset_name: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Try to infer time/performance/energy/domain columns using:
    1) preset hints (if provided),
    2) global COLUMN_ALIASES,
    3) last-resort simple heuristics.
    """
    out = {"time": None, "performance": None, "energy": None, "domain": None}
    cols = list(df.columns)

    # 1) From preset hints
    preset = PRESETS.get(preset_name) if preset_name and preset_name in PRESETS else None
    if preset:
        for key, hint in [
            ("time", getattr(preset, "time", None)),
            ("performance", getattr(preset, "performance", None)),
            ("energy", getattr(preset, "energy", None)),
            ("domain", [getattr(preset, "domain")] if getattr(preset, "domain", None) else None),
        ]:
            if hint:
                pick = _pick_from_aliases(cols, [str(x) for x in hint if x])
                if pick:
                    out[key] = pick

    # 2) Global aliases
    for key in out:
        if out[key] is None:
            pick = _pick_from_aliases(cols, COLUMN_ALIASES.get(key, []))
            if pick:
                out[key] = pick

    # 3) Heuristics if still missing
    if out["performance"] is None:
        # prefer a numeric column that varies
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            out["performance"] = numeric_cols[0]
    if out["energy"] is None:
        # if a column named 'energy' exists numerically, otherwise second numeric
        e = [c for c in cols if "energy" in c.lower()]
        out["energy"] = e[0] if e else (numeric_cols[1] if len(numeric_cols) > 1 else None)
    if out["time"] is None:
        # prefer any column with 'time' or a monotonically non-decreasing numeric
        t = [c for c in cols if "time" in c.lower() or "date" in c.lower() or "epoch" in c.lower()]
        out["time"] = t[0] if t else None
    if out["domain"] is None:
        d = [c for c in cols if c.lower() in COLUMN_ALIASES["domain"]]
        out["domain"] = d[0] if d else None

    return out

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
    return np.array([safe_float(v) for v in series], dtype=float)

# ------------------------------
# RYE core
# ------------------------------
def compute_rye_from_df(
    df: pd.DataFrame,
    repair_col: str,
    energy_col: str,
    time_col: Optional[str] = None,
    clamp_negative_delta: bool = True,
    energy_floor: float = 1e-9,
) -> np.ndarray:
    """
    RYE per step = max(Δperformance, 0)/max(energy, eps)  (default),
    or Δperformance/energy if clamp_negative_delta=False.
    energy_floor protects against zero/negative/NaN denominators.
    """
    perf = _coerce_numeric(df[repair_col])
    energy = _coerce_numeric(df[energy_col])

    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)
    if clamp_negative_delta:
        dperf = np.maximum(dperf, 0.0)

    denom = np.where(np.isfinite(energy) & (energy > 0), energy, energy_floor)
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
    Adds p10/p50/p90 and IQR for richer report text (non-breaking).
    """
    a = np.array(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0,
            "count": 0.0, "std": 0.0, "resilience": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0, "iqr": 0.0,
        }
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a))
    cv = std / (abs(mean) + 1e-9)
    resilience = float(np.clip(1.0 - cv, 0.0, 1.0))
    p10, p50, p90 = np.nanpercentile(a, [10, 50, 90])
    q1, q3 = np.nanpercentile(a, [25, 75])
    return {
        "mean": mean,
        "median": float(np.nanmedian(a)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "count": float(a.size),
        "std": std,
        "resilience": resilience,
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "iqr": float(q3 - q1),
    }

# ------------------------------
# Optional analytics (safe fallbacks)
# ------------------------------
def detect_regimes(
    series: Sequence[float],
    min_len: int = 5,
    gap: float = 0.05
) -> List[Dict[str, Union[int, str]]]:
    """
    Heuristic regime detector: segments where rolling mean stays within a band.
    Returns list of {'start': i0, 'end': i1, 'label': text}.
    """
    x = np.array(series, dtype=float)
    if x.size == 0:
        return []
    roll = rolling_series(x, max(3, min_len))
    regimes: List[Dict[str, Union[int, str]]] = []
    s = 0
    for i in range(1, len(roll)):
        if abs(roll[i] - roll[i - 1]) > gap:
            if (i - 1) - s + 1 >= min_len:
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
    Falls back to numpy if SciPy is unavailable.
    """
    perf = _coerce_numeric(df[perf_col])
    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)

    energy = _coerce_numeric(df[energy_col])
    m = np.isfinite(dperf) & np.isfinite(energy)
    if m.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}

    # Try SciPy first
    try:
        from scipy.stats import pearsonr, spearmanr  # type: ignore
        pr = float(pearsonr(energy[m], dperf[m]).statistic)
        sr = float(spearmanr(energy[m], dperf[m]).statistic)
        return {"pearson": pr, "spearman": sr}
    except Exception:
        pass

    # Fallback using numpy
    try:
        pr = float(np.corrcoef(energy[m], dperf[m])[0, 1])
    except Exception:
        pr = float("nan")
    try:
        # Spearman fallback: rank transform then Pearson
        def _rank(v):
            order = v.argsort(kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(v) + 1, dtype=float)
            return ranks
        r_energy = _rank(energy[m])
        r_dperf = _rank(dperf[m])
        sr = float(np.corrcoef(r_energy, r_dperf)[0, 1])
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

# ------------------------------
# Explicit export surface for the app
# ------------------------------
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
    # helpers
    "infer_columns",
    "COLUMN_ALIASES",
]
