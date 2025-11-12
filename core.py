# core.py
# Core utilities for RYE Analyzer:
# file loading (CSV/TSV/XLS/XLSX + optional Parquet/Feather/JSON/NDJSON/HDF5/Arrow/NetCDF)
# column normalization + auto inference
# numeric coercion
# RYE computation (delta performance divided by energy) with guards
# rolling helpers: SMA and EMA and recommended window size
# summaries with resilience and quantiles
# optional analytics: regimes, correlations, noise floor, bootstrap bands
# extras: cumulative RYE, per domain RYE, outlier flags, simple unit scaling

from __future__ import annotations
import io
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError  # added

# ------------------------------
# PRESETS
# ------------------------------
try:
    from presets import PRESETS  # type: ignore
except Exception:
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
        return list(dict.fromkeys([s.strip() for s in items if s]))

    PRESETS: Dict[str, Preset] = {
        # Note: domain now refers to the COLUMN NAME "domain" so the sidebar default is correct.
        "AI": Preset(
            "AI",
            time=_kw("time", "step", "iteration", "epoch", "t"),
            performance=_kw("performance", "accuracy", "acc", "f1", "reward", "score", "coherence", "loss_inv", "bleu", "rouge"),
            energy=_kw("energy", "tokens", "compute", "cost", "gradient_updates", "lr", "batch_tokens"),
            domain="domain",
            default_rolling=10,
            tooltips={"loss_inv": "1/loss, higher is better"},
        ),
        "Biology": Preset(
            "Biology",
            time=_kw("time", "t", "hours", "days", "samples"),
            performance=_kw("performance", "viability", "function", "yield", "recovery", "signal", "od", "growth", "fitness"),
            energy=_kw("energy", "dose", "stressor", "input", "treatment", "drug", "radiation"),
            domain="domain",
            default_rolling=10,
        ),
        "Robotics": Preset(
            "Robotics",
            time=_kw("time", "t", "cycle", "episode"),
            performance=_kw("performance", "task_success", "score", "stability", "tracking_inv", "uptime", "mean_reward"),
            energy=_kw("energy", "power", "torque_int", "battery_used", "effort", "cpu_load"),
            domain="domain",
            default_rolling=10,
        ),
        "Marine Biology": Preset(
            "Marine Biology",
            time=_kw("time", "t", "date", "day", "doy", "timestamp", "sample_time"),
            performance=_kw(
                "performance", "survival", "growth", "calcification", "recruitment", "photosynthesis",
                "chlorophyll", "chl", "coverage", "abundance", "diversity", "shannon", "richness"
            ),
            energy=_kw(
                "energy", "dose", "nutrients", "nitrate", "phosphate", "silicate", "light", "par",
                "effort", "treatment", "temperature", "temp", "pco2", "salinity", "stress"
            ),
            domain="domain",
            default_rolling=10,
            tooltips={"chl": "Chlorophyll a proxy", "par": "Photosynthetically active radiation"},
        ),
        "Fisheries": Preset(
            "Fisheries",
            time=_kw("time", "t", "date", "trip", "haul", "set", "tow"),
            performance=_kw("performance", "cpue", "yield", "biomass", "catch_rate", "survival", "recruitment"),
            energy=_kw("energy", "effort", "soak_time", "net_hours", "trawl_hours", "fuel", "cost"),
            domain="domain",
            default_rolling=10,
        ),
        "Coral Reef Monitoring": Preset(
            "Coral Reef Monitoring",
            time=_kw("time", "t", "date", "survey", "dive"),
            performance=_kw("performance", "live_coral_cover", "juvenile_density", "calcification", "photosynthesis", "recovery"),
            energy=_kw("energy", "intervention", "outplanting", "nursery_cost", "effort", "par", "dose"),
            domain="domain",
            default_rolling=10,
        ),
        "Oceanography/CTD": Preset(
            "Oceanography/CTD",
            time=_kw("time", "t", "cast", "profile", "date"),
            performance=_kw("performance", "signal", "stability", "coherence", "recovery", "oxygen", "chlorophyll", "fluorescence"),
            energy=_kw("energy", "pump_power", "ship_time", "fuel", "cost", "cast_depth", "niskin_trips"),
            domain="domain",
            default_rolling=10,
        ),
        "Aquaculture": Preset(
            "Aquaculture",
            time=_kw("time", "t", "day", "date", "batch"),
            performance=_kw("performance", "growth_rate", "survival", "feed_conversion_inv", "yield", "biomass", "health_score"),
            energy=_kw("energy", "feed", "aeration_power", "oxygenation", "water_exchange", "temperature", "dose", "cost"),
            domain="domain",
            default_rolling=10,
            tooltips={"feed_conversion_inv": "Inverse FCR, higher is better"},
        ),
    }

# ------------------------------
# Column aliases for inference
# ------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    "time": [
        "time", "t", "timestamp", "date", "datetime", "step", "iteration", "epoch",
        "hours", "days", "sample_time", "survey", "cast", "profile", "dive", "trip", "haul", "set", "tow", "batch"
    ],
    "performance": [
        "performance", "repair", "accuracy", "acc", "f1", "reward", "score", "coherence", "loss_inv", "bleu", "rouge",
        "viability", "function", "yield", "recovery", "signal", "od", "growth", "fitness",
        "survival", "calcification", "recruitment", "photosynthesis", "chlorophyll", "chl",
        "coverage", "abundance", "diversity", "shannon", "richness",
        "cpue", "biomass", "catch_rate", "juvenile_density", "live_coral_cover",
        "oxygen", "fluorescence", "growth_rate", "feed_conversion_inv", "health_score",
        "soh", "capacity_retained"
    ],
    "energy": [
        "energy", "effort", "cost", "compute", "tokens", "gradient_updates", "lr", "batch_tokens",
        "dose", "stressor", "input", "treatment", "drug", "radiation",
        "par", "light", "temperature", "temp", "pco2", "salinity", "nutrients", "nitrate", "phosphate", "silicate",
        "feed", "aeration_power", "oxygenation", "water_exchange",
        "pump_power", "ship_time", "fuel", "cast_depth", "niskin_trips", "soak_time", "net_hours", "trawl_hours",
        "power", "cpu_load", "torque_int", "battery_used"
    ],
    "domain": ["domain", "group", "condition", "treatment_group", "species", "site", "station", "reef", "habitat"],
}

# ------------------------------
# File IO
# ------------------------------
def _read_text_table(text: str) -> pd.DataFrame:
    # added guard for empty uploads
    if not text or not text.strip():
        raise EmptyDataError("Uploaded text file is empty.")
    first = text.splitlines()[0] if text.splitlines() else ""
    sep = "\t" if ("\t" in text and "," not in first) else ","
    return pd.read_csv(io.StringIO(text), sep=sep)

def load_table(src) -> pd.DataFrame:
    """
    Read CSV/TSV/XLS/XLSX. If the filename suggests another format, try it too.
    Falls back to CSV or TSV sniffing on failure.
    """
    # Uploaded file like Streamlit's UploadedFile
    if hasattr(src, "read") and not isinstance(src, (str, bytes)):
        data = src.read()
        name = getattr(src, "name", "upload")
        lower = name.lower()
        buf = io.BytesIO(data)

        if lower.endswith((".xls", ".xlsx")):
            return pd.read_excel(buf)

        try:
            if lower.endswith(".parquet"):
                return pd.read_parquet(buf)
            if lower.endswith(".feather"):
                return pd.read_feather(buf)
            if lower.endswith(".arrow"):
                import pyarrow.ipc as ipc  # type: ignore
                reader = ipc.RecordBatchFileReader(buf)
                table = reader.read_all()
                return table.to_pandas()  # type: ignore
            if lower.endswith((".h5", ".hdf5")):
                return pd.read_hdf(buf)
            if lower.endswith(".json"):
                try:
                    return pd.read_json(buf, lines=True)
                except Exception:
                    buf.seek(0)
                    return pd.read_json(buf)
            if lower.endswith((".nc", ".netcdf")):
                import xarray as xr  # type: ignore
                ds = xr.open_dataset(buf)
                return ds.to_dataframe().reset_index()
        except Exception:
            pass

        text = data.decode("utf-8", errors="replace")
        return _read_text_table(text)

    # Path like
    path = str(src)
    lower = path.lower()

    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(path)

    try:
        if lower.endswith(".parquet"):
            return pd.read_parquet(path)
        if lower.endswith(".feather"):
            return pd.read_feather(path)
        if lower.endswith(".arrow"):
            import pyarrow.ipc as ipc  # type: ignore
            with open(path, "rb") as f:
                reader = ipc.RecordBatchFileReader(f)
                table = reader.read_all()
                return table.to_pandas()  # type: ignore
        if lower.endswith((".h5", ".hdf5")):
            return pd.read_hdf(path)
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
        pass

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")

# ------------------------------
# Column normalization and inference
# ------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c: str) -> str:
        s = str(c).strip().lower()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "col"
    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out

def _pick_from_aliases(df_cols: List[str], candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in df_cols]
    for a in candidates:
        a = a.lower()
        if a in cols:
            return df_cols[cols.index(a)]
    for a in candidates:
        a = a.lower()
        for i, c in enumerate(cols):
            if a in c:
                return df_cols[i]
    return None

def infer_columns(df: pd.DataFrame, preset_name: Optional[str] = None) -> Dict[str, Optional[str]]:
    out = {"time": None, "performance": None, "energy": None, "domain": None}
    cols = list(df.columns)

    preset = PRESETS.get(preset_name) if preset_name and preset_name in PRESETS else None
    if preset:
        hints = {
            "time": getattr(preset, "time", []),
            "performance": getattr(preset, "performance", []),
            "energy": getattr(preset, "energy", []),
            "domain": [getattr(preset, "domain")] if getattr(preset, "domain", None) else [],
        }
        for k, cand in hints.items():
            if cand:
                pick = _pick_from_aliases(cols, [str(x) for x in cand if x])
                if pick:
                    out[k] = pick

    for key in out:
        if out[key] is None:
            pick = _pick_from_aliases(cols, COLUMN_ALIASES.get(key, []))
            if pick:
                out[key] = pick

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if out["performance"] is None and numeric_cols:
        out["performance"] = numeric_cols[0]
    if out["energy"] is None:
        e = [c for c in cols if "energy" in c.lower()]
        out["energy"] = e[0] if e else (numeric_cols[1] if len(numeric_cols) > 1 else None)
    if out["time"] is None:
        t = [c for c in cols if any(k in c.lower() for k in ("time", "date", "epoch", "step", "iteration"))]
        out["time"] = t[0] if t else None
    if out["domain"] is None:
        d = [c for c in cols if c.lower() in COLUMN_ALIASES["domain"]]
        out["domain"] = d[0] if d else None

    return out

# ------------------------------
# Numerics and helpers
# ------------------------------
def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return float("nan")

def _coerce_numeric(series: Iterable) -> np.ndarray:
    return np.array([safe_float(v) for v in series], dtype=float)

def scale_units(arr: Sequence[float], factor: float) -> np.ndarray:
    a = np.array(arr, dtype=float)
    return a * float(factor)

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
    RYE per step = max(delta performance, 0) divided by max(energy, energy_floor)
    Set clamp_negative_delta=False to allow negative deltas.
    """
    perf = _coerce_numeric(df[repair_col])
    energy = _coerce_numeric(df[energy_col])

    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)
    if clamp_negative_delta:
        dperf = np.maximum(dperf, 0.0)

    denom = np.where(np.isfinite(energy) & (energy > 0.0), energy, energy_floor)
    rye = dperf / denom
    rye = np.where(np.isfinite(rye), rye, 0.0)
    return rye

def compute_rye_cumulative(rye_series: Sequence[float]) -> np.ndarray:
    a = np.asarray(rye_series, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return np.cumsum(a)

# ------------------------------
# Rolling helpers
# ------------------------------
def rolling_series(series: Sequence[float], window: int) -> np.ndarray:
    s = pd.Series(series, dtype=float)
    if window <= 1:
        return s.values
    return s.rolling(window=window, min_periods=1).mean().values

def ema_series(series: Sequence[float], span: int) -> np.ndarray:
    if span is None or span <= 1:
        return np.asarray(series, dtype=float)
    s = pd.Series(series, dtype=float)
    return s.ewm(span=span, adjust=False).mean().values

def recommend_window(n_rows: int, preset_default: Optional[int]) -> int:
    if preset_default and preset_default > 0:
        return int(preset_default)
    if n_rows <= 0:
        return 10
    guess = max(3, min(200, int(round(max(3, n_rows * 0.05)))))
    return guess

# ------------------------------
# Summaries
# ------------------------------
def summarize_series(series: Sequence[float], with_shape: bool = False) -> Dict[str, float]:
    a = np.array(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        base = {
            "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0,
            "count": 0.0, "std": 0.0, "resilience": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0, "iqr": 0.0,
        }
        if with_shape:
            base.update({"skew": 0.0, "kurtosis": 0.0})
        return base
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a))
    cv = std / (abs(mean) + 1e-9)
    resilience = float(np.clip(1.0 - cv, 0.0, 1.0))
    p10, p50, p90 = np.nanpercentile(a, [10, 50, 90])
    q1, q3 = np.nanpercentile(a, [25, 75])
    out = {
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
    if with_shape:
        try:
            from scipy.stats import skew, kurtosis  # type: ignore
            out["skew"] = float(skew(a, nan_policy="omit"))
            out["kurtosis"] = float(kurtosis(a, nan_policy="omit"))
        except Exception:
            out["skew"] = float("nan")
            out["kurtosis"] = float("nan")
    return out

def summarize_by_domain(
    df: pd.DataFrame,
    domain_col: str,
    repair_col: str,
    energy_col: str,
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute per domain RYE mean and resilience. If window is given, also include rolling mean.
    """
    rows = []
    for dom, sub in df.groupby(domain_col):
        rye = compute_rye_from_df(sub, repair_col=repair_col, energy_col=energy_col)
        rec = {"domain": dom}
        rec.update({f"rye_{k}": v for k, v in summarize_series(rye).items()})
        if window and window > 1:
            rroll = rolling_series(rye, window)
            rec.update({f"rye_roll_{k}": v for k, v in summarize_series(rroll).items()})
        rows.append(rec)
    return pd.DataFrame(rows)

# ------------------------------
# Optional analytics
# ------------------------------
def detect_regimes(series: Sequence[float], min_len: int = 5, gap: float = 0.05) -> List[Dict[str, Union[int, str]]]:
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
    perf = _coerce_numeric(df[perf_col])
    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)

    energy = _coerce_numeric(df[energy_col])
    m = np.isfinite(dperf) & np.isfinite(energy)
    if m.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}

    try:
        from scipy.stats import pearsonr, spearmanr  # type: ignore
        pr = float(pearsonr(energy[m], dperf[m]).statistic)
        sr = float(spearmanr(energy[m], dperf[m]).statistic)
        return {"pearson": pr, "spearman": sr}
    except Exception:
        pass

    try:
        pr = float(np.corrcoef(energy[m], dperf[m])[0, 1])
    except Exception:
        pr = float("nan")
    try:
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
    x = np.array(series, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if x.size == 0:
        return {"low": [], "mid": [], "high": []}

    rolls = []
    rng = np.random.default_rng(12345)
    n = len(x)
    for _ in range(max(10, n_boot)):
        idx = rng.integers(0, n, size=n)
        rs = rolling_series(x[idx], max(1, window))
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
# Outliers
# ------------------------------
def flag_outliers(series: Sequence[float], z: float = 3.0) -> np.ndarray:
    a = np.asarray(series, dtype=float)
    mu = np.nanmean(a)
    sd = np.nanstd(a) + 1e-12
    zscores = (a - mu) / sd
    return np.where(np.abs(zscores) >= z, 1, 0)

# ------------------------------
# Explicit export surface
# ------------------------------
__all__ = [
    "load_table",
    "normalize_columns",
    "infer_columns",
    "COLUMN_ALIASES",
    "safe_float",
    "compute_rye_from_df",
    "compute_rye_cumulative",
    "rolling_series",
    "ema_series",
    "recommend_window",
    "summarize_series",
    "summarize_by_domain",
    "detect_regimes",
    "energy_delta_performance_correlation",
    "estimate_noise_floor",
    "bootstrap_rolling_mean",
    "flag_outliers",
    "scale_units",
    "PRESETS",
]
