# core.py
# Core utilities for RYE Analyzer:
# - file loading (CSV/TSV/XLS/XLSX + optional Parquet/Feather/JSON/NDJSON/HDF5/Arrow/NetCDF)
# - column normalization + robust auto-inference (aliases + preset hints)
# - robust numeric coercion
# - RYE computation (Δperformance / energy) with guards
# - rolling helpers: SMA + EMA + recommended window size
# - summaries (incl. resilience, quantiles, IQR, optional skew/kurtosis)
# - optional analytics: regimes, correlation (with SciPy fallback), noise floor, bootstrap bands
# - extras: cumulative RYE, per-domain RYE, outlier flags, simple unit scaling
# Exposes:
#   load_table, normalize_columns, infer_columns, COLUMN_ALIASES,
#   safe_float, compute_rye_from_df, compute_rye_cumulative,
#   rolling_series, ema_series, recommend_window,
#   summarize_series, summarize_by_domain,
#   detect_regimes, energy_delta_performance_correlation,
#   estimate_noise_floor, bootstrap_rolling_mean,
#   flag_outliers, scale_units, PRESETS

from __future__ import annotations
import io
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# ------------------------------
# PRESETS (import your long list if present)
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
        # AI
        "AI": Preset(
            "AI",
            time=_kw("step","iteration","epoch","t","time"),
            performance=_kw("accuracy","acc","f1","reward","score","coherence","loss_inv","bleu","rouge","performance"),
            energy=_kw("tokens","compute","energy","cost","gradient_updates","lr","batch_tokens"),
            domain="ai",
            default_rolling=10,
            tooltips={"loss_inv":"1/loss (higher=better)"}
        ),
        # Biology (wet lab)
        "Biology": Preset(
            "Biology",
            time=_kw("time","t","hours","days","samples"),
            performance=_kw("viability","function","yield","recovery","signal","od","growth","fitness","performance"),
            energy=_kw("dose","stressor","input","energy","treatment","drug","radiation"),
            domain="bio",
            default_rolling=10,
        ),
        # Robotics
        "Robotics": Preset(
            "Robotics",
            time=_kw("t","time","cycle","episode"),
            performance=_kw("task_success","score","stability","tracking_inv","uptime","mean_reward","performance"),
            energy=_kw("power","torque_int","battery_used","energy","effort","cpu_load"),
            domain="robot",
            default_rolling=10,
        ),
        # Marine-friendly presets
        "Marine Biology": Preset(
            "Marine Biology",
            time=_kw("time","t","date","day","doy","timestamp","sample_time"),
            performance=_kw("survival","growth","calcification","recruitment","photosynthesis",
                            "chlorophyll","chl","coverage","abundance","diversity","shannon","richness"),
            energy=_kw("dose","nutrients","nitrate","phosphate","silicate","light","par","effort",
                       "treatment","temperature","temp","pco2","salinity","stress"),
            domain="marine",
            default_rolling=10,
            tooltips={"chl":"Chlorophyll-a proxy","par":"Photosynthetically Active Radiation"}
        ),
        "Fisheries": Preset(
            "Fisheries",
            time=_kw("time","t","date","trip","haul","set","tow"),
            performance=_kw("cpue","yield","biomass","catch_rate","survival","recruitment"),
            energy=_kw("effort","soak_time","net_hours","trawl_hours","fuel","cost"),
            domain="fisheries",
            default_rolling=10,
        ),
        "Coral Reef Monitoring": Preset(
            "Coral Reef Monitoring",
            time=_kw("time","t","date","survey","dive"),
            performance=_kw("live_coral_cover","juvenile_density","calcification","photosynthesis","recovery"),
            energy=_kw("intervention","outplanting","nursery_cost","effort","par","dose"),
            domain="reef",
            default_rolling=10,
        ),
        "Oceanography/CTD": Preset(
            "Oceanography/CTD",
            time=_kw("time","t","cast","profile","date"),
            performance=_kw("signal","stability","coherence","recovery","oxygen","chlorophyll","fluorescence"),
            energy=_kw("pump_power","ship_time","fuel","cost","cast_depth","niskin_trips"),
            domain="ctd",
            default_rolling=10,
        ),
        "Aquaculture": Preset(
            "Aquaculture",
            time=_kw("time","t","day","date","batch"),
            performance=_kw("growth_rate","survival","feed_conversion_inv","yield","biomass","health_score"),
            energy=_kw("feed","aeration_power","oxygenation","water_exchange","temperature","dose","cost"),
            domain="aq",
            default_rolling=10,
            tooltips={"feed_conversion_inv":"Inverse FCR; higher=better"}
        ),
    }

# ------------------------------
# Column aliases (auto-inference)
# ------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    "time": [
        "time","t","timestamp","date","datetime","step","iteration","epoch",
        "hours","days","sample_time","survey","cast","profile","dive","trip","haul","set","tow","batch"
    ],
    "performance": [
        "performance","repair","accuracy","acc","f1","reward","score","coherence","loss_inv","bleu","rouge",
        "viability","function","yield","recovery","signal","od","growth","fitness",
        "survival","calcification","recruitment","photosynthesis","chlorophyll","chl",
        "coverage","abundance","diversity","shannon","richness",
        "cpue","biomass","catch_rate","juvenile_density","live_coral_cover",
        "oxygen","fluorescence","growth_rate","feed_conversion_inv","health_score",
        "soh","capacity_retained"
    ],
    "energy": [
        "energy","effort","cost","compute","tokens","gradient_updates","lr","batch_tokens",
        "dose","stressor","input","treatment","drug","radiation",
        "par","light","temperature","temp","pco2","salinity","nutrients","nitrate","phosphate","silicate",
        "feed","aeration_power","oxygenation","water_exchange",
        "pump_power","ship_time","fuel","cast_depth","niskin_trips","soak_time","net_hours","trawl_hours",
        "power","cpu_load","torque_int","battery_used"
    ],
    "domain": ["domain","group","condition","treatment_group","species","site","station","reef","habitat"],
}

# ------------------------------
# File IO
# ------------------------------
def _read_text_table(text: str) -> pd.DataFrame:
    first = text.splitlines()[0] if text.splitlines() else ""
    sep = "\t" if ("\t" in text and "," not in first) else ","
    return pd.read_csv(io.StringIO(text), sep=sep)

def load_table(src) -> pd.DataFrame:
    """
    Read CSV/TSV/XLS/XLSX from path/UploadedFile.
    Optionally tries Parquet/Feather/JSON/NDJSON/HDF5/Arrow/NetCDF when the extension matches.
    Gracefully falls back to CSV/TSV sniff if necessary.
    """
    # Stream/bytes
    if hasattr(src, "read") and not isinstance(src, (str, bytes)):
        data = src.read()
        name = getattr(src, "name", "upload")
        lower = name.lower()
        buf = io.BytesIO(data)

        if lower.endswith((".xls",".xlsx")):
            return pd.read_excel(buf)

        try:
            if lower.endswith(".parquet"):
                return pd.read_parquet(buf)          # pyarrow/fastparquet
            if lower.endswith(".feather"):
                return pd.read_feather(buf)          # pyarrow
            if lower.endswith(".arrow"):
                import pyarrow as pa, pyarrow.ipc as ipc  # type: ignore
                reader = ipc.RecordBatchFileReader(buf)
                table = reader.read_all()
                return table.to_pandas()  # type: ignore
            if lower.endswith((".h5",".hdf5")):
                return pd.read_hdf(buf)              # pytables
            if lower.endswith(".json"):
                try:
                    return pd.read_json(buf, lines=True)  # NDJSON
                except Exception:
                    buf.seek(0)
                    return pd.read_json(buf)              # records/array
            if lower.endswith((".nc",".netcdf")):
                import xarray as xr  # type: ignore
                ds = xr.open_dataset(buf)
                return ds.to_dataframe().reset_index()
        except Exception:
            pass

        text = data.decode("utf-8", errors="replace")
        return _read_text_table(text)

    # Path-like
    path = str(src)
    lower = path.lower()

    if lower.endswith((".xls",".xlsx")):
        return pd.read_excel(path)

    try:
        if lower.endswith(".parquet"):
            return pd.read_parquet(path)
        if lower.endswith(".feather"):
            return pd.read_feather(path)
        if lower.endswith(".arrow"):
            import pyarrow as pa, pyarrow.ipc as ipc  # type: ignore
            with open(path, "rb") as f:
                reader = ipc.RecordBatchFileReader(f)
                table = reader.read_all()
                return table.to_pandas()  # type: ignore
        if lower.endswith((".h5",".hdf5")):
            return pd.read_hdf(path)
        if lower.endswith(".json"):
            try:
                return pd.read_json(path, lines=True)
            except Exception:
                return pd.read_json(path)
        if lower.endswith((".nc",".netcdf")):
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
# Column normalization & inference
# ------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c: str) -> str:
        s = str(c).strip().lower()
        s = re.sub(r"[^\w]+","_", s)
        s = re.sub(r"_+","_", s).strip("_")
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
    # fuzzy contains
    for a in candidates:
        a = a.lower()
        for i, c in enumerate(cols):
            if a in c:
                return df_cols[i]
    return None

def infer_columns(df: pd.DataFrame, preset_name: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Resolve time/performance/energy/domain columns:
      1) preset hints (if provided)
      2) global aliases
      3) simple heuristics (numeric preference)
    """
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

    # heuristics
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if out["performance"] is None and numeric_cols:
        out["performance"] = numeric_cols[0]
    if out["energy"] is None:
        e = [c for c in cols if "energy" in c.lower()]
        out["energy"] = e[0] if e else (numeric_cols[1] if len(numeric_cols) > 1 else None)
    if out["time"] is None:
        t = [c for c in cols if any(k in c.lower() for k in ("time","date","epoch","step","iteration"))]
        out["time"] = t[0] if t else None
    if out["domain"] is None:
        d = [c for c in cols if c.lower() in COLUMN_ALIASES["domain"]]
        out["domain"] = d[0] if d else None

    return out

# ------------------------------
# Numerics & helpers
# ------------------------------
def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float,int,np.floating,np.integer)):
            return float(x)
        s = str(x).strip().replace(",","")
        return float(s)
    except Exception:
        return float("nan")

def _coerce_numeric(series: Iterable) -> np.ndarray:
    return np.array([safe_float(v) for v in series], dtype=float)

def scale_units(arr: Sequence[float], factor: float) -> np.ndarray:
    """Multiply a numeric sequence by a constant factor (e.g., mW→W)."""
    a = np.array(arr, dtype=float)
    return a * float(factor)

# ------------------------------
# RYE core
# ------------------------------
def compute_rye_from_df(
   
