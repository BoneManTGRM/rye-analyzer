import numpy as np
import pandas as pd


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
    """
    Compute RYE directly from a dataframe using column names.
    """
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
