# app_streamlit.py
import io
import math
import pandas as pd
import numpy as np
import streamlit as st

from presets import PRESETS, get_preset
from report import build_pdf

st.set_page_config(page_title="RYE Analyzer", page_icon="🧠", layout="wide")

st.title("RYE Analyzer")
st.caption("Compute Repair Yield per Energy from any time series.")

# -----------------------
# Sidebar: inputs
# -----------------------
with st.sidebar:
    st.header("Inputs")

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1)  # Biology first
    preset = get_preset(preset_name)

    rolling_window = st.slider(
        "Rolling window",
        min_value=preset.min_window,
        max_value=preset.max_window,
        value=preset.default_window,
        help="Size of the moving window used for the rolling RYE curve."
    )

    dataset_link = st.text_input("Zenodo DOI or dataset link (optional)", placeholder="10.5281/zenodo.xxxxx or https://...")

    st.markdown("---")
    st.write("**Upload CSV** (needs at least `performance` and `energy` columns; optional `time`, `domain`).")
    file = st.file_uploader("Primary file", type=["csv", "tsv"], accept_multiple_files=False, label_visibility="collapsed")

# -----------------------
# Helpers
# -----------------------
def _read_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    sep = "\t" if name.endswith(".tsv") else ","
    return pd.read_csv(file, sep=sep)

def compute_rye(df: pd.DataFrame) -> pd.Series:
    """
    RYE = ΔR / E.
    Here we take performance as R, compute first difference (improvement),
    and divide by energy per step. Negative allowed.
    """
    if "performance" not in df.columns or "energy" not in df.columns:
        raise ValueError("CSV must include 'performance' and 'energy' columns.")
    perf = pd.to_numeric(df["performance"], errors="coerce").fillna(method="ffill").fillna(0.0)
    energy = pd.to_numeric(df["energy"], errors="coerce").fillna(0.0)
    dR = perf.diff().fillna(0.0)
    # Avoid div-by-zero — treat 0 energy as eps
    energy_safe = energy.replace(0, np.finfo(float).eps)
    rye = dR / energy_safe
    return rye.astype(float)

def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    if window > len(series):
        window = len(series)
    return series.rolling(window=window, min_periods=1).mean()

def resilience_index(series: pd.Series) -> float:
    # Simple stability proxy: 1 - coefficient of variation over positive values (clamped)
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return 0.0
    m = float(np.mean(np.abs(s)))
    sd = float(np.std(s))
    if m <= 1e-12:
        return 0.0
    r = 1.0 - (sd / (m + 1e-12))
    return float(max(0.0, min(1.0, r)))

# -----------------------
# Main
# -----------------------
if not file:
    st.info("Upload a CSV to begin.")
    st.stop()

df = _read_csv(file)
try:
    rye = compute_rye(df)
except Exception as e:
    st.error(str(e))
    st.stop()

rye_rolling = rolling_mean(rye, rolling_window)

# Summary
summary = {
    "mean": float(np.mean(rye)),
    "median": float(np.median(rye)),
    "max": float(np.max(rye)),
    "min": float(np.min(rye)),
    "count": int(len(rye)),
    "resilience": float(resilience_index(rye)),
}

# Metadata
cols = list(df.columns)
metadata = {
    "rows": int(len(df)),
    "preset": preset.name,
    "columns": cols,
    "sample_n": 100,
    "dataset_link": dataset_link.strip(),
}

# Layout
left, right = st.columns([1, 1])
with left:
    st.subheader("Summary")
    st.json(summary)
with right:
    st.subheader("Plot")
    st.line_chart(
        pd.DataFrame({
            preset.series_label: rye.values,
            preset.rolling_label: rye_rolling.values
        })
    )

st.subheader("Reports")
interp = (
    f"Average efficiency (RYE mean) is {summary['mean']:.3f}. "
    "Values vary by domain and data preparation. "
    f"Resilience index is {summary['resilience']:.3f} — higher means steadier efficiency under fluctuation. "
    "Use the rolling curve to spot short-term noise and trend. "
    + preset.interpretation_hint
).strip()

if st.button("Generate PDF report", type="primary"):
    pdf_bytes = build_pdf(
        rye=list(rye.values),
        summary=summary,
        metadata=metadata,
        plot_series={
            preset.series_label: list(rye.values),
            preset.rolling_label: list(rye_rolling.values),
        },
        interpretation=interp,
        logo_path=None,
        plot_title_override=preset.plot_title,
    )
    st.download_button(
        "Download report PDF",
        data=pdf_bytes,
        file_name="rye_report.pdf",
        mime="application/pdf",
    )
