# app_streamlit.py
from __future__ import annotations

import io
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from presets import PRESETS, Preset  # <- uses YOUR current presets.py exactly
import core
import report as report_mod

st.set_page_config(page_title="RYE Analyzer", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def _find_first_present(df_cols, candidates):
    """Return the first candidate that exists in df columns (case-insensitive)."""
    if not candidates:
        return None
    cols_lower = {c.lower(): c for c in df_cols}
    for c in candidates:
        k = c.lower()
        if k in cols_lower:
            return cols_lower[k]
    return None

def _guess_columns(df: pd.DataFrame, preset: Preset) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, float]]:
    """Guess time / performance / energy columns using preset keyword lists."""
    scores = {}
    def score_list(candidates):
        s = 0.0
        for c in candidates:
            if c.lower() in (x.lower() for x in df.columns):
                s += 1.0
        return s

    t_col = _find_first_present(df.columns, preset.time)
    p_col = _find_first_present(df.columns, preset.performance)
    e_col = _find_first_present(df.columns, preset.energy)

    scores["time"] = score_list(preset.time)
    scores["performance"] = score_list(preset.performance)
    scores["energy"] = score_list(preset.energy)

    return t_col, p_col, e_col, scores

def _read_any(upload):
    """Read CSV/TSV by extension, otherwise try csv."""
    if upload is None:
        return None
    name = upload.name.lower()
    data = upload.read()
    if name.endswith(".tsv"):
        return pd.read_csv(io.BytesIO(data), sep="\t")
    return pd.read_csv(io.BytesIO(data))

def _nan_safe(a):
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return a

# ----------------------------
# UI — Header
# ----------------------------
st.title("RYE Analyzer")
st.caption("Compute Repair Yield per Energy from any time series.")

# Sidebar: preset & inputs
with st.sidebar:
    st.subheader("Inputs")

    # Choose preset from YOUR dict (no default constant assumed)
    preset_name = st.selectbox(
        "Preset",
        options=sorted(PRESETS.keys()),
        index=sorted(PRESETS.keys()).index("Biology") if "Biology" in PRESETS else 0,
    )
    preset: Preset = PRESETS[preset_name]

    primary = st.file_uploader("Primary file", type=["csv", "tsv"])
    comparison = st.file_uploader("Comparison file (optional)", type=["csv", "tsv"])

    # DOI / link for the report
    dataset_link = st.text_input("Zenodo DOI or dataset link (optional)", value="", help="Example: 10.5281/zenodo.12345 or https://zenodo.org/record/...")

# Load data
df = _read_any(primary)
df2 = _read_any(comparison) if comparison else None

if df is None or df.empty:
    st.info("Upload a CSV/TSV to begin.")
    st.stop()

# Guess columns via the chosen preset
g_time, g_perf, g_energy, scores = _guess_columns(df, preset)

# Column selectors (pre-filled by guesses)
st.subheader("Select columns")
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    time_col = st.selectbox("Time/index", options=list(df.columns), index=list(df.columns).index(g_time) if g_time in df.columns else 0)

with col2:
    perf_col = st.selectbox("Performance (improves when higher)", options=list(df.columns), index=list(df.columns).index(g_perf) if g_perf in df.columns else 0)

with col3:
    energy_col = st.selectbox("Energy/effort/cost", options=list(df.columns), index=list(df.columns).index(g_energy) if g_energy in df.columns else 0)

with col4:
    roll_default = getattr(preset, "default_rolling", 10) or 10
    roll_win = st.number_input("Rolling window", min_value=1, max_value=500, value=int(roll_default), step=1, help="Smoothing for the rolling mean of RYE")

st.caption(f"Preset matcher scores — time: {scores['time']:.0f}, performance: {scores['performance']:.0f}, energy: {scores['energy']:.0f}")

# Map columns and compute RYE
mapped = core.map_columns(df, time_col, perf_col, energy_col)
perf = _nan_safe(mapped["performance"].to_numpy())
energy = _nan_safe(mapped["energy"].to_numpy())
rye = core.compute_rye(perf, energy)
rye_roll = core.rolling_mean(rye, roll_win)

# Plots
st.subheader("RYE results")
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
ax1.plot(rye, label="RYE", linewidth=1.6)
ax1.plot(rye_roll, label="RYE rolling", linewidth=1.6)
ax1.set_xlabel("Index"); ax1.set_ylabel("RYE"); ax1.legend(); ax1.grid(True, alpha=0.3)
st.pyplot(fig1, clear_figure=True)

# Optional comparison
if df2 is not None and not df2.empty:
    st.markdown("**Comparison dataset**")
    g2_time, g2_perf, g2_energy, _ = _guess_columns(df2, preset)
    c1, c2, c3 = st.columns(3)
    with c1:
        time2 = st.selectbox("Time (B)", options=list(df2.columns), index=list(df2.columns).index(g2_time) if g2_time in df2.columns else 0, key="t2")
    with c2:
        perf2 = st.selectbox("Performance (B)", options=list(df2.columns), index=list(df2.columns).index(g2_perf) if g2_perf in df2.columns else 0, key="p2")
    with c3:
        energy2 = st.selectbox("Energy (B)", options=list(df2.columns), index=list(df2.columns).index(g2_energy) if g2_energy in df2.columns else 0, key="e2")

    mapped2 = core.map_columns(df2, time2, perf2, energy2)
    rye2 = core.compute_rye(mapped2["performance"].to_numpy(), mapped2["energy"].to_numpy())
    rye2_roll = core.rolling_mean(rye2, roll_win)

    fig2, ax2 = plt.subplots()
    ax2.plot(rye_roll, label=f"{preset_name} A", linewidth=1.6)
    ax2.plot(rye2_roll, label=f"{preset_name} B", linewidth=1.6)
    ax2.set_xlabel("Index"); ax2.set_ylabel("RYE (rolling)"); ax2.legend(); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

# Summary block
summary = core.summarize(rye, rye, roll_win)
with st.expander("Summary statistics", expanded=True):
    st.json(summary)

# Report generation
st.subheader("Reports")
interp = st.text_area(
    "Interpretation (optional)",
    value=(
        f"Average efficiency (RYE mean) is {summary['mean']:.3f}. "
        f"Resilience index is {summary['resilience']:.3f}. "
        "Interpret spikes or dips in the RYE curve, map to events or interventions, and iterate TGRM loops to raise average RYE."
    ),
)

if st.button("Generate PDF report"):
    meta = {
        "dataset_link": dataset_link.strip(),
        "columns": [str(time_col), str(perf_col), str(energy_col)],
        "domain": preset.domain or "",
        "sample_n": 100,
    }
    plot_series = {"RYE": rye, "RYE rolling": rye_roll}
    pdf_bytes = report_mod.build_pdf(
        rye=rye,
        summary=summary,
        metadata=meta,
        plot_series=plot_series,
        interpretation=interp,
        logo_path=None,  # supply a path if you add a logo file
    )
    st.download_button("Download PDF", data=pdf_bytes, file_name="rye_report.pdf", mime="application/pdf")

# JSON export of analysis config (nice to keep preset + chosen columns)
export = {
    "preset": preset_name,
    "columns": {"time": time_col, "performance": perf_col, "energy": energy_col},
    "rolling_window": int(roll_win),
    "summary": summary,
    "dataset_link": dataset_link.strip(),
}
st.download_button("Download JSON summary", data=json.dumps(export, indent=2).encode("utf-8"), file_name="rye_summary.json", mime="application/json")
