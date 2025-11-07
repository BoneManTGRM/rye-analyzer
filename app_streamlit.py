# app_streamlit.py — RYE Analyzer (full upgraded build)

from __future__ import annotations
import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Local modules (same folder)
from core import (
    compute_rye_from_df,
    rolling_series,
    safe_float,
    summarize_series,
)
from report import build_pdf  # returns bytes

# ---------------- UI helpers ----------------
def section(title: str):
    st.subheader(title)

def note(msg: str):
    st.caption(msg)

# ---------------- Page config ----------------
st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
st.title("RYE Analyzer")
st.caption("Compute Repair Yield per Energy from any time series.")

with st.expander("What is RYE"):
    st.write(
        "Repair Yield per Energy (RYE) measures how efficiently a system converts effort or energy into "
        "successful repair or performance gains. Higher RYE means better efficiency. "
        "Upload a CSV to compute RYE and explore improvements, rolling windows, comparisons, and reports."
    )

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    st.write("Upload one CSV to analyze. Optionally upload a second CSV to compare.")
    file1 = st.file_uploader("Primary CSV", type=["csv"], key="csv1")
    file2 = st.file_uploader("Comparison CSV (optional)", type=["csv"], key="csv2")

    st.divider()
    st.write("Column names in your CSV")
    col_repair = st.text_input("Repair column", value="performance")
    col_energy = st.text_input("Energy column", value="energy")
    col_time = st.text_input("Time column (optional)", value="time")
    col_domain = st.text_input("Domain column (optional)", value="domain")

    st.divider()
    window = st.number_input("Rolling window", min_value=1, max_value=500, value=10, step=1)

    st.divider()
    st.write("Δ Energy simulator")
    sim_factor = st.slider("Multiply energy by", min_value=0.10, max_value=3.0, value=1.0, step=0.05)

    st.divider()
    st.write("No CSV yet?")
    if st.button("Download example CSV"):
        example = pd.DataFrame({
            "time": np.arange(0, 15),
            "domain": ["AI"] * 5 + ["Bio"] * 5 + ["Robotics"] * 5,
            "performance": [0, 0, 0.1, 0.2, 0.35, 0.38, 0.5, 0.46, 0.52, 0.6, 0.6, 0.6, 0.62, 0.62, 0.65],
            "energy":      [1, 1, 1, 1,    1,    1.1, 1.0, 1.02, 1.05, 1.1, 1.1, 1.12, 1.09, 1.1, 1.1],
        })
        b = example.to_csv(index=False).encode("utf-8")
        st.download_button("Save example.csv", b, file_name="example.csv", mime="text/csv")

# ---------------- Core workers ----------------
def load_csv(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Could not read CSV. {e}")
        return None

def ensure_columns(df: pd.DataFrame, repair: str, energy: str) -> bool:
    miss = [c for c in [repair, energy] if c not in df.columns]
    if miss:
        st.error(f"Missing columns: {', '.join(miss)}")
        st.write("Found columns:", list(df.columns))
        return False
    return True

def compute_block(df: pd.DataFrame, label: str, sim_mult: float) -> dict:
    df_sim = df.copy()
    if col_energy in df_sim.columns:
        df_sim[col_energy] = df_sim[col_energy].apply(lambda x: safe_float(x) * sim_mult)

    rye = compute_rye_from_df(df_sim, repair_col=col_repair, energy_col=col_energy)
    rye_roll = rolling_series(rye, window)

    summary = summarize_series(rye)
    summary_roll = summarize_series(rye_roll)

    return {
        "label": label,
        "df": df_sim,
        "rye": rye,
        "rye_roll": rye_roll,
        "summary": summary,
        "summary_roll": summary_roll,
    }

# ---------------- Main UI ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Single analysis", "Compare datasets", "Multi domain", "Reports"])

# Load files
df1 = load_csv(file1)
df2 = load_csv(file2)

# ---------- Tab 1: Single analysis ----------
with tab1:
    if df1 is None:
        st.info("Upload a CSV in the sidebar to begin.")
    else:
        if ensure_columns(df1, col_repair, col_energy):
            block = compute_block(df1, "primary", sim_factor)
            rye = block["rye"]
            rye_roll = block["rye_roll"]
            summary = block["summary"]

            # Scorecard
            st.metric("Mean RYE", f"{summary['mean']:.4f}", help="Average RYE across rows")

            # Column preview
            with st.expander("Columns detected"):
                st.json(list(df1.columns))

            # Charts
            st.plotly_chart(px.line(rye, title="RYE"), use_container_width=True)
            st.plotly_chart(px.line(rye_roll, title=f"RYE (rolling window = {window})"), use_container_width=True)

            # Summary JSON
            section("Summary")
            st.code(json.dumps(summary, indent=2))

            # Downloads
            csv_bytes = pd.Series(rye, name="RYE").to_csv(index_label="index").encode("utf-8")
            st.download_button("Download RYE CSV", csv_bytes, file_name="rye.csv", mime="text/csv")

            json_bytes = io.BytesIO(json.dumps(summary, indent=2).encode("utf-8"))
            st.download_button("Download summary JSON", json_bytes.getvalue(), file_name="summary.json", mime="application/json")

# ---------- Tab 2: Compare datasets ----------
with tab2:
    if df1 is None or df2 is None:
        st.info("Upload two CSV files to compare.")
    else:
        if ensure_columns(df1, col_repair, col_energy) and ensure_columns(df2, col_repair, col_energy):
            b1 = compute_block(df1, "A", sim_factor)
            b2 = compute_block(df2, "B", sim_factor)

            s1 = b1["summary"]["mean"]
            s2 = b2["summary"]["mean"]
            delta = (s2 - s1)
            pct = (delta / s1) * 100 if s1 != 0 else float("inf")

            colA, colB, colC = st.columns(3)
            colA.metric("Mean RYE A", f"{s1:.4f}")
            colB.metric("Mean RYE B", f"{s2:.4f}")
            colC.metric("Δ (B - A)", f"{delta:.4f}", f"{pct:.2f}%")

            fig = px.line(b1["rye"], title="RYE comparison")
            fig.add_scatter(y=b2["rye"], mode="lines", name="B")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(b1["rye_roll"], title=f"RYE rolling {window} comparison")
            fig2.add_scatter(y=b2["rye_roll"], mode="lines", name="B")
            st.plotly_chart(fig2, use_container_width=True)

# ---------- Tab 3: Multi domain ----------
with tab3:
    if df1 is None:
        st.info("Upload a CSV to see domain splits.")
    else:
        if col_domain not in df1.columns:
            st.info(f"No domain column named '{col_domain}' found.")
        elif ensure_columns(df1, col_repair, col_energy):
            block = compute_block(df1, "primary", sim_factor)
            dfp = block["df"].copy()
            dfp["RYE"] = block["rye"]

            if col_time in dfp.columns:
                fig = px.line(dfp, x=col_time, y="RYE", color=col_domain, title="RYE by domain")
            else:
                fig = px.line(dfp, y="RYE", color=col_domain, title="RYE by domain")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 4: Reports ----------
with tab4:
    if df1 is None:
        st.info("Upload a CSV to generate a report.")
    else:
        if ensure_columns(df1, col_repair, col_energy):
            block = compute_block(df1, "primary", sim_factor)
            rye = block["rye"]
            rye_roll = block["rye_roll"]
            summary = block["summary"]

            st.write("Build a portable report to share with teams.")
            colx, coly = st.columns(2)
            with colx:
                if st.button("Generate PDF report", use_container_width=True):
                    meta = {
                        "rows": len(block["df"]),
                        "columns": ", ".join(list(block["df"].columns)),
                        "repair_col": col_repair,
                        "energy_col": col_energy,
                        "rolling_window": window,
                        "energy_multiplier": sim_factor,
                    }
                    pdf_bytes = build_pdf(
                        list(rye),
                        summary,
                        title="RYE Report",
                        metadata=meta,
                        plot_series={"rye": list(rye), "rye_roll": list(rye_roll)},
                    )
                    st.download_button(
                        "Download RYE report PDF",
                        pdf_bytes,
                        file_name="rye_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            with coly:
                csv_bytes = pd.Series(rye, name="RYE").to_csv(index_label="index").encode("utf-8")
                st.download_button("Download RYE CSV", csv_bytes, file_name="rye.csv", mime="text/csv", use_container_width=True)

# Footer
st.write("")
note("Open science by Cody Ryan Jenkins. CC BY 4.0. Add your Zenodo links in the sidebar Help section.")
