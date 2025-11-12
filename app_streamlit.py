# app_streamlit.py
# RYE Analyzer full app with CSV/TSV/XLSX support, presets, stability bands, resilience, and PDF reporting

from __future__ import annotations
import io, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local helpers
from core import (
    load_table,              # reads csv/tsv/xls/xlsx
    normalize_columns,       # snake_case headers
    PRESETS,                 # AI / Biology / Robotics labels + tooltips
    compute_rye_from_df,
    rolling_series,
    safe_float,
    summarize_series,        # now includes "resilience"
    append_rye_columns,      # adds RYE_step / RYE_cum (for future use) — optional
)

try:
    from report import build_pdf  # returns bytes
except Exception:
    build_pdf = None

# ---------------- UI helpers ----------------
def section(title: str):
    st.subheader(title)

def note(msg: str):
    st.caption(msg)

def add_stability_bands(fig: go.Figure, y_max_hint: float | None = None):
    # Visual efficiency zones (soft overlays)
    # Green: >= 0.6, Yellow: 0.3–0.6, Red: < 0.3
    # Use generous y extents so bands are visible regardless of data range
    top = y_max_hint if y_max_hint is not None else 1.0
    fig.add_hrect(y0=0.6, y1=max(0.6, top), fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.3, y1=0.6,             fillcolor="yellow", opacity=0.08, line_width=0)
    fig.add_hrect(y0=min(-0.5, 0.3 - 1.0), y1=0.3, fillcolor="red", opacity=0.08, line_width=0)

# ---------------- Page config ----------------
st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
st.title("RYE Analyzer")
st.caption("Compute Repair Yield per Energy from any time series.")

with st.expander("What is RYE"):
    st.write(
        "Repair Yield per Energy (RYE) measures how efficiently a system converts effort or energy into successful "
        "repair or performance gains. Higher RYE means better efficiency. Upload a dataset to compute RYE and explore "
        "rolling windows, comparisons, and reports."
    )

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")

    # Preset labels for copy only, internal column names can stay the same
    preset_name = st.selectbox("Preset", ["AI", "Biology", "Robotics"], index=1)
    preset = PRESETS[preset_name]

    st.write("Upload one file to analyze. Optionally upload a second file to compare.")
    file1 = st.file_uploader("Primary file", type=["csv", "tsv", "xls", "xlsx"], key="csv1")
    file2 = st.file_uploader("Comparison file (optional)", type=["csv", "tsv", "xls", "xlsx"], key="csv2")

    st.divider()
    st.write("Column names in your data")
    col_repair = st.text_input(f"{preset['repair_label']} column", value="performance")
    col_energy = st.text_input(f"{preset['energy_label']} column", value="energy")
    col_time   = st.text_input("Time column (optional)", value="time")
    col_domain = st.text_input("Domain column (optional)", value="domain")

    st.divider()
    window = st.number_input("Rolling window", min_value=1, max_value=500, value=10, step=1)

    st.divider()
    st.write("Energy simulator")
    sim_factor = st.slider("Multiply energy by", min_value=0.10, max_value=3.0, value=1.0, step=0.05)

    st.divider()
    doi_or_link = st.text_input("Zenodo DOI or dataset link (optional)", value="")

    st.divider()
    st.write("No data yet")
    if st.button("Download example CSV"):
        example = pd.DataFrame({
            "time": np.arange(0, 15),
            "domain": ["AI"] * 5 + ["Bio"] * 5 + ["Robotics"] * 5,
            "performance": [0, 0, 0.1, 0.2, 0.35, 0.38, 0.5, 0.46, 0.52, 0.6, 0.6, 0.6, 0.62, 0.62, 0.65],
            "energy":      [1, 1, 1, 1, 1, 1.1, 1.0, 1.02, 1.05, 1.1, 1.1, 1.12, 1.09, 1.1, 1.1],
        })
        b = example.to_csv(index=False).encode("utf-8")
        st.download_button("Save example.csv", b, file_name="example.csv", mime="text/csv")

# ---------------- Core workers ----------------
def load_any(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        df = load_table(file)          # supports csv/tsv/xls/xlsx
        df = normalize_columns(df)     # snake_case headers
        return df
    except Exception as e:
        import traceback
        st.error(f"Could not read file. {e}")
        st.text(traceback.format_exc())
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

    summary = summarize_series(rye)          # includes "resilience"
    summary_roll = summarize_series(rye_roll)

    return {
        "label": label,
        "df": df_sim,
        "rye": rye,
        "rye_roll": rye_roll,
        "summary": summary,
        "summary_roll": summary_roll,
    }

def make_interpretation(summary: dict, window: int, sim_mult: float) -> str:
    mean_v = float(summary.get("mean", 0) or 0)
    max_v  = float(summary.get("max", 0) or 0)
    min_v  = float(summary.get("min", 0) or 0)
    resil  = float(summary.get("resilience", 0) or 0)

    lines = []
    lines.append(f"Average efficiency (RYE mean) is {mean_v:.3f}. "
                 f"Values typically range between {min_v:.3f} and {max_v:.3f}.")
    lines.append(f"Resilience index (stability) is {resil:.3f} — higher means steadier efficiency under fluctuation.")
    if mean_v > 1.0:
        lines.append("On average, each unit of energy returned more than one unit of repair which is an excellent efficiency level.")
    elif mean_v > 0.5:
        lines.append("Efficiency is solid. Small process changes that reduce energy or boost repair should lift the mean further.")
    else:
        lines.append("Efficiency is modest. Look for high energy and low return segments to prune or repair.")

    lines.append(f"The report used a rolling window of {window} to smooth short term noise.")
    if sim_mult != 1.0:
        if sim_mult < 1.0:
            lines.append(f"An energy down scaling factor of {sim_mult:.2f} was simulated. RYE should increase under this scenario.")
        else:
            lines.append(f"An energy up scaling factor of {sim_mult:.2f} was simulated. RYE may fall unless repair improved proportionally.")

    lines.append("Next steps: identify spikes or dips in the RYE curve, map them to events or interventions, "
                 "and iterate TGRM loops to raise average RYE.")
    return " ".join(lines)

# ---------------- Main UI ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Single analysis", "Compare datasets", "Multi domain", "Reports"])

df1 = load_any(file1)
df2 = load_any(file2)

# ---------- Tab 1 ----------
with tab1:
    if df1 is None:
        st.info("Upload a file in the sidebar to begin.")
    else:
        if ensure_columns(df1, col_repair, col_energy):
            block = compute_block(df1, "primary", sim_factor)
            rye = block["rye"]
            rye_roll = block["rye_roll"]
            summary = block["summary"]

            colA, colB = st.columns(2)
            colA.metric("RYE mean", f"{summary['mean']:.4f}", help="Average RYE across rows")
            colB.metric("Resilience Index", f"{summary.get('resilience',0):.3f}", help="How stable efficiency remains under fluctuation")

            st.write("Columns:")
            st.json(list(df1.columns))

            # RYE line
            if col_time in df1.columns:
                idx = df1[col_time]
                fig = px.line(x=idx, y=rye, labels={"x": col_time, "y": "RYE"}, title="RYE")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(x=idx, y=rye_roll, labels={"x": col_time, "y": f"RYE rolling {window}"},
                               title=f"RYE rolling window {window}")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig = px.line(rye, title="RYE")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(rye_roll, title=f"RYE rolling window {window}")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            section("Summary")
            st.code(json.dumps(summary, indent=2))

            # Enriched table export (original df + RYE + rolling RYE)
            enriched = df1.copy()
            enriched["RYE"] = rye
            enriched[f"RYE_rolling_{window}"] = rye_roll
            enriched_bytes = enriched.to_csv(index=False).encode("utf-8")
            st.download_button("Download enriched CSV (with RYE)", enriched_bytes, file_name="rye_enriched.csv", mime="text/csv")

            # Plain RYE + summary exports
            csv_bytes = pd.Series(rye, name="RYE").to_csv(index_label="index").encode("utf-8")
            st.download_button("Download RYE series CSV", csv_bytes, file_name="rye.csv", mime="text/csv")

            json_bytes = io.BytesIO(json.dumps(summary, indent=2).encode("utf-8"))
            st.download_button("Download summary JSON", json_bytes.getvalue(), file_name="summary.json", mime="application/json")

# ---------- Tab 2 ----------
with tab2:
    if df1 is None or df2 is None:
        st.info("Upload two files to compare.")
    else:
        if ensure_columns(df1, col_repair, col_energy) and ensure_columns(df2, col_repair, col_energy):
            b1 = compute_block(df1, "A", sim_factor)
            b2 = compute_block(df2, "B", sim_factor)

            s1 = b1["summary"]["mean"]
            s2 = b2["summary"]["mean"]
            r1 = b1["summary"].get("resilience", 0)
            r2 = b2["summary"].get("resilience", 0)
            delta = (s2 - s1)
            pct = (delta / s1) * 100 if s1 != 0 else float("inf")

            colA, colB, colC, colD = st.columns(4)
            colA.metric("Mean RYE A", f"{s1:.4f}")
            colB.metric("Mean RYE B", f"{s2:.4f}")
            colC.metric("Δ Mean", f"{delta:.4f}", f"{pct:.2f}%")
            colD.metric("Resilience A / B", f"{r1:.3f} / {r2:.3f}")

            if col_time in df1.columns and col_time in df2.columns:
                x1 = df1[col_time]
                x2 = df2[col_time]
                fig = px.line(x=x1, y=b1["rye"], labels={"x": col_time, "y": "RYE"}, title="RYE comparison")
                fig.add_scatter(x=x2, y=b2["rye"], mode="lines", name="B")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(x=x1, y=b1["rye_roll"], labels={"x": col_time, "y": f"RYE rolling {window}"},
                               title=f"RYE rolling {window} comparison")
                fig2.add_scatter(x=x2, y=b2["rye_roll"], mode="lines", name="B")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig = px.line(b1["rye"], title="RYE comparison")
                fig.add_scatter(y=b2["rye"], mode="lines", name="B")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(b1["rye_roll"], title=f"RYE rolling {window} comparison")
                fig2.add_scatter(y=b2["rye_roll"], mode="lines", name="B")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            # Optional: combined export (side-by-side RYE series for A and B)
            combined = pd.DataFrame({
                "RYE_A": b1["rye"],
                "RYE_B": b2["rye"],
                f"RYE_A_rolling_{window}": b1["rye_roll"],
                f"RYE_B_rolling_{window}": b2["rye_roll"],
            })
            st.download_button(
                "Download combined CSV (A vs B)",
                combined.to_csv(index_label="index").encode("utf-8"),
                file_name="rye_combined.csv",
                mime="text/csv"
            )

# ---------- Tab 3 ----------
with tab3:
    if df1 is None:
        st.info("Upload a file to see domain splits.")
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
            add_stability_bands(fig)
            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 4 ----------
with tab4:
    if df1 is None:
        st.info("Upload a file to generate a report.")
    else:
        if ensure_columns(df1, col_repair, col_energy):
            block = compute_block(df1, "primary", sim_factor)
            rye = block["rye"]
            rye_roll = block["rye_roll"]
            summary = block["summary"]

            st.write("Build a portable report to share with teams.")

            metadata = {
                "rows": len(df1),
                "preset": preset_name,
                "repair_col": col_repair,
                "energy_col": col_energy,
                "time_col": col_time if col_time in df1.columns else "",
                "domain_col": col_domain if col_domain in df1.columns else "",
                "rolling_window": window,
            }
            if doi_or_link.strip():
                metadata["dataset_link"] = doi_or_link.strip()

            interp = make_interpretation(summary, window, sim_factor)

            colx, coly = st.columns(2)
            with colx:
                if st.button("Generate PDF report", use_container_width=True):
                    if build_pdf is None:
                        st.error("PDF generator not available. Ensure report.py exists and fpdf is in requirements.txt")
                    else:
                        pdf_bytes = build_pdf(
                            list(rye),
                            summary,
                            metadata=metadata,
                            plot_series={"RYE": list(rye), "RYE rolling": list(rye_roll)},
                            interpretation=interp,
                        )
                        st.download_button(
                            "Download RYE report PDF",
                            data=pdf_bytes,
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
