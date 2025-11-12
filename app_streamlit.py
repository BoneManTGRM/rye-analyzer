# app_streamlit.py
# RYE Analyzer — rich, mobile-friendly Streamlit app with:
# Multi-format ingest • Presets & tooltips • Auto column detect • Auto rolling window
# Single / Compare / Multi-domain / Reports tabs • Diagnostics & PDF reporting

from __future__ import annotations
import io, json, os, sys, traceback, importlib.util, importlib.machinery
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------- Make sure we can see core.py regardless of folder layout -------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    ROOT,
    os.path.join(ROOT, "src"),
    os.path.join(ROOT, "app"),
    os.path.join(ROOT, "rye_analyzer"),
    os.path.join(ROOT, "rye-analyzer"),
]
for p in CANDIDATE_PATHS:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

def _try_import_core():
    # 1) normal import
    try:
        import core  # type: ignore
        return core
    except Exception:
        pass
    # 2) package-style (if running inside a pkg)
    try:
        from . import core  # type: ignore
        return core
    except Exception:
        pass
    # 3) direct load from file if it exists nearby
    for base in CANDIDATE_PATHS:
        fp = os.path.join(base, "core.py")
        if os.path.isfile(fp):
            try:
                loader = importlib.machinery.SourceFileLoader("core", fp)
                mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("core", loader))
                loader.exec_module(mod)  # type: ignore
                sys.modules["core"] = mod
                return mod
            except Exception:
                pass
    return None

_core_mod = _try_import_core()
if _core_mod is None:
    st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
    st.error(
        "Could not import **core.py**. Ensure a file named `core.py` is in the same folder "
        "as this app or in one of: `src/`, `app/`, `rye_analyzer/`, `rye-analyzer/`."
    )
    st.stop()

# ---------------- Try to import PRESETS ----------------
PRESETS = None
preset_import_error: Optional[str] = None
try:
    from presets import PRESETS as _PRESETS  # preferred location
    PRESETS = _PRESETS
except Exception as e1:
    try:
        PRESETS = _core_mod.PRESETS  # fallback if kept in core.py
    except Exception as e2:
        preset_import_error = (
            "Could not import PRESETS (presets.py/core.py). Using a tiny default. "
            f"{e1.__class__.__name__}: {e1} / {e2.__class__.__name__}: {e2}"
        )
        PRESETS = {
            "Generic": type("Preset", (), {
                "name": "Generic",
                "time": ["time"],
                "performance": ["performance"],
                "energy": ["energy"],
                "domain": "domain",
                "default_rolling": 10,
                "tooltips": {"Generic": "Basic preset used when presets are not available."}
            })()
        }

# ---------------- Import core helpers (with graceful fallbacks) ----------------
load_table = _core_mod.load_table
normalize_columns = _core_mod.normalize_columns
safe_float = _core_mod.safe_float

# compute_rye_from_df may be named differently in older cores; alias whichever exists
if hasattr(_core_mod, "compute_rye_from_df"):
    _compute_rye_from_df = _core_mod.compute_rye_from_df
elif hasattr(_core_mod, "compute_rye"):
    _compute_rye_from_df = _core_mod.compute_rye
else:
    st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
    st.error("Neither `compute_rye_from_df` nor `compute_rye` found in core.py — add one of them.")
    st.stop()

# summarize function may be named summarize_series or summarize
if hasattr(_core_mod, "summarize_series"):
    _summarize_series = _core_mod.summarize_series
elif hasattr(_core_mod, "summarize"):
    _summarize_series = _core_mod.summarize
else:
    st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
    st.error("Neither `summarize_series` nor `summarize` found in core.py — add one of them.")
    st.stop()

# rolling_series is optional; provide a pandas fallback if missing
if hasattr(_core_mod, "rolling_series"):
    _rolling_series = _core_mod.rolling_series
else:
    def _rolling_series(arr, window: int):
        s = pd.Series(arr, dtype=float)
        if window <= 1:
            return s.fillna(0.0).values
        return s.rolling(window=window, min_periods=1).mean().values

# optional column inference
_infer_columns = getattr(_core_mod, "infer_columns", None)

# Optional advanced analytics (each may be absent)
detect_regimes = getattr(_core_mod, "detect_regimes", None)
energy_delta_performance_correlation = getattr(_core_mod, "energy_delta_performance_correlation", None)
estimate_noise_floor = getattr(_core_mod, "estimate_noise_floor", None)
bootstrap_rolling_mean = getattr(_core_mod, "bootstrap_rolling_mean", None)

# ---------------- Local helpers (work even if core lacks them) ----------------
def ema_series(x, span: int) -> np.ndarray:
    if span is None or span <= 1:
        return np.asarray(x, dtype=float)
    s = pd.Series(x, dtype=float)
    return s.ewm(span=span, adjust=False).mean().values

def cumulative_series(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return np.cumsum(a)

def smart_window(n_rows: int, preset_default: Optional[int]) -> int:
    if preset_default and preset_default > 0:
        return int(preset_default)
    if n_rows <= 0:
        return 10
    # ~5% of series length, clipped [3, 200]
    guess = max(3, min(200, int(round(max(3, n_rows * 0.05)))))
    return guess

# ---------------- PDF builder (optional) + diagnostics ----------------
def _probe_fpdf_version() -> str:
    try:
        import fpdf
        ver = getattr(fpdf, "__version__", "installed (version unknown)")
        return f"fpdf module found: {ver}"
    except Exception as e:
        return f"fpdf import failed: {e.__class__.__name__}: {e}"

build_pdf = None
_pdf_import_error = None
try:
    spec = importlib.util.find_spec("report")
    if spec is None:
        _pdf_import_error = "report.py not found in the working directory."
    else:
        from report import build_pdf  # noqa
except Exception:
    _pdf_import_error = traceback.format_exc()

# ---------------- UI helpers ----------------
def add_stability_bands(fig: go.Figure, y_max_hint: Optional[float] = None):
    top = y_max_hint if y_max_hint is not None else 1.0
    fig.add_hrect(y0=0.6, y1=max(0.6, top), fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.3, y1=0.6,             fillcolor="yellow", opacity=0.08, line_width=0)
    fig.add_hrect(y0=min(-0.5, 0.3 - 1.0), y1=0.3, fillcolor="red", opacity=0.08, line_width=0)

def _delta_performance(series):
    arr = pd.Series(series, dtype=float)
    return arr.diff().fillna(0.0).values

def _first_or(default, lst):
    return (lst[0] if isinstance(lst, list) and lst else default)

_HAS_POPOVER = hasattr(st, "popover")

# ---------------- Page config ----------------
st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
st.title("RYE Analyzer")

with st.expander("What is RYE?"):
    st.write(
        "Repair Yield per Energy (RYE) measures how efficiently a system converts effort or energy into "
        "successful repair or performance gains. Higher RYE means better efficiency."
    )

# ---------------- Seed session_state BEFORE widgets to allow safe updates ----------------
# This prevents the "cannot be modified after the widget with key ... is instantiated" error
_default_preset_name = list(PRESETS.keys())[0]
_default_preset = PRESETS.get(_default_preset_name, next(iter(PRESETS.values())))
if "col_time" not in st.session_state:
    st.session_state["col_time"] = _first_or("time", getattr(_default_preset, "time", ["time"]))
if "col_domain" not in st.session_state:
    st.session_state["col_domain"] = getattr(_default_preset, "domain", "domain") or "domain"
if "col_repair" not in st.session_state:
    st.session_state["col_repair"] = _first_or("performance", getattr(_default_preset, "performance", ["performance"]))
if "col_energy" not in st.session_state:
    st.session_state["col_energy"] = _first_or("energy", getattr(_default_preset, "energy", ["energy"]))

# ---------------- Sidebar (inputs) ----------------
with st.sidebar:
    st.header("Inputs")

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS.get(preset_name, next(iter(PRESETS.values())))

    ttips = getattr(preset, "tooltips", None) or {}
    if isinstance(ttips, dict) and ttips:
        if _HAS_POPOVER:
            with st.popover("Preset tips", use_container_width=True):
                for k, v in ttips.items():
                    st.markdown(f"**{k}** — {v}")
        else:
            with st.expander("Preset tips"):
                for k, v in ttips.items():
                    st.markdown(f"**{k}** — {v}")

    st.write("Upload one file to analyze. Optionally upload a second file to compare.")
    file_types = ["csv", "tsv", "xls", "xlsx", "parquet", "feather", "json", "ndjson", "h5", "hdf5", "nc", "netcdf"]
    file1 = st.file_uploader("Primary file", type=file_types, key="file1")
    file2 = st.file_uploader("Comparison file (optional)", type=file_types, key="file2")

    st.divider()
    st.write("Column names in your data")
    col_time   = st.text_input("Time column (optional)", value=st.session_state["col_time"],   key="col_time")
    col_domain = st.text_input("Domain column (optional)", value=st.session_state["col_domain"], key="col_domain")
    col_repair = st.text_input("Performance/Repair column", value=st.session_state["col_repair"], key="col_repair")
    col_energy = st.text_input("Energy/Effort column",     value=st.session_state["col_energy"],  key="col_energy")

    if st.button("Auto-detect columns from data"):
        if _infer_columns is None:
            st.warning("Column inference not available (core.infer_columns missing).")
        elif file1 is None:
            st.warning("Upload a primary file first.")
        else:
            try:
                _tmp = normalize_columns(load_table(file1))
                guess = _infer_columns(_tmp, preset_name=preset_name)
                updates = {}
                if guess.get("time"):        updates["col_time"] = guess["time"]
                if guess.get("domain"):      updates["col_domain"] = guess["domain"]
                if guess.get("performance"): updates["col_repair"] = guess["performance"]
                if guess.get("energy"):      updates["col_energy"] = guess["energy"]
                if updates:
                    st.session_state.update(updates)
                st.success(f"Detected: {guess}")
                st.rerun()
            except Exception as e:
                st.error(f"Auto-detect failed: {e}")

    st.divider()
    default_window = int(getattr(preset, "default_rolling", 10) or 10)
    auto_roll = st.checkbox("Auto rolling window", value=True, help="Use preset default or smart guess by series length.")
    window = st.number_input("Rolling window", min_value=1, max_value=1000, value=default_window, step=1,
                             help="Moving average length applied to the RYE series.", disabled=auto_roll)

    ema_span = st.number_input("EMA smoothing (optional)", min_value=0, max_value=1000, value=0, step=1,
                               help="Extra smoothing; 0 disables EMA.")
    sim_factor = st.slider("Multiply energy by", min_value=0.10, max_value=3.0, value=1.0, step=0.05,
                           help="What-if: scale energy before computing RYE.")

    st.divider()
    doi_or_link = st.text_input(
        "Zenodo DOI or dataset link (optional)",
        value="",
        help="Example: 10.5281/zenodo.123456 or a dataset URL"
    )

    st.divider()
    if preset_import_error:
        st.info(preset_import_error)
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
def load_any(file) -> Optional[pd.DataFrame]:
    if file is None:
        return None
    try:
        df = load_table(file)
        df = normalize_columns(df)
        if df.empty:
            st.error("The file was read successfully, but it contains no rows.")
            return None
        return df
    except Exception as e:
        st.error(f"Could not read file. {e}")
        st.code(traceback.format_exc(), language="text")
        return None

def ensure_columns(df: pd.DataFrame, repair: str, energy: str) -> bool:
    miss = [c for c in [repair, energy] if c not in df.columns]
    if miss:
        st.error(f"Missing columns: {', '.join(miss)}")
        st.write("Found columns:", list(df.columns))
        return False
    return True

def compute_block(df: pd.DataFrame, label: str, sim_mult: float, auto_roll_flag: bool) -> Dict[str, Any]:
    df_sim = df.copy()
    if col_energy in df_sim.columns:
        df_sim[col_energy] = pd.to_numeric(df_sim[col_energy], errors="coerce").apply(
            lambda x: safe_float(x) * sim_mult
        )

    rye = _compute_rye_from_df(df_sim, repair_col=col_repair, energy_col=col_energy)

    # window selection
    w = window
    if auto_roll_flag:
        w = smart_window(len(df_sim), getattr(preset, "default_rolling", None))

    rye_roll = _rolling_series(rye, w)
    rye_ema = ema_series(rye, span=int(ema_span)) if ema_span and ema_span > 1 else None
    rye_cum = cumulative_series(rye)

    summary = _summarize_series(rye)
    summary_roll = _summarize_series(rye_roll)

    out: Dict[str, Any] = {
        "label": label,
        "df": df_sim,
        "w": w,
        "rye": rye,
        "rye_roll": rye_roll,
        "rye_ema": rye_ema,
        "rye_cum": rye_cum,
        "summary": summary,
        "summary_roll": summary_roll,
    }

    # Optional advanced analytics
    try:
        if detect_regimes is not None:
            out["regimes"] = detect_regimes(rye_roll if len(rye_roll) >= max(3, w) else rye)
    except Exception:
        out["regimes"] = None

    try:
        if energy_delta_performance_correlation is not None and col_repair in df_sim.columns and col_energy in df_sim.columns:
            out["correlation"] = energy_delta_performance_correlation(df_sim, perf_col=col_repair, energy_col=col_energy)
    except Exception:
        out["correlation"] = None

    try:
        if estimate_noise_floor is not None:
            out["noise_floor"] = estimate_noise_floor(rye)
    except Exception:
        out["noise_floor"] = None

    try:
        if bootstrap_rolling_mean is not None:
            out["bands"] = bootstrap_rolling_mean(rye, window=w, n_boot=100)
    except Exception:
        out["bands"] = None

    return out

def make_interpretation(summary: dict, w: int, sim_mult: float) -> str:
    mean_v = float(summary.get("mean", 0) or 0)
    max_v  = float(summary.get("max", 0) or 0)
    min_v  = float(summary.get("min", 0) or 0)
    resil  = float(summary.get("resilience", 0) or 0) if "resilience" in summary else None

    lines = []
    lines.append(f"Average efficiency (RYE mean) is {mean_v:.3f}. Range ≈ [{min_v:.3f}, {max_v:.3f}].")
    if resil is not None:
        lines.append(f"Resilience index is {resil:.3f} — higher means steadier efficiency under fluctuation.")
    if mean_v > 1.0:
        lines.append("Each unit of energy returned more than one unit of repair on average — excellent efficiency.")
    elif mean_v > 0.5:
        lines.append("Efficiency is solid. Trim energy overhead and target low-yield segments to lift the mean further.")
    else:
        lines.append("Efficiency is modest. Hunt for high-energy/low-return regions to prune or repair.")
    lines.append(f"Rolling window of {w} smooths short-term noise.")
    if sim_mult != 1.0:
        lines.append(("Energy down-scaling" if sim_mult < 1.0 else "Energy up-scaling") +
                     f" factor = {sim_mult:.2f}. Expect RYE to {'rise' if sim_mult < 1.0 else 'fall unless repair also improves'}.")
    lines.append("Next: map spikes/dips to interventions and iterate TGRM loops (detect → minimal fix → verify).")
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
            block = compute_block(df1, "primary", sim_factor, auto_roll)
            rye = block["rye"]
            rye_roll = block["rye_roll"]
            rye_ema = block["rye_ema"]
            rye_cum = block["rye_cum"]
            w = block["w"]
            summary = block["summary"]

            colA, colB, colC = st.columns(3)
            colA.metric("RYE mean", f"{summary.get('mean', 0):.4f}", help="Average RYE across rows")
            colB.metric("Median",   f"{summary.get('median', 0):.4f}")
            colC.metric("Resilience", f"{summary.get('resilience', 0):.3f}" if "resilience" in summary else "—",
                        help="Stability of efficiency under fluctuation (if computed)")

            st.write("Columns:")
            st.json(list(df1.columns))

            # RYE line(s)
            if col_time in df1.columns:
                idx = df1[col_time]
                fig = px.line(x=idx, y=rye, labels={"x": col_time, "y": "RYE"}, title="RYE")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(x=idx, y=rye_roll, labels={"x": col_time, "y": f"RYE rolling {w}"},
                               title=f"RYE rolling window {w}")
                if rye_ema is not None:
                    fig2.add_scatter(x=idx, y=rye_ema, mode="lines", name=f"EMA {ema_span}")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.line(x=idx, y=rye_cum, labels({"x": col_time, "y": "Cumulative RYE"}),
                               title="Cumulative RYE (sum of step yields)")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                fig = px.line(pd.DataFrame({"RYE": rye}), y="RYE", title="RYE")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                ycols = {f"RYE rolling {w}": rye_roll}
                if rye_ema is not None:
                    ycols[f"EMA {ema_span}"] = rye_ema
                fig2 = px.line(pd.DataFrame(ycols), title=f"Smoothed RYE (rolling {w} & EMA)")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.line(pd.DataFrame({"Cumulative RYE": rye_cum}), y="Cumulative RYE", title="Cumulative RYE")
                st.plotly_chart(fig3, use_container_width=True)

            # Extra charts
            with st.expander("More visuals"):
                hist = px.histogram(pd.DataFrame({"RYE": rye}), x="RYE", nbins=30, title="RYE distribution (histogram)")
                st.plotly_chart(hist, use_container_width=True)

                if col_repair in df1.columns and col_energy in df1.columns:
                    dperf = _delta_performance(df1[col_repair])
                    scatter = px.scatter(
                        x=df1[col_energy], y=dperf,
                        labels={"x": col_energy, "y": "Δ" + col_repair},
                        title="Energy vs ΔPerformance"
                    )
                    st.plotly_chart(scatter, use_container_width=True)

            # Diagnostics (optional analytics)
            with st.expander("Diagnostics"):
                if energy_delta_performance_correlation is not None:
                    try:
                        corr = energy_delta_performance_correlation(df1, perf_col=col_repair, energy_col=col_energy)
                        st.write("Energy–ΔPerformance correlation:", corr)
                    except Exception:
                        pass
                if estimate_noise_floor is not None:
                    try:
                        noise = estimate_noise_floor(rye_roll)
                        st.write("Noise floor:", noise)
                    except Exception:
                        pass
                if detect_regimes is not None:
                    try:
                        regimes = detect_regimes(rye_roll)
                        if regimes:
                            st.write("Detected regimes:")
                            st.json(regimes)
                    except Exception:
                        pass

            st.divider()
            st.subheader("Summary")
            st.code(json.dumps(summary, indent=2))

            enriched = df1.copy()
            enriched["RYE"] = rye
            enriched[f"RYE_rolling_{w}"] = rye_roll
            if rye_ema is not None:
                enriched[f"RYE_ema_{ema_span}"] = rye_ema
            enriched["RYE_cumulative"] = rye_cum
            st.download_button(
                "Download enriched CSV (with RYE)",
                enriched.to_csv(index=False).encode("utf-8"),
                file_name="rye_enriched.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download RYE series CSV",
                pd.Series(rye, name="RYE").to_csv(index_label="index").encode("utf-8"),
                file_name="rye.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download summary JSON",
                io.BytesIO(json.dumps(summary, indent=2).encode("utf-8")).getvalue(),
                file_name="summary.json",
                mime="application/json"
            )

# ---------- Tab 2 ----------
with tab2:
    if df1 is None or df2 is None:
        st.info("Upload two files to compare.")
    else:
        if ensure_columns(df1, col_repair, col_energy) and ensure_columns(df2, col_repair, col_energy):
            b1 = compute_block(df1, "A", sim_factor, auto_roll)
            b2 = compute_block(df2, "B", sim_factor, auto_roll)

            s1 = b1["summary"].get("mean", 0.0)
            s2 = b2["summary"].get("mean", 0.0)
            r1 = b1["summary"].get("resilience", 0)
            r2 = b2["summary"].get("resilience", 0)
            delta = (s2 - s1)
            pct = (delta / s1) * 100 if s1 != 0 else float("inf")

            colA, colB, colC, colD = st.columns(4)
            colA.metric("Mean RYE A", f"{s1:.4f}")
            colB.metric("Mean RYE B", f"{s2:.4f}")
            colC.metric("Δ Mean", f"{delta:.4f}", f"{pct:.2f}%")
            colD.metric("Resilience A / B", f"{r1:.3f} / {r2:.3f}" if r1 or r2 else "—")

            if col_time in df1.columns and col_time in df2.columns:
                x1 = df1[col_time]; x2 = df2[col_time]
                fig = px.line(x=x1, y=b1["rye"], labels={"x": col_time, "y": "RYE"}, title="RYE comparison")
                fig.add_scatter(x=x2, y=b2["rye"], mode="lines", name="B")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(x=x1, y=b1["rye_roll"], labels({"x": col_time, "y": f"RYE rolling {b1['w']}"}),
                               title=f"RYE rolling comparison (A:{b1['w']} / B:{b2['w']})")
                fig2.add_scatter(x=x2, y=b2["rye_roll"], mode="lines", name="B")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig = px.line(pd.DataFrame({"RYE_A": b1["rye"]}), y="RYE_A", title="RYE comparison")
                fig.add_scatter(y=b2["rye"], mode="lines", name="RYE_B")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(pd.DataFrame({f"RYE_A_rolling_{b1['w']}": b1["rye_roll"]}),
                               y=f"RYE_A_rolling_{b1['w']}", title="RYE rolling comparison")
                fig2.add_scatter(y=b2["rye_roll"], mode="lines", name=f"RYE_B_rolling_{b2['w']}")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            combined = pd.DataFrame({
                "RYE_A": b1["rye"],
                "RYE_B": b2["rye"],
                f"RYE_A_rolling_{b1['w']}": b1["rye_roll"],
                f"RYE_B_rolling_{b2['w']}": b2["rye_roll"],
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
            b = compute_block(df1, "primary", sim_factor, auto_roll)
            dfp = b["df"].copy()
            dfp["RYE"] = b["rye"]

            if col_time in dfp.columns:
                fig = px.line(dfp, x=col_time, y="RYE", color=col_domain, title="RYE by domain")
            else:
                fig = px.line(dfp, y="RYE", color=col_domain, title="RYE by domain")
            add_stability_bands(fig)
            st.plotly_chart(fig)
# ---------- Tab 4 ----------
with tab4:
    if df1 is None:
        st.info("Upload a file to generate a report.")
    else:
        if ensure_columns(df1, col_repair, col_energy):
            b = compute_block(df1, "primary", sim_factor, auto_roll)
            rye = b["rye"]; rye_roll = b["rye_roll"]; w = b["w"]; summary = b["summary"]

            st.write("Build a portable report to share with teams.")

            metadata = {
                "rows": len(df1),
                "preset": preset_name,
                "repair_col": col_repair,
                "energy_col": col_energy,
                "time_col": col_time if col_time in df1.columns else "",
                "domain_col": col_domain if col_domain in df1.columns else "",
                "rolling_window": w,
                "columns": list(df1.columns),
            }
            if doi_or_link.strip():
                metadata["dataset_link"] = doi_or_link.strip()
            for k in ("regimes", "correlation", "noise_floor", "bands"):
                if b.get(k) is not None:
                    metadata[k] = b[k]

            interp = make_interpretation(summary, w, sim_factor)

            with st.expander("PDF diagnostics", expanded=False):
                if build_pdf is None:
                    st.error("PDF builder is not loaded.")
                    st.write(_probe_fpdf_version())
                    if _pdf_import_error:
                        st.code(_pdf_import_error, language="text")
                    try:
                        st.write("Working directory:", os.getcwd())
                        st.write("Files:", os.listdir("."))
                        st.write("Python:", sys.version)
                        fonts_ok = os.path.exists("fonts")
                        st.write("fonts/ exists:", fonts_ok)
                        if fonts_ok:
                            st.write("fonts/ contents:", os.listdir("fonts"))
                    except Exception as e:
                        st.write("Diag error:", e)
                else:
                    st.success("PDF builder loaded.")
                    st.write(_probe_fpdf_version())

            if st.button("Generate PDF report", use_container_width=True):
                if build_pdf is None:
                    st.error("PDF generator not available. Ensure report.py exists and **fpdf2** is in requirements.txt")
                else:
                    try:
                        pdf_bytes = build_pdf(
                            list(rye),
                            summary,
                            metadata=metadata,
                            plot_series={"RYE": list(rye), f"RYE rolling {w}": list(rye_roll)},
                            interpretation=interp,
                        )
                        st.download_button(
                            "Download RYE report PDF",
                            data=pdf_bytes,
                            file_name="rye_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        st.code(traceback.format_exc(), language="text")
