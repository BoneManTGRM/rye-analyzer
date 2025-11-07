# report.py
# PDF builder with safe text wrapping, margins, and an auto interpretation block.

from fpdf import FPDF
import io
from datetime import datetime
from typing import Dict, List, Optional

PAGE_W = 210  # A4 mm
PAGE_H = 297
LM = 12       # left margin
RM = 12       # right margin
TM = 12       # top margin
COL_GAP = 6

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "RYE Report", ln=True, align="L")
        self.set_font("Arial", "", 10)
        self.cell(0, 6, f"Generated: {datetime.utcnow().isoformat()}Z", ln=True, align="L")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 9)
        self.cell(
            0, 8,
            "Open science by Cody Ryan Jenkins (CC BY 4.0). Learn more: Reparodynamics — RYE (Repair Yield per Energy).",
            ln=True, align="C"
        )

def _safe_multicell(pdf: FPDF, w: float, h: float, txt: str, align: str = "L"):
    """
    FPDF can throw when a single long 'word' exceeds the line width.
    We pre-wrap very long tokens so multi_cell always has break points.
    """
    if not isinstance(txt, str):
        txt = str(txt)

    # soft wrap extra-long tokens
    max_token = 60
    tokens = []
    for t in txt.split(" "):
        if len(t) <= max_token:
            tokens.append(t)
        else:
            # insert zero-width spaces to hint wrapping for very long tokens/urls
            chunks = [t[i:i+max_token] for i in range(0, len(t), max_token)]
            tokens.append("\u200b".join(chunks))
    safe = " ".join(tokens)

    pdf.multi_cell(w, h, safe, align=align)

def build_pdf(
    rye_series: List[float],
    summary: Dict,
    *,
    title: str = "RYE Report",
    meta: Optional[Dict] = None,
    plot_series: Optional[Dict[str, List[float]]] = None,
    interpretation: Optional[str] = None,
) -> bytes:
    """
    Returns PDF bytes.
    - rye_series: list of RYE values
    - summary: dict with mean, median, max, min, count
    - meta: optional dataset metadata you want listed
    - plot_series: optional extra series to list (for quick inspection)
    - interpretation: optional paragraph to print under 'Interpretation'
    """
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_left_margin(LM)
    pdf.set_right_margin(RM)

    # Title (again to make export stand-alone if header is cropped by previewers)
    pdf.set_font("Arial", "B", 14)
    _safe_multicell(pdf, PAGE_W - LM - RM, 8, title)

    # --- Dataset metadata in two columns (label / value) ---
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Dataset metadata", ln=True)
    pdf.set_font("Arial", "", 11)

    if meta is None:
        meta = {}
    # standard items if present in meta
    lines = []
    for k in ["rows", "repair_col", "energy_col", "time_col", "domain_col", "rolling_window"]:
        if k in meta:
            lines.append((k.replace("_", " "), str(meta[k])))

    # render key/val grid with controlled column widths
    label_w = 38
    val_w = PAGE_W - LM - RM - label_w
    for label, value in lines:
        y_before = pdf.get_y()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(label_w, 6, f"{label}:", ln=0)
        pdf.set_font("Arial", "", 11)
        # value wraps inside remaining width
        x = pdf.get_x()
        _safe_multicell(pdf, val_w, 6, value, align="L")
        # ensure next row stays aligned
        pdf.set_xy(LM, max(y_before + 6, pdf.get_y()))

    # --- Summary stats in two columns so nothing runs off the page ---
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "Summary statistics", ln=True)
    pdf.set_font("Arial", "", 11)

    stats_map = {
        "mean": summary.get("mean", ""),
        "median": summary.get("median", ""),
        "max": summary.get("max", ""),
        "min": summary.get("min", ""),
        "count": summary.get("count", ""),
    }

    keys = list(stats_map.keys())
    half = (len(keys) + 1) // 2
    col1 = keys[:half]
    col2 = keys[half:]

    col_w = (PAGE_W - LM - RM - COL_GAP) / 2
    y_start = pdf.get_y()
    # left column
    x_left = LM
    pdf.set_xy(x_left, y_start)
    for k in col1:
        _safe_multicell(pdf, col_w, 6, f"{k}: {stats_map[k]}", align="L")
    y_after_left = pdf.get_y()
    # right column
    x_right = LM + col_w + COL_GAP
    pdf.set_xy(x_right, y_start)
    for k in col2:
        _safe_multicell(pdf, col_w, 6, f"{k}: {stats_map[k]}", align="L")
    pdf.set_y(max(y_after_left, pdf.get_y()))

    # --- Optional quick list preview of series (first 100) ---
    if rye_series:
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "RYE sample values (first 100)", ln=True)
        pdf.set_font("Arial", "", 10)
        col_w = (PAGE_W - LM - RM - COL_GAP) / 2
        left_vals = []
        right_vals = []
        for i, val in enumerate(rye_series[:100]):
            (left_vals if i % 2 == 0 else right_vals).append(f"{i}: {val}")
        y_start = pdf.get_y()
        pdf.set_xy(LM, y_start)
        for line in left_vals:
            _safe_multicell(pdf, col_w, 5, line)
        y_left = pdf.get_y()
        pdf.set_xy(LM + col_w + COL_GAP, y_start)
        for line in right_vals:
            _safe_multicell(pdf, col_w, 5, line)
        pdf.set_y(max(y_left, pdf.get_y()))

    # --- Optional plot data echo (for debugging / inspection) ---
    if plot_series:
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Series preview", ln=True)
        pdf.set_font("Arial", "", 10)
        for name, arr in plot_series.items():
            _safe_multicell(pdf, PAGE_W - LM - RM, 5, f"{name}: {arr[:30]}")

    # --- Interpretation section ---
    if interpretation:
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 7, "Interpretation", ln=True)
        pdf.set_font("Arial", "", 11)
        _safe_multicell(pdf, PAGE_W - LM - RM, 6, interpretation)

    # --- Output (handle fpdf / fpdf2 differences) ---
    try:
        data = pdf.output(dest="S").encode("latin-1")
    except AttributeError:
        data = pdf.output(dest="S").encode("utf-8")
    except TypeError:
        data = pdf.output(dest="S")

    # Ensure bytes for Streamlit download
    if isinstance(data, str):
        data = data.encode("latin-1", errors="ignore")
    return data
