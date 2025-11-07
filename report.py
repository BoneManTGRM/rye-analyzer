# report.py
from fpdf import FPDF
from datetime import datetime
from typing import Dict, List, Optional
import io
import os
import tempfile
import matplotlib.pyplot as plt

class Report(FPDF):
    """FPDF with proper margins and page breaks"""
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)

    @property
    def content_w(self):
        return self.w - self.l_margin - self.r_margin


def _safe_text(text):
    if text is None:
        return ""
    try:
        text.encode("latin-1")
        return text
    except Exception:
        return text.encode("latin-1", "replace").decode("latin-1")


def _kv_block(pdf, items, col_w=None, font_size=11):
    if col_w is None:
        k = max(35.0, pdf.content_w * 0.32)
        v = pdf.content_w - k
        col_w = [k, v]

    pdf.set_font("Arial", size=font_size)
    for key, value in items:
        key = _safe_text(str(key))
        value = _safe_text("" if value is None else str(value))

        y_before = pdf.get_y()
        pdf.set_font("Arial", "B", font_size)
        pdf.cell(col_w[0], 6, key, border=0)
        pdf.set_font("Arial", "", font_size)
        x_after_key = pdf.get_x()
        pdf.set_xy(x_after_key, y_before)
        pdf.multi_cell(col_w[1], 6, value, border=0, align="L")
    pdf.ln(1)


def _add_heading(pdf, text, level=1):
    size = 16 if level == 1 else 13
    style = "B" if level <= 2 else ""
    pdf.set_font("Arial", style, size)
    pdf.cell(0, 9, _safe_text(text), ln=True)
    pdf.ln(1)


def _add_paragraph(pdf, text, size=11):
    pdf.set_font("Arial", "", size)
    pdf.multi_cell(0, 6, _safe_text(text), align="L")
    pdf.ln(1)


def _add_series_plot(pdf, series_dict):
    """Save chart to a temp file so fpdf 1.x can load it"""
    if not series_dict:
        return

    fig, ax = plt.subplots(figsize=(6.3, 2.6), dpi=200)
    for label, arr in series_dict.items():
        if arr is None:
            continue
        ax.plot(list(range(len(arr))), arr, label=label)
    ax.set_title("Repair Yield per Energy")
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()

    tmp_path = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp_path, format="png", bbox_inches="tight")
    plt.close(fig)

    pdf.image(tmp_path, w=pdf.content_w)
    pdf.ln(2)

    os.remove(tmp_path)


def build_pdf(
    rye_series: List[float],
    summary: Dict,
    meta: Dict,
    title: str = "RYE Report",
    plot_series: Optional[Dict[str, List[float]]] = None,
    interpretation: Optional[str] = None,
) -> bytes:
    pdf = Report()
    pdf.add_page()

    # Header
    _add_heading(pdf, title, level=1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().isoformat()}Z", ln=True)
    pdf.ln(2)

    # Dataset metadata
    _add_heading(pdf, "Dataset metadata", level=2)
    meta_items = [(k, meta[k]) for k in meta]
    _kv_block(pdf, meta_items)

    # Summary stats
    _add_heading(pdf, "Summary statistics", level=2)
    items = []
    for k, v in summary.items():
        val = f"{v:.6f}" if isinstance(v, float) else v
        items.append((k, val))
    _kv_block(pdf, items)

    # Chart
    if plot_series is None:
        plot_series = {"RYE": rye_series}
    _add_series_plot(pdf, plot_series)

    # Sample values (first 100)
    _add_heading(pdf, "RYE sample values (first 100)", level=2)
    pdf.set_font("Arial", "", 10)
    first_n = list(rye_series)[:100]
    col_w = pdf.content_w / 2 - 2
    left = [f"{i}: {v:.12f}" for i, v in enumerate(first_n) if i % 2 == 0]
    right = [f"{i}: {v:.12f}" for i, v in enumerate(first_n) if i % 2 == 1]

    max_rows = max(len(left), len(right))
    for i in range(max_rows):
        ltxt = left[i] if i < len(left) else ""
        rtxt = right[i] if i < len(right) else ""
        y = pdf.get_y()
        x = pdf.get_x()
        pdf.multi_cell(col_w, 5, _safe_text(ltxt), border=0, align="L")
        y2 = pdf.get_y()
        pdf.set_xy(x + col_w + 4, y)
        pdf.multi_cell(col_w, 5, _safe_text(rtxt), border=0, align="L")
        pdf.set_y(max(y2, pdf.get_y()))
    pdf.ln(2)

    # Interpretation
    if interpretation:
        _add_heading(pdf, "Interpretation", level=2)
        _add_paragraph(pdf, interpretation)

    # Footer
    pdf.ln(2)
    pdf.set_font("Arial", "I", 9)
    footer = (
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics  RYE (Repair Yield per Energy)  TGRM."
    )
    pdf.multi_cell(0, 5, _safe_text(footer), align="L")

    # Output (safe for Streamlit Cloud)
    data = pdf.output(dest="S")
    if isinstance(data, str):
        try:
            data = data.encode("latin-1")
        except Exception:
            data = data.encode("utf-8", errors="ignore")
    return data
