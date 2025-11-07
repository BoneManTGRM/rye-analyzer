# report.py
from fpdf import FPDF
from datetime import datetime
from typing import Dict, List, Optional
import io

# Matplotlib is used only to render the in-PDF chart
import matplotlib.pyplot as plt


MM = 1.0  # convenience

class Report(FPDF):
    """FPDF with sensible margins and auto page breaks."""
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        # consistent margins; leave room at bottom for footer
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)

    @property
    def content_w(self) -> float:
        """Printable width inside margins."""
        return self.w - self.l_margin - self.r_margin


def _safe_text(text: str) -> str:
    """
    FPDF base14 fonts are latin-1. Replace characters that can't be encoded.
    """
    if text is None:
        return ""
    try:
        text.encode("latin-1")
        return text
    except Exception:
        return text.encode("latin-1", "replace").decode("latin-1")


def _kv_block(pdf: Report, items: List[tuple], col_w: Optional[List[float]] = None, font_size: int = 11):
    """
    Render simple two-column key/value rows that never run off the page.
    Long values wrap using multi_cell.
    """
    if col_w is None:
        # 30% for key, 70% for value
        k = max(35.0, pdf.content_w * 0.32)
        v = pdf.content_w - k
        col_w = [k, v]

    pdf.set_font("Arial", size=font_size)
    for key, value in items:
        key = _safe_text(str(key))
        value = _safe_text("" if value is None else str(value))

        # key cell (no wrapping)
        y_before = pdf.get_y()
        pdf.set_font("Arial", "B", font_size)
        pdf.cell(col_w[0], 6, key, border=0)
        pdf.set_font("Arial", "", font_size)

        # value cell (wrapping)
        x_after_key = pdf.get_x()
        pdf.set_xy(x_after_key, y_before)
        pdf.multi_cell(col_w[1], 6, value, border=0, align="L")
    pdf.ln(1)


def _add_heading(pdf: Report, text: str, level: int = 1):
    size = 16 if level == 1 else 13
    style = "B" if level <= 2 else ""
    pdf.set_font("Arial", style, size)
    pdf.cell(0, 9, _safe_text(text), ln=True)
    pdf.ln(1)


def _add_paragraph(pdf: Report, text: str, size: int = 11):
    pdf.set_font("Arial", "", size)
    pdf.multi_cell(0, 6, _safe_text(text), align="L")
    pdf.ln(1)


def _add_series_plot(pdf: Report, series_dict: Dict[str, List[float]]):
    """
    Draw a simple line chart (RYE + optional rolling).
    """
    if not series_dict:
        return

    fig, ax = plt.subplots(figsize=(6.3, 2.6), dpi=200)  # wide & short for A4
    for label, arr in series_dict.items():
        if arr is None:
            continue
        ax.plot(list(range(len(arr))), arr, label=label)
    ax.set_title("Repair Yield per Energy")
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # image scaled to full text width
    pdf.image(buf, w=pdf.content_w)
    pdf.ln(2)


def build_pdf(
    rye_series: List[float],
    summary: Dict,
    meta: Dict,
    title: str = "RYE Report",
    plot_series: Optional[Dict[str, List[float]]] = None,
    interpretation: Optional[str] = None,
) -> bytes:
    """
    Create a portable multi-page PDF report.

    Parameters
    ----------
    rye_series : list of floats
        Computed RYE values (primary series).
    summary : dict
        Keys like mean, median, max, min, count. Values may be float/int/str.
    meta : dict
        Dataset metadata (e.g., rows, repair_col, energy_col, rolling_window).
    title : str
        Report title (shown in header).
    plot_series : dict[str, list[float]] | None
        Named series to plot, e.g., {"RYE": [...], "RYE (rolling)": [...] }.
    interpretation : str | None
        Auto/expert interpretation block appended at the end.
    """
    pdf = Report()
    pdf.add_page()

    # Header
    _add_heading(pdf, _safe_text(title), level=1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().isoformat()}Z", ln=True)
    pdf.ln(2)

    # Dataset metadata
    _add_heading(pdf, "Dataset metadata", level=2)
    meta_items = []
    for k in ["rows", "repair_col", "energy_col", "rolling_window"]:
        if k in meta:
            meta_items.append((k, meta[k]))
    # show any extra keys as well
    for k, v in meta.items():
        if k not in {"rows", "repair_col", "energy_col", "rolling_window"}:
            meta_items.append((k, v))
    _kv_block(pdf, meta_items)

    # Summary statistics
    _add_heading(pdf, "Summary statistics", level=2)
    # Order common metrics first, then any extras
    ordered = ["mean", "median", "max", "min", "count"]
    items = []
    for k in ordered:
        if k in summary:
            val = summary[k]
            if isinstance(val, float):
                val = f"{val:.6f}"
            items.append((k, val))
    for k, v in summary.items():
        if k not in ordered:
            val = f"{v:.6f}" if isinstance(v, float) else v
            items.append((k, val))
    _kv_block(pdf, items)

    # Chart (optional, but recommended)
    if plot_series is None:
        plot_series = {"RYE": rye_series}
    _add_series_plot(pdf, plot_series)

    # Sample values (first 100) in two columns, wrapped safely
    _add_heading(pdf, "RYE sample values (first 100)", level=2)
    pdf.set_font("Arial", "", 10)

    first_n = list(rye_series)[:100]
    left = []
    right = []
    for i, v in enumerate(first_n):
        s = f"{i}: {v:.12f}" if isinstance(v, float) else f"{i}: {v}"
        (left if i % 2 == 0 else right).append(s)

    # render two columns using multi_cell to avoid overflow
    col_w = pdf.content_w / 2 - 2
    max_rows = max(len(left), len(right))
    for i in range(max_rows):
        ltxt = left[i] if i < len(left) else ""
        rtxt = right[i] if i < len(right) else ""

        y = pdf.get_y()
        x = pdf.get_x()
        pdf.multi_cell(col_w, 5, _safe_text(ltxt), border=0, align="L")
        y2 = pdf.get_y()
        pdf.set_xy(x + col_w + 4, y)   # small gutter
        pdf.multi_cell(col_w, 5, _safe_text(rtxt), border=0, align="L")
        pdf.set_y(max(y2, pdf.get_y()))
    pdf.ln(2)

    # Interpretation (optional)
    if interpretation:
        _add_heading(pdf, "Interpretation", level=2)
        _add_paragraph(pdf, interpretation, size=11)

    # Footer
    pdf.ln(2)
    pdf.set_font("Arial", "I", 9)
    footer = (
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics  RYE (Repair Yield per Energy)  TGRM."
    )
    pdf.multi_cell(0, 5, _safe_text(footer), align="L")

    # --- Output (handle fpdf / fpdf2 differences safely) ---
    data = pdf.output(dest="S")
    if isinstance(data, str):
        # Some FPDF versions return str, others bytes.
        try:
            data = data.encode("latin-1")
        except Exception:
            data = data.encode("utf-8", errors="ignore")
    return data
