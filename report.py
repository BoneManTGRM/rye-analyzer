# report.py
# Build a compact, multi-page PDF report for RYE analyses (fpdf/fpdf2 compatible).

from __future__ import annotations

import io
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    # Works for both fpdf and fpdf2.
    from fpdf import FPDF
except Exception as e:
    raise RuntimeError("fpdf/fpdf2 is required. Add 'fpdf>=1.7.2' to requirements.txt") from e


# -----------------------------
# Text / encoding helpers
# -----------------------------

_REPLACEMENTS = {
    # dashes
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
    # quotes
    "\u2018": "'", "\u2019": "'", "\u201A": ",",
    "\u201C": '"', "\u201D": '"', "\u201E": '"',
    # arrows / bullets / misc
    "\u2192": "->", "\u27A1": "->", "\u2022": "*", "\u00A0": " ",
}

def _to_pdf_text(x: Union[str, float, int]) -> str:
    """Convert to a Latin-1 safe string for FPDF while preserving readability."""
    s = str(x) if not isinstance(x, str) else x
    if not s:
        return ""
    # Replace common Unicode symbols with ASCII equivalents
    for u, a in _REPLACEMENTS.items():
        s = s.replace(u, a)
    # Ensure it can be encoded in latin-1; drop anything still unsupported
    return s.encode("latin-1", "ignore").decode("latin-1")


# -----------------------------
# PDF base
# -----------------------------

class Report(FPDF):
    """FPDF/FPDF2-compatible report with margins, auto page breaks, and footer page numbers."""
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)
        try:
            self.alias_nb_pages()
        except Exception:
            pass

    def footer(self):
        self.set_y(-12)
        _safe_set_font(self, "Arial", "I", 9)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="R")

    @property
    def content_w(self) -> float:
        return self.w - self.l_margin - self.r_margin


def _safe_set_font(pdf: Report, family="Arial", style="", size=11):
    try:
        pdf.set_font(family, style, size)
    except Exception:
        pdf.set_font("Helvetica", style, size)


def _h1(pdf: Report, text: str):
    _safe_set_font(pdf, "Arial", "B", 18)
    pdf.cell(0, 10, _to_pdf_text(text), ln=1)


def _h2(pdf: Report, text: str):
    _safe_set_font(pdf, "Arial", "B", 13)
    pdf.cell(0, 8, _to_pdf_text(text), ln=1)


def _body(pdf: Report, size=11):
    _safe_set_font(pdf, "Arial", "", size)


def _add_logo(pdf: Report, path: str = "logo.png", w: float = 22.0):
    """Add a top-right logo if the file exists (non-fatal if missing)."""
    try:
        if os.path.exists(path):
            x = pdf.w - pdf.r_margin - w
            y = pdf.t_margin
            pdf.image(path, x=x, y=y, w=w)
    except Exception:
        pass


def _key_val_rows(
    data: List[Tuple[str, Union[str, float, int]]],
    key_w: float,
    val_w: float,
    key_style=("Arial", "B", 11),
    val_style=("Arial", "", 11),
) -> callable:
    """Return a renderer that prints aligned key:value rows within the given widths."""
    def render(pdf: Report):
        for k, v in data:
            try:
                pdf.set_font(*key_style)
            except Exception:
                _safe_set_font(pdf, "Arial", "B", 11)
            pdf.cell(key_w, 6, _to_pdf_text(k), align="L")

            try:
                pdf.set_font(*val_style)
            except Exception:
                _safe_set_font(pdf, "Arial", "", 11)
            pdf.multi_cell(val_w, 6, _to_pdf_text(v), align="L")
    return render


# -----------------------------
# Chart
# -----------------------------

def _add_series_plot(
    pdf: Report,
    series_dict: Dict[str, Sequence[float]],
    title: str = "Repair Yield per Energy",
):
    """Render a simple line chart with matplotlib and embed as PNG."""
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=180)
    for label, series in series_dict.items():
        ax.plot(list(range(len(series))), list(series), label=_to_pdf_text(label), linewidth=1.8)

    ax.set_title(_to_pdf_text(title))
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout(pad=0.7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)

    # fpdf2 supports file-like objects; older fpdf prefers a name or BytesIO with .name
    try:
        pdf.image(buf, w=pdf.content_w)
    except TypeError:
        tmp = io.BytesIO(buf.getvalue())
        setattr(tmp, "name", "plot.png")  # hint a name for older backends
        pdf.image(tmp, w=pdf.content_w)


# -----------------------------
# Main builder
# -----------------------------

def build_pdf(
    rye: Iterable[float],
    summary: Dict[str, Union[float, int, str]],
    metadata: Dict[str, Union[str, int, float]],
    plot_series: Optional[Dict[str, Sequence[float]]] = None,
    interpretation: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> bytes:
    """
    Build a multi-section PDF and return bytes.
    """
    pdf = Report()
    pdf.add_page()
    if logo_path:
        _add_logo(pdf, logo_path, w=22)

    # Title
    _h1(pdf, "RYE Report")

    # Generated timestamp (if provided)
    _body(pdf, 10)
    generated = metadata.get("generated") or metadata.get("timestamp") or ""
    if generated:
        pdf.cell(0, 6, _to_pdf_text(f"Generated: {generated}"), ln=1)

    pdf.ln(2)

    # --- Dataset metadata
    _h2(pdf, "Dataset metadata")
    meta_rows: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in metadata.items():
        if k in ("generated", "timestamp"):
            continue
        if isinstance(v, float):
            meta_rows.append((k, f"{v:.3f}"))
        else:
            meta_rows.append((k, v))
    kv = _key_val_rows(meta_rows, key_w=35, val_w=pdf.content_w - 35)
    kv(pdf)

    pdf.ln(2)

    # --- Summary statistics
    _h2(pdf, "Summary statistics")
    _body(pdf, 11)
    items: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in summary.items():
        if isinstance(v, float):
            items.append((k, f"{v:.3f}"))
        else:
            items.append((k, v))
    kv2 = _key_val_rows(items, key_w=35, val_w=pdf.content_w - 35)
    kv2(pdf)

    pdf.ln(2)

    # --- Plot
    if plot_series:
        _add_series_plot(pdf, plot_series)

    pdf.ln(2)

    # --- Sample values (first 100), two columns
    _h2(pdf, "RYE sample values (first 100)")
    first_n = list(rye)[:100]
    left = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 0]
    right = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 1]

    col_w = (pdf.content_w - 5) / 2  # 5 mm gutter
    _body(pdf, 11)
    max_lines = max(len(left), len(right))
    for idx in range(max_lines):
        ltxt = _to_pdf_text(left[idx]) if idx < len(left) else ""
        rtxt = _to_pdf_text(right[idx]) if idx < len(right) else ""
        y0 = pdf.get_y()
        x0 = pdf.get_x()

        pdf.multi_cell(col_w, 6, ltxt, align="L")
        h_left = pdf.get_y() - y0

        pdf.set_xy(x0 + col_w + 5, y0)
        pdf.multi_cell(col_w, 6, rtxt, align="L")
        h_right = pdf.get_y() - y0

        pdf.set_y(y0 + max(h_left, h_right))

    pdf.ln(2)

    # --- Interpretation (auto if not provided)
    if not interpretation:
        mean = float(summary.get("mean", 0.0) or 0.0)
        minv = float(summary.get("min", 0.0) or 0.0)
        maxv = float(summary.get("max", 0.0) or 0.0)
        level = "high" if mean > 0.6 else ("moderate" if mean > 0.3 else "modest")
        interpretation = (
            f"Average efficiency (RYE mean) is {mean:.3f}, within [{minv:.3f}, {maxv:.3f}]. "
            f"Overall efficiency is {level}. Look for low-yield segments to prune or repair. "
            "Use a rolling window to smooth short-term noise, map spikes/dips to events, "
            "and iterate TGRM loops (detect -> minimal fix -> verify) to lift average RYE "
            "and compress variance."
        )

    _h2(pdf, "Interpretation")
    _body(pdf, 11)
    pdf.multi_cell(0, 6, _to_pdf_text(interpretation), align="L")

    pdf.ln(4)

    # --- Footer note
    _body(pdf, 10)
    pdf.cell(0, 6, _to_pdf_text("RYE."), ln=1)
    _body(pdf, 9)
    pdf.multi_cell(
        0, 5,
        _to_pdf_text(
            "Open science by Cody Ryan Jenkins (CC BY 4.0). "
            "Learn more: Reparodynamics   RYE (Repair Yield per Energy)   TGRM."
        ),
        align="L"
    )

    # --- Output as bytes with safe fallbacks
    try:
        out = pdf.output(dest="S")
        if isinstance(out, (bytes, bytearray)):
            data = out
        else:
            # Prefer latin-1 with replacement so fpdf stays happy; fall back to utf-8.
            try:
                data = out.encode("latin-1", "replace")
            except Exception:
                data = out.encode("utf-8", "replace")
    except Exception:
        data = pdf.output(dest="S").encode("utf-8", "replace")

    return data
