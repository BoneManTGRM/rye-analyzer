# report.py
# Build a compact, multi-page PDF report for RYE analyses.

from __future__ import annotations
import io, os, tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    from fpdf import FPDF  # works for fpdf and fpdf2
except Exception as e:
    raise RuntimeError("fpdf/fpdf2 is required. Add 'fpdf>=1.7.2' to requirements.txt") from e


# -----------------------------
# PDF helpers
# -----------------------------

class Report(FPDF):
    """A4 portrait with margins, auto page breaks, and footer page numbers."""
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_text_color(0, 0, 0)
        try:
            self.alias_nb_pages()
        except Exception:
            pass

    def footer(self):
        self.set_y(-12)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="R")

    @property
    def content_w(self) -> float:
        return self.w - self.l_margin - self.r_margin


def _font(pdf: Report, style: str = "", size: int = 11):
    """Always use a core font and reset text color to black."""
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", style, size)


def _h1(pdf: Report, text: str):
    _font(pdf, "B", 18)
    pdf.cell(0, 10, text, ln=1)


def _h2(pdf: Report, text: str):
    _font(pdf, "B", 13)
    pdf.cell(0, 8, text, ln=1)


def _body(pdf: Report, size: int = 11):
    _font(pdf, "", size)


def _add_logo(pdf: Report, path: str = "logo.png", w: float = 22.0):
    """Add a top-right logo if present (non-fatal if missing)."""
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
):
    """Renderer that prints aligned key:value rows within the given widths."""
    def render(pdf: Report):
        for k, v in data:
            _font(pdf, "B", 11)
            pdf.cell(key_w, 6, str(k), align="L")
            _body(pdf, 11)
            pdf.multi_cell(val_w, 6, str(v), align="L")
    return render


# -----------------------------
# Chart
# -----------------------------

def _add_series_plot(pdf: Report,
                     series_dict: Dict[str, Sequence[float]],
                     title: str = "Repair Yield per Energy"):
    """Render a simple line chart into the PDF using matplotlib."""
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=180)
    for label, series in series_dict.items():
        ax.plot(range(len(series)), list(series), label=label, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout(pad=0.7)

    # Save to a real temporary file (fpdf is happiest with paths)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=180)
    finally:
        plt.close(fig)

    try:
        pdf.image(tmp.name, w=pdf.content_w)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


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

    # Title + timestamp
    _h1(pdf, "RYE Report")
    generated = metadata.get("generated") or metadata.get("timestamp") or ""
    _body(pdf, 10)
    if generated:
        pdf.cell(0, 6, f"Generated: {generated}", ln=1)
    pdf.ln(2)

    # Dataset metadata
    _h2(pdf, "Dataset metadata")
    meta_rows: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in metadata.items():
        if k in ("generated", "timestamp"):
            continue
        meta_rows.append((k, f"{v:.3f}" if isinstance(v, float) else v))
    _key_val_rows(meta_rows, key_w=35, val_w=pdf.content_w - 35)(pdf)
    pdf.ln(2)

    # Summary statistics
    _h2(pdf, "Summary statistics")
    items: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in summary.items():
        items.append((k, f"{v:.3f}" if isinstance(v, float) else v))
    _key_val_rows(items, key_w=35, val_w=pdf.content_w - 35)(pdf)
    pdf.ln(2)

    # Plot
    if plot_series:
        _add_series_plot(pdf, plot_series)
    pdf.ln(2)

    # Sample values
    _h2(pdf, "RYE sample values (first 100)")
    first_n = list(rye)[:100]
    left  = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 0]
    right = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 1]
    col_w = (pdf.content_w - 5) / 2
    _body(pdf, 11)
    max_lines = max(len(left), len(right))
    for idx in range(max_lines):
        ltxt = left[idx] if idx < len(left) else ""
        rtxt = right[idx] if idx < len(right) else ""
        y0 = pdf.get_y(); x0 = pdf.get_x()
        pdf.multi_cell(col_w, 6, ltxt, align="L")
        h_left = pdf.get_y() - y0
        pdf.set_xy(x0 + col_w + 5, y0)
        pdf.multi_cell(col_w, 6, rtxt, align="L")
        h_right = pdf.get_y() - y0
        pdf.set_y(y0 + max(h_left, h_right))
    pdf.ln(2)

    # Interpretation (ASCII-safe arrow)
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
    pdf.multi_cell(0, 6, interpretation, align="L")
    pdf.ln(4)

    # Footer note
    _body(pdf, 10); pdf.cell(0, 6, "RYE.", ln=1)
    _body(pdf, 9)
    pdf.multi_cell(
        0, 5,
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics   RYE (Repair Yield per Energy)   TGRM.",
        align="L"
    )

    # Return exact bytes. For fpdf2, this is already bytes.
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    # Classic fpdf returns a Latin-1 string representing the PDF bytes.
    return str(out).encode("latin-1", errors="replace")
