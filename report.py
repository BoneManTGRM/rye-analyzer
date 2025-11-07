# report.py
# Build a compact, multi-page PDF report for RYE analyses.

from __future__ import annotations

import io
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    # Works for both fpdf and fpdf2, but import path is the same.
    from fpdf import FPDF
except Exception as e:
    raise RuntimeError("fpdf/fpdf2 is required. Add 'fpdf>=1.7.2' to requirements.txt") from e


# -----------------------------
# PDF helpers
# -----------------------------

class Report(FPDF):
    """FPDF/FPDF2-compatible report with margins, auto page breaks, and footer page numbers."""
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)
        # Enable total pages for footer
        try:
            self.alias_nb_pages()
        except Exception:
            pass

    def footer(self):
        # Called automatically at the bottom of each page
        self.set_y(-12)
        self.set_font("Arial", "I", 9)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}", align="R")

    # Convenience property for usable text width
    @property
    def content_w(self) -> float:
        return self.w - self.l_margin - self.r_margin


def _safe_set_font(pdf: Report, family="Arial", style="", size=11):
    """Use a built-in core font; fall back to Helvetica if Arial not available."""
    try:
        pdf.set_font(family, style, size)
    except Exception:
        pdf.set_font("Helvetica", style, size)


def _h1(pdf: Report, text: str):
    _safe_set_font(pdf, "Arial", "B", 18)
    pdf.cell(0, 10, text, ln=1)


def _h2(pdf: Report, text: str):
    _safe_set_font(pdf, "Arial", "B", 13)
    pdf.cell(0, 8, text, ln=1)


def _body(pdf: Report, size=11):
    _safe_set_font(pdf, "Arial", "", size)


def _add_logo(pdf: Report, path: str = "logo.png", w: float = 22.0):
    """Add a top-right logo if the file exists."""
    try:
        if os.path.exists(path):
            x = pdf.w - pdf.r_margin - w
            y = pdf.t_margin
            pdf.image(path, x=x, y=y, w=w)
    except Exception:
        # Non-fatal
        pass


def _key_val_rows(data: List[Tuple[str, Union[str, float, int]]],
                  key_w: float,
                  val_w: float,
                  key_style=("Arial", "B", 11),
                  val_style=("Arial", "", 11)) -> callable:
    """Return a renderer that prints aligned key:value rows within the given widths."""
    def render(pdf: Report):
        for k, v in data:
            # Keys
            try:
                pdf.set_font(*key_style)
            except Exception:
                _safe_set_font(pdf, "Arial", "B", 11)
            pdf.cell(key_w, 6, str(k), align="L")

            # Values
            try:
                pdf.set_font(*val_style)
            except Exception:
                _safe_set_font(pdf, "Arial", "", 11)
            text = str(v)
            # Multi-line value clipped to column width
            pdf.multi_cell(val_w, 6, text, align="L")
    return render


# -----------------------------
# Chart
# -----------------------------

def _add_series_plot(pdf: Report,
                     series_dict: Dict[str, Sequence[float]],
                     title: str = "Repair Yield per Energy"):
    """Render a simple line chart into the PDF without relying on seaborn."""
    # Build matplotlib figure
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=180)
    # We do not set colors explicitly to keep defaults (and avoid policy constraints).
    for label, series in series_dict.items():
        ax.plot(list(range(len(series))), list(series), label=label, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout(pad=0.7)

    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)

    # Place image in PDF using the content width
    try:
        pdf.image(buf, w=pdf.content_w)
    except TypeError:
        # Older fpdf expects a name-like parameter; fallback to a named buffer
        tmp = io.BytesIO(buf.getvalue())
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

    Parameters
    ----------
    rye : sequence of floats
        Primary RYE series to sample for the "first 100" block.
    summary : dict
        Summary statistics (mean, max, min, count, etc). Floats will be rounded here.
    metadata : dict
        Key metadata (e.g., rows, repair_col, energy_col, etc).
    plot_series : dict[str, sequence[float]], optional
        Labeled series for the line chart. Example:
        {"RYE": rye, "RYE (rolling)": rye_roll}
    interpretation : str, optional
        Analyst interpretation text. Auto-generated if not provided.
    logo_path : str, optional
        If provided and exists, draw in the header.
    """
    pdf = Report()
    pdf.add_page()
    if logo_path:
        _add_logo(pdf, logo_path, w=22)

    # Title
    _h1(pdf, "RYE Report")

    # Generated timestamp (UTC-like ISO if provided in metadata)
    generated = metadata.get("generated") or metadata.get("timestamp") or ""
    _body(pdf, 10)
    if generated:
        pdf.cell(0, 6, f"Generated: {generated}", ln=1)

    pdf.ln(2)

    # --- Dataset metadata
    _h2(pdf, "Dataset metadata")
    # Round floats and stringify
    meta_rows: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in metadata.items():
        if k in ("generated", "timestamp"):  # already shown above
            continue
        if isinstance(v, float):
            meta_rows.append((k, f"{v:.3f}"))
        else:
            meta_rows.append((k, v))
    # Two columns: keys left narrow, values wide
    kv = _key_val_rows(meta_rows, key_w=35, val_w=pdf.content_w - 35)
    kv(pdf)

    pdf.ln(2)

    # --- Summary statistics (rounded)
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

    # --- Sample values (first 100)
    _h2(pdf, "RYE sample values (first 100)")
    first_n = list(rye)[:100]
    # Two columns side-by-side, rounded to 4 decimals
    left = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 0]
    right = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 1]

    col_w = (pdf.content_w - 5) / 2  # 5mm gutter
    _body(pdf, 11)

    # Height accounting for the longer column
    max_lines = max(len(left), len(right))
    for idx in range(max_lines):
        ltxt = left[idx] if idx < len(left) else ""
        rtxt = right[idx] if idx < len(right) else ""
        y_before = pdf.get_y()

        # Left cell
        x_before = pdf.get_x()
        pdf.multi_cell(col_w, 6, ltxt, align="L")
        h_left = pdf.get_y() - y_before

        # Right cell at same row
        pdf.set_xy(x_before + col_w + 5, y_before)
        pdf.multi_cell(col_w, 6, rtxt, align="L")
        h_right = pdf.get_y() - y_before

        # Move down by the max height of the two multi-cells
        pdf.set_y(y_before + max(h_left, h_right))

    pdf.ln(2)

    # --- Interpretation
    if not interpretation:
        # Auto-generate a concise interpretation
        mean = float(summary.get("mean", 0.0) or 0.0)
        minv = float(summary.get("min", 0.0) or 0.0)
        maxv = float(summary.get("max", 0.0) or 0.0)
        level = "high" if mean > 0.6 else ("moderate" if mean > 0.3 else "modest")
        interpretation = (
            f"Average efficiency (RYE mean) is {mean:.3f}, within [{minv:.3f}, {maxv:.3f}]. "
            f"Overall efficiency is {level}. Look for low-yield segments to prune or repair. "
            "Use a rolling window to smooth short-term noise, map spikes/dips to events, "
            "and iterate TGRM loops (detect → minimal fix → verify) to lift average RYE "
            "and compress variance."
        )

    _h2(pdf, "Interpretation")
    _body(pdf, 11)
    pdf.multi_cell(0, 6, interpretation, align="L")

    pdf.ln(4)

    # --- Footer note (left edge, above PDF footer)
    _body(pdf, 10)
    pdf.cell(0, 6, "RYE.", ln=1)
    _body(pdf, 9)
    pdf.multi_cell(
        0, 5,
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics   RYE (Repair Yield per Energy)   TGRM.",
        align="L"
    )

    # Build byte-safe output for both fpdf and fpdf2
    try:
        data = pdf.output(dest="S").encode("latin-1")
    except AttributeError:
        # Some builds of fpdf2 already return bytes
        out = pdf.output(dest="S")
        data = out if isinstance(out, (bytes, bytearray)) else str(out).encode("utf-8")
    except TypeError:
        # Older fpdf returns bytes without encode
        data = pdf.output(dest="S")

    return data
