# report.py
# Build a compact, multi-page PDF report for RYE analyses.

from __future__ import annotations

import io
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    # Works for fpdf and fpdf2.
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
        try:
            self.alias_nb_pages()
        except Exception:
            pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "I", 9)
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
    pdf.cell(0, 10, text, ln=1)


def _h2(pdf: Report, text: str):
    _safe_set_font(pdf, "Arial", "B", 13)
    pdf.cell(0, 8, text, ln=1)


def _body(pdf: Report, size=11):
    _safe_set_font(pdf, "Arial", "", size)


def _add_logo(pdf: Report, path: str = "logo.png", w: float = 22.0):
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
            pdf.cell(key_w, 6, str(k), align="L")

            try:
                pdf.set_font(*val_style)
            except Exception:
                _safe_set_font(pdf, "Arial", "", 11)

            text = str(v)
            pdf.multi_cell(val_w, 6, text, align="L")
    return render


# -----------------------------
# Chart
# -----------------------------

def _add_series_plot(pdf: Report,
                     series_dict: Dict[str, Sequence[float]],
                     title: str = "Repair Yield per Energy"):
    """Render a simple line chart and embed using a real file path for fpdf compatibility."""
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=180)
    for label, series in series_dict.items():
        ax.plot(list(range(len(series))), list(series), label=label, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout(pad=0.7)

    # Write to a temp PNG so fpdf can infer the type from the extension.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=180)
        plt.close(fig)
        img_path = tmp.name

    try:
        pdf.image(img_path, w=pdf.content_w)
    finally:
        try:
            os.remove(img_path)
        except Exception:
            pass


# -----------------------------
# Main builder
# -----------------------------

def build_pdf(
    rye: Iterable[float],
    summary: Dict[str, Union[float, int, str]],
    # Accept both "meta" (used by app) and "metadata" (legacy)
    meta: Optional[Dict[str, Union[str, int, float]]] = None,
    metadata: Optional[Dict[str, Union[str, int, float]]] = None,
    plot_series: Optional[Dict[str, Sequence[float]]] = None,
    interpretation: Optional[str] = None,
    logo_path: Optional[str] = None,
    title: str = "RYE Report",
) -> bytes:
    """
    Build a multi-section PDF and return bytes.
    Compatible with calls like:
        build_pdf(rye, summary, title="RYE Report", meta=meta, plot_series=..., interpretation=...)
    """
    # Merge meta/metadata
    md = {}
    if metadata:
        md.update(metadata)
    if meta:
        md.update(meta)

    pdf = Report()
    pdf.add_page()
    if logo_path:
        _add_logo(pdf, logo_path, w=22)

    # Title
    _h1(pdf, title)

    # Generated timestamp (if provided)
    generated = md.get("generated") or md.get("timestamp") or ""
    _body(pdf, 10)
    if generated:
        pdf.cell(0, 6, f"Generated: {generated}", ln=1)

    pdf.ln(2)

    # --- Dataset metadata
    _h2(pdf, "Dataset metadata")
    meta_rows: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in md.items():
        if k in ("generated", "timestamp"):  # shown above
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

    # --- Sample values (first 100)
    _h2(pdf, "RYE sample values (first 100)")
    first_n = list(rye)[:100]
    left = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 0]
    right = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 1]

    col_w = (pdf.content_w - 5) / 2  # gutter = 5mm
    _body(pdf, 11)

    max_lines = max(len(left), len(right))
    for idx in range(max_lines):
        ltxt = left[idx] if idx < len(left) else ""
        rtxt = right[idx] if idx < len(right) else ""
        y_before = pdf.get_y()

        x_before = pdf.get_x()
        pdf.multi_cell(col_w, 6, ltxt, align="L")
        h_left = pdf.get_y() - y_before

        pdf.set_xy(x_before + col_w + 5, y_before)
        pdf.multi_cell(col_w, 6, rtxt, align="L")
        h_right = pdf.get_y() - y_before

        pdf.set_y(y_before + max(h_left, h_right))

    pdf.ln(2)

    # --- Interpretation
    if not interpretation:
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

    # --- Footer note
    _body(pdf, 10)
    pdf.cell(0, 6, "RYE.", ln=1)
    _body(pdf, 9)
    pdf.multi_cell(
        0, 5,
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics   RYE (Repair Yield per Energy)   TGRM.",
        align="L"
    )

    # Output bytes (latin-1 fallback for fpdf)
    try:
        data = pdf.output(dest="S").encode("latin-1")
    except AttributeError:
        out = pdf.output(dest="S")
        data = out if isinstance(out, (bytes, bytearray)) else str(out).encode("utf-8")
    except TypeError:
        data = pdf.output(dest="S")

    return data
