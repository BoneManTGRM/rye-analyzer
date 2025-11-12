# report.py
# Build a compact, multi-page PDF report for RYE analyses
# (Unicode-safe, clickable links, safe wrapping, richer metadata).

from __future__ import annotations
import os, tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    from fpdf import FPDF  # fpdf2
except Exception as e:
    raise RuntimeError("fpdf2 is required. Add 'fpdf2>=2.7.9' to requirements.txt") from e

# -----------------------------
# Unicode handling
# -----------------------------
UNICODE_FONT_PATH = "fonts/DejaVuSans.ttf"
UNICODE_FONT_NAME = "DejaVu"

def _sanitize(txt: Union[str, float, int], unicode_ok: bool) -> str:
    """
    Convert any value to str. If no Unicode font is loaded, keep text Latin-1 safe
    and strip zero-width characters that core fonts can't encode.
    """
    s = str(txt)
    if not unicode_ok:
        # remove zero-width chars that break core fonts
        s = s.replace("\u200b", "")
        # make Latin-1 safe
        s = s.encode("latin-1", "replace").decode("latin-1")
    return s

# -----------------------------
# Utilities
# -----------------------------
def _fmt_val(v: Union[str, int, float]) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

def _normalize_doi_or_url(val: str) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip()
    if s.startswith("10."):
        return f"https://doi.org/{s}"
    if s.startswith(("http://", "https://")):
        return s
    if "zenodo.org" in s and not s.startswith("http"):
        return f"https://{s}"
    return s

def _soft_wrap_token(s: str, allow_zwb: bool) -> str:
    """
    Insert break opportunities in long tokens (URLs, ids).
    If a Unicode font is available, use zero-width breaks; otherwise insert spaces.
    """
    if allow_zwb:
        s = s.replace("/", "/\u200b")
        s = s.replace("?", "?\u200b")
        s = s.replace("&", "&\u200b")
        s = s.replace("=", "=\u200b")
        s = s.replace("-", "-\u200b")
        s = s.replace("_", "_\u200b")
        s = s.replace(".", ".\u200b")
    else:
        s = s.replace("/", "/ ")
        s = s.replace("?", "? ")
        s = s.replace("&", "& ")
        s = s.replace("=", "= ")
        s = s.replace("-", "- ")
        s = s.replace("_", "_ ")
        s = s.replace(".", ". ")
    return s

def _hyperlink(pdf: "Report", display_text: str, url: str):
    url = (url or "").strip()
    txt = _sanitize(display_text, pdf.unicode_ok)
    if not url:
        pdf.multi_cell(0, 6, txt, align="L")
        return
    pdf.set_text_color(0, 0, 200)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, "U", 11)
    else:
        pdf.set_font("Helvetica", "U", 11)
    pdf.multi_cell(0, 6, txt, align="L", link=url)
    pdf.set_text_color(0, 0, 0)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, "", 11)
    else:
        pdf.set_font("Helvetica", "", 11)

# -----------------------------
# PDF helpers
# -----------------------------
class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_text_color(0, 0, 0)

        self.unicode_ok = False
        try:
            if os.path.exists(UNICODE_FONT_PATH):
                self.add_font(UNICODE_FONT_NAME, "", UNICODE_FONT_PATH, uni=True)
                self.unicode_ok = True
            self.alias_nb_pages()
        except Exception:
            self.unicode_ok = False

    def footer(self):
        self.set_y(-12)
        self.set_text_color(0, 0, 0)
        if self.unicode_ok:
            self.set_font(UNICODE_FONT_NAME, "", 9)
        else:
            self.set_font("Helvetica", "I", 9)
        self.cell(0, 6, _sanitize(f"Page {self.page_no()}/{{nb}}", self.unicode_ok), align="R")

    @property
    def content_w(self) -> float:
        return self.w - self.l_margin - self.r_margin

def _font(pdf: Report, style: str = "", size: int = 11):
    pdf.set_text_color(0, 0, 0)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, style, size)
    else:
        pdf.set_font("Helvetica", style, size)

def _h1(pdf: Report, text: str):
    _font(pdf, "B", 18); pdf.cell(0, 10, _sanitize(text, pdf.unicode_ok), ln=1)

def _h2(pdf: Report, text: str):
    _font(pdf, "B", 13); pdf.cell(0, 8, _sanitize(text, pdf.unicode_ok), ln=1)

def _body(pdf: Report, size: int = 11):
    _font(pdf, "", size)

def _add_logo(pdf: Report, path: str = "logo.png", w: float = 22.0):
    try:
        if os.path.exists(path):
            x = pdf.w - pdf.r_margin - w
            y = pdf.t_margin
            pdf.image(path, x=x, y=y, w=w)
    except Exception:
        pass

def _key_val_rows(data: List[Tuple[str, Union[str, float, int]]], key_w: float, val_w: float):
    def render(pdf: Report):
        for k, v in data:
            key = _sanitize(str(k), pdf.unicode_ok)
            val = _sanitize(str(v), pdf.unicode_ok)
            val = _soft_wrap_token(val, allow_zwb=pdf.unicode_ok)
            _font(pdf, "B", 11); pdf.cell(key_w, 6, key, align="L")
            _body(pdf, 11); pdf.multi_cell(val_w, 6, val, align="L")
    return render

# -----------------------------
# Chart
# -----------------------------
def _add_series_plot(pdf: Report, series_dict: Dict[str, Sequence[float]], title: str = "Repair Yield per Energy"):
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=180)
    for label, series in series_dict.items():
        ax.plot(range(len(series)), list(series), label=label, linewidth=1.8)
    ax.set_title(title); ax.set_xlabel("Index"); ax.set_ylabel("RYE")
    ax.legend(loc="best", frameon=False); ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout(pad=0.7)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=180)
    finally:
        plt.close(fig)
    try:
        pdf.image(tmp.name, w=pdf.content_w)
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass

# -----------------------------
# Main builder
# -----------------------------
def build_pdf(
    rye: Iterable[float],
    summary: Dict[str, Union[float, int, str]],
    metadata: Dict[str, Union[str, int, float]],
    plot_series: Optional[Union[Dict[str, Sequence[float]], List[Dict[str, Sequence[float]]]]] = None,
    interpretation: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> bytes:
    pdf = Report()
    pdf.add_page()
    if logo_path: _add_logo(pdf, logo_path, w=22)

    _h1(pdf, "RYE Report")
    generated = metadata.get("generated") or metadata.get("timestamp") or ""
    _body(pdf, 10)
    if generated:
        pdf.multi_cell(0, 6, _sanitize(f"Generated: {generated}", pdf.unicode_ok), align="L")
    pdf.ln(2)

    # Dataset metadata
    _h2(pdf, "Dataset metadata")
    meta_rows: List[Tuple[str, Union[str, float, int]]] = []

    ds_link_raw = str(metadata.get("dataset_link", "") or "").strip()
    if ds_link_raw:
        url = _normalize_doi_or_url(ds_link_raw)
        if url:
            _body(pdf, 11)
            _hyperlink(pdf, "Dataset link (tap to open)", url)

    for k, v in metadata.items():
        if k in ("generated", "timestamp", "dataset_link", "columns", "notes", "sample_n"):
            continue
        meta_rows.append((str(k), _fmt_val(v)))
    if meta_rows:
        _key_val_rows(meta_rows, key_w=42, val_w=pdf.content_w - 42)(pdf)
    pdf.ln(2)

    # Summary statistics
    _h2(pdf, "Summary statistics")
    items: List[Tuple[str, Union[str, float, int]]] = [(str(k), _fmt_val(v)) for k, v in summary.items()]
    _key_val_rows(items, key_w=42, val_w=pdf.content_w - 42)(pdf)
    pdf.ln(2)

    # Plots
    if plot_series:
        if isinstance(plot_series, dict):
            _add_series_plot(pdf, plot_series)
        elif isinstance(plot_series, list):
            for ps in plot_series:
                if isinstance(ps, dict):
                    _add_series_plot(pdf, ps)
        pdf.ln(2)

    # Sample values
    n_show = int(metadata.get("sample_n", 100) or 100)
    _h2(pdf, f"RYE sample values (first {n_show})")
    first_n = list(rye)[:n_show]
    left  = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 0]
    right = [f"{i}: {v:.4f}" for i, v in enumerate(first_n) if i % 2 == 1]
    col_w = (pdf.content_w - 6) / 2
    _body(pdf, 11)
    max_lines = max(len(left), len(right))
    for idx in range(max_lines):
        ltxt = left[idx] if idx < len(left) else ""
        rtxt = right[idx] if idx < len(right) else ""
        y0 = pdf.get_y(); x0 = pdf.get_x()
        pdf.multi_cell(col_w, 6, _sanitize(ltxt, pdf.unicode_ok), align="L")
        h_left = pdf.get_y() - y0
        pdf.set_xy(x0 + col_w + 6, y0)
        pdf.multi_cell(col_w, 6, _sanitize(rtxt, pdf.unicode_ok), align="L")
        h_right = pdf.get_y() - y0
        pdf.set_y(y0 + max(h_left, h_right))
    pdf.ln(2)

    # Columns list
    cols = metadata.get("columns")
    if isinstance(cols, (list, tuple)) and len(cols) > 0:
        _h2(pdf, "Columns in dataset")
        cols_text = _soft_wrap_token(", ".join(map(str, cols)), allow_zwb=pdf.unicode_ok)
        _body(pdf, 11)
        pdf.multi_cell(0, 6, _sanitize(cols_text, pdf.unicode_ok), align="L")
        pdf.ln(1)

    # Interpretation
    if not interpretation:
        mean = float(summary.get("mean", 0.0) or 0.0)
        minv = float(summary.get("min", 0.0) or 0.0)
        maxv = float(summary.get("max", 0.0) or 0.0)
        level = "high" if mean > 0.6 else ("moderate" if mean > 0.3 else "modest")
        interpretation = (
            f"Average efficiency (RYE mean) is {mean:.3f}, within [{minv:.3f}, {maxv:.3f}]. "
            f"Overall efficiency is {level}. Look for low-yield segments to prune or repair. "
            "Use a rolling window to smooth short-term noise, map spikes/dips to events, and "
            "iterate TGRM loops (detect -> minimal fix -> verify) to raise average RYE."
        )
    _h2(pdf, "Interpretation")
    _body(pdf, 11)
    pdf.multi_cell(0, 6, _sanitize(interpretation.replace("—", "-"), pdf.unicode_ok), align="L")
    pdf.ln(4)

    # Notes (optional)
    notes = metadata.get("notes")
    if isinstance(notes, str) and notes.strip():
        _h2(pdf, "Notes")
        _body(pdf, 11)
        pdf.multi_cell(0, 6, _sanitize(notes, pdf.unicode_ok), align="L")
        pdf.ln(2)

    # Footer note
    _body(pdf, 10); pdf.cell(0, 6, _sanitize("RYE.", pdf.unicode_ok), ln=1)
    _body(pdf, 9)
    pdf.multi_cell(0, 5, _sanitize(
        "Open science by Cody Ryan Jenkins (CC BY 4.0). Learn more: Reparodynamics, RYE, TGRM.",
        pdf.unicode_ok
    ), align="L")

    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytes, bytearray)) else str(out).encode("latin-1", "replace")
