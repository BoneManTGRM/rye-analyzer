# report.py
# Build a compact, multi-page PDF report for RYE analyses (Unicode-safe, clickable links, richer metadata).

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
# Unicode handling
# -----------------------------
UNICODE_FONT_PATH = "fonts/DejaVuSans.ttf"
UNICODE_FONT_NAME = "DejaVu"

def _sanitize(txt: Union[str, float, int], unicode_ok: bool) -> str:
    """
    Convert any value to str. If we don't have a Unicode font loaded,
    replace characters outside Latin-1 so classic Helvetica won't crash.
    """
    s = str(txt)
    if unicode_ok:
        return s
    return s.encode("latin-1", "replace").decode("latin-1")


# -----------------------------
# Small utilities
# -----------------------------
def _fmt_val(v: Union[str, int, float]) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

def _normalize_doi_or_url(val: str) -> Optional[str]:
    """
    Accepts a DOI (e.g., '10.5281/zenodo.12345') or a URL, returns a clickable https URL.
    """
    if not val:
        return None
    s = str(val).strip()
    if s.startswith("10."):
        return f"https://doi.org/{s}"
    if s.startswith("http://") or s.startswith("https://"):
        return s
    # mild heuristic: user pasted zenodo record id
    if "zenodo.org" in s and not s.startswith("http"):
        return f"https://{s}"
    return s

def _hyperlink(pdf: "Report", text: str, url: str):
    """
    Render a clickable link (blue + underline if unicode font available).
    """
    url = url.strip()
    if not url:
        pdf.multi_cell(0, 6, _sanitize(text, pdf.unicode_ok), align="L")
        return
    # choose font/underline style
    if pdf.unicode_ok:
        pdf.set_text_color(0, 0, 200)
        pdf.set_font(UNICODE_FONT_NAME, "U", 11)
    else:
        pdf.set_text_color(0, 0, 200)
        pdf.set_font("Helvetica", "U", 11)
    pdf.multi_cell(0, 6, _sanitize(text, pdf.unicode_ok), align="L", link=url)
    # restore defaults
    pdf.set_text_color(0, 0, 0)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, "", 11)
    else:
        pdf.set_font("Helvetica", "", 11)


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
    """Use Unicode font if available; otherwise Helvetica. Always reset text color to black."""
    pdf.set_text_color(0, 0, 0)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, style, size)
    else:
        pdf.set_font("Helvetica", style, size)

def _h1(pdf: Report, text: str):
    _font(pdf, "B", 18)
    pdf.cell(0, 10, _sanitize(text, pdf.unicode_ok), ln=1)

def _h2(pdf: Report, text: str):
    _font(pdf, "B", 13)
    pdf.cell(0, 8, _sanitize(text, pdf.unicode_ok), ln=1)

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
            pdf.cell(key_w, 6, _sanitize(k, pdf.unicode_ok), align="L")
            _body(pdf, 11)
            pdf.multi_cell(val_w, 6, _sanitize(v, pdf.unicode_ok), align="L")
    return render


# -----------------------------
# Charts
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
    plot_series: Optional[Union[Dict[str, Sequence[float]], List[Dict[str, Sequence[float]]]]] = None,
    interpretation: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> bytes:
    """
    Build a multi-section PDF and return bytes.

    Backward-compatible:
      - plot_series may be a single dict {label: series} or a list of such dicts.
      - metadata may include:
          - 'dataset_link' or raw DOI (auto-normalized and made clickable)
          - 'generated' or 'timestamp'
          - 'sample_n' (int, default 100)
          - 'notes' (str)
          - 'columns' (list of column names)
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
        pdf.cell(0, 6, _sanitize(f"Generated: {generated}", pdf.unicode_ok), ln=1)
    pdf.ln(2)

    # Dataset metadata
    _h2(pdf, "Dataset metadata")
    meta_rows: List[Tuple[str, Union[str, float, int]]] = []

    # render link if provided
    ds_link_raw = str(metadata.get("dataset_link", "") or "").strip()
    if ds_link_raw:
        url = _normalize_doi_or_url(ds_link_raw)
        if url:
            _body(pdf, 11)
            _hyperlink(pdf, f"Dataset link: {url}", url)

    # add other metadata (except special keys we render separately)
    for k, v in metadata.items():
        if k in ("generated", "timestamp", "dataset_link", "columns", "notes", "sample_n"):
            continue
        meta_rows.append((str(k), _fmt_val(v)))
    if meta_rows:
        _key_val_rows(meta_rows, key_w=40, val_w=pdf.content_w - 40)(pdf)
    pdf.ln(2)

    # Summary statistics (includes anything you pass, e.g., 'resilience')
    _h2(pdf, "Summary statistics")
    items: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in summary.items():
        items.append((str(k), _fmt_val(v)))
    _key_val_rows(items, key_w=40, val_w=pdf.content_w - 40)(pdf)
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
    col_w = (pdf.content_w - 5) / 2
    _body(pdf, 11)
    max_lines = max(len(left), len(right))
    for idx in range(max_lines):
        ltxt = left[idx] if idx < len(left) else ""
        rtxt = right[idx] if idx < len(right) else ""
        y0 = pdf.get_y(); x0 = pdf.get_x()
        pdf.multi_cell(col_w, 6, _sanitize(ltxt, pdf.unicode_ok), align="L")
        h_left = pdf.get_y() - y0
        pdf.set_xy(x0 + col_w + 5, y0)
        pdf.multi_cell(col_w, 6, _sanitize(rtxt, pdf.unicode_ok), align="L")
        h_right = pdf.get_y() - y0
        pdf.set_y(y0 + max(h_left, h_right))
    pdf.ln(2)

    # Optional: Columns list
    cols = metadata.get("columns")
    if isinstance(cols, (list, tuple)) and len(cols) > 0:
        _h2(pdf, "Columns in dataset")
        _body(pdf, 11
