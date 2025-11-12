# report.py
# Build a compact, multi-page PDF report for RYE analyses
# (Unicode-safe, clickable links, tidy key/value layout, and robust wrapping)

from __future__ import annotations
import io, os, tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# --- Matplotlib (for inline plots) ---
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --- PDF (fpdf2 preferred; classic fpdf also works) ---
try:
    from fpdf import FPDF
except Exception as e:
    raise RuntimeError(
        "fpdf/fpdf2 is required. Add 'fpdf2>=2.7.9' (or 'fpdf>=1.7.2') to requirements.txt"
    ) from e


# =========================
# Unicode handling
# =========================
UNICODE_FONT_PATH = "fonts/DejaVuSans.ttf"   # add this file to your repo
UNICODE_FONT_NAME = "DejaVu"

def _strip_zw(text: str) -> str:
    """Remove zero-width and other invisible nasties that break Latin-1 fonts."""
    return text.replace("\u200b", "").replace("\ufeff", "")

def _sanitize(val: Union[str, float, int], unicode_ok: bool) -> str:
    """Coerce to str; if not using a Unicode font, force Latin-1 friendly text."""
    s = _strip_zw(str(val))
    if unicode_ok:
        return s
    # Latin-1 safe fallback (avoids exceptions when Helvetica is active)
    return s.encode("latin-1", "replace").decode("latin-1")


# =========================
# Small helpers
# =========================
def _fmt_val(v: Union[str, int, float]) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

def _normalize_doi_or_url(val: str) -> Optional[str]:
    """Accept '10.xxxx/...' or full URL; return clickable https URL."""
    if not val:
        return None
    s = str(val).strip()
    if s.startswith("10."):
        return f"https://doi.org/{s}"
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if "zenodo.org" in s and not s.startswith("http"):
        return f"https://{s}"
    return s

def _hyperlink(pdf: "Report", text: str, url: str):
    """Clickable link with underline; restores font and color after."""
    url = (url or "").strip()
    shown = _sanitize(text, pdf.unicode_ok)
    if not url:
        pdf.multi_cell(pdf.content_w, 6, shown, align="L")
        return
    pdf.set_text_color(0, 0, 200)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, "U", 11)
    else:
        pdf.set_font("Helvetica", "U", 11)
    pdf.multi_cell(pdf.content_w, 6, shown, align="L", link=url)
    pdf.set_text_color(0, 0, 0)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, "", 11)
    else:
        pdf.set_font("Helvetica", "", 11)


# =========================
# PDF scaffolding
# =========================
class Report(FPDF):
    """A4 portrait with margins, auto page breaks, and footer page numbers."""
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        # Slightly wider margins to avoid edge clipping on mobile viewers
        self.set_margins(14, 14, 14)
        self.set_auto_page_break(auto=True, margin=16)

        self.unicode_ok = False
        try:
            if os.path.exists(UNICODE_FONT_PATH):
                self.add_font(UNICODE_FONT_NAME, "", UNICODE_FONT_PATH, uni=True)
                self.unicode_ok = True
        except Exception:
            self.unicode_ok = False

        self.alias_nb_pages()

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
    _font(pdf, "B", 18)
    pdf.cell(0, 10, _sanitize(text, pdf.unicode_ok), ln=1)

def _h2(pdf: Report, text: str):
    _font(pdf, "B", 13)
    pdf.cell(0, 8, _sanitize(text, pdf.unicode_ok), ln=1)

def _body(pdf: Report, size: int = 11):
    _font(pdf, "", size)

def _add_logo(pdf: Report, path: str = "logo.png", w: float = 22.0):
    """Top-right logo if present (non-fatal if missing)."""
    try:
        if os.path.exists(path):
            x = pdf.w - pdf.r_margin - w
            y = pdf.t_margin
            pdf.image(path, x=x, y=y, w=w)
    except Exception:
        pass


# =========================
# Key/value rows (fixed layout)
# =========================
def _key_val_rows(
    data: List[Tuple[str, Union[str, float, int]]],
    key_w: float,
    val_w: float,
):
    """
    Render aligned key:value rows that never overflow the page.
    Prevents the 'preset/median drifting at far right' issue by:
      - fixing x/y for every row,
      - wrapping values within val_w,
      - resetting X to left margin each iteration.
    """
    def render(pdf: Report):
        for k, v in data:
            key_txt = _sanitize(k, pdf.unicode_ok)
            val_txt = _sanitize(v, pdf.unicode_ok)

            x0, y0 = pdf.get_x(), pdf.get_y()

            _font(pdf, "B", 11)
            pdf.cell(key_w, 6, key_txt, align="L")

            _body(pdf, 11)
            pdf.set_xy(x0 + key_w, y0)
            pdf.multi_cell(val_w, 6, val_txt, align="L")

            # Cursor to left margin for the next row
            pdf.set_x(pdf.l_margin)
    return render


# =========================
# Plots
# =========================
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


# =========================
# Main builder
# =========================
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

    - plot_series may be a single dict {label: series} or a list of such dicts.
    - metadata may include:
        'dataset_link' (or raw DOI),
        'generated' / 'timestamp',
        'sample_n' (default 100),
        'notes', 'columns' (list)
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
    # Render dataset link (clickable) if provided
    ds_link_raw = str(metadata.get("dataset_link", "") or "").strip()
    if ds_link_raw:
        url = _normalize_doi_or_url(ds_link_raw)
        if url:
            _hyperlink(pdf, f"Dataset link: {url}", url)

    # Everything else as key/value rows (skip fields we render elsewhere)
    kv_rows: List[Tuple[str, Union[str, float, int]]] = []
    for k, v in metadata.items():
        if k in ("generated", "timestamp", "dataset_link", "columns", "notes", "sample_n"):
            continue
        kv_rows.append((str(k), _fmt_val(v)))
    if kv_rows:
        _key_val_rows(kv_rows, key_w=40, val_w=pdf.content_w - 40)(pdf)
    pdf.ln(2)

    # Summary statistics
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
        ltxt = _sanitize(left[idx] if idx < len(left) else "", pdf.unicode_ok)
        rtxt = _sanitize(right[idx] if idx < len(right) else "", pdf.unicode_ok)
        y0 = pdf.get_y(); x0 = pdf.get_x()
        pdf.multi_cell(col_w, 6, ltxt, align="L")
        h_left = pdf.get_y() - y0
        pdf.set_xy(x0 + col_w + 5, y0)
        pdf.multi_cell(col_w, 6, rtxt, align="L")
        h_right = pdf.get_y() - y0
        pdf.set_y(y0 + max(h_left, h_right))
    pdf.ln(2)

    # Optional: Columns list
    cols = metadata.get("columns")
    if isinstance(cols, (list, tuple)) and len(cols) > 0:
        _h2(pdf, "Columns in dataset")
        _body(pdf, 11)
        pdf.multi_cell(pdf.content_w, 6, _sanitize(", ".join(map(str, cols)), pdf.unicode_ok), align="L")
        pdf.ln(1)

    # Interpretation paragraph
    if interpretation:
        _h2(pdf, "Interpretation")
        _body(pdf, 11)
        pdf.multi_cell(pdf.content_w, 6, _sanitize(interpretation, pdf.unicode_ok), align="L")
        pdf.ln(2)

    # Notes
    notes = metadata.get("notes")
    if notes:
        _h2(pdf, "Notes")
        _body(pdf, 11)
        pdf.multi_cell(pdf.content_w, 6, _sanitize(str(notes), pdf.unicode_ok), align="L")
        pdf.ln(2)

    # Footer attribution (kept brief)
    _body(pdf, 9)
    pdf.multi_cell(
        pdf.content_w,
        5,
        _sanitize("Open science by Cody Ryan Jenkins (CC BY 4.0). Terms: RYE (Repair Yield per Energy), TGRM.", pdf.unicode_ok),
        align="L",
    )

    # Output
    out = io.BytesIO()
    pdf.output(out)  # fpdf2 writes bytes into buffer
    return out.getvalue()
