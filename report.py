# report.py
# Build a compact, multi-page PDF report for RYE analyses.
# Unicode-safe (NotoSans/DejaVu), safe links, clean wrapping.

from __future__ import annotations
import os, tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from fpdf import FPDF  # fpdf2
except Exception as e:
    raise RuntimeError("fpdf2 is required. Add 'fpdf2>=2.7.9' to requirements.txt") from e

# ---------- Fonts ----------
FONT_DIR = "fonts"
NOTO_PATH = os.path.join(FONT_DIR, "NotoSans-Regular.ttf")
DEJAVU_PATH = os.path.join(FONT_DIR, "DejaVuSans.ttf")

UNICODE_FONT_PATH: Optional[str] = None
UNICODE_FONT_NAME: Optional[str] = None

def _pick_local_font() -> Optional[Tuple[str, str]]:
    if os.path.exists(NOTO_PATH):
        return ("NotoSans", NOTO_PATH)
    if os.path.exists(DEJAVU_PATH):
        return ("DejaVu", DEJAVU_PATH)
    return None

def _resolve_font() -> None:
    global UNICODE_FONT_NAME, UNICODE_FONT_PATH
    local = _pick_local_font()
    if local:
        UNICODE_FONT_NAME, UNICODE_FONT_PATH = local

# ---------- Text / URL helpers ----------
def _strip_zero_width(s: str) -> str:
    return (s.replace("\u200b", "")
             .replace("\u200e", "")
             .replace("\u200f", "")
             .replace("\xad", ""))

def _sanitize(txt: Union[str, float, int], unicode_ok: bool) -> str:
    s = f"{txt:.3f}" if isinstance(txt, float) else str(txt)
    s = _strip_zero_width(s)
    if unicode_ok:
        return s
    return s.encode("latin-1", "replace").decode("latin-1")

def _fmt_val(v: Union[str, int, float]) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)

def _normalize_doi_or_url(val: str) -> Optional[str]:
    if not val:
        return None
    s = _strip_zero_width(str(val).strip())
    if s.startswith("10."):
        return f"https://doi.org/{s}"
    low = s.lower()
    if low.startswith("http://") or low.startswith("https://"):
        return s
    if low.startswith("www."):
        return f"https://{s}"
    if "." in s and " " not in s:
        return f"https://{s}"
    return None

# ---------- PDF helpers ----------
class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_text_color(0, 0, 0)

        self.unicode_ok = False
        try:
            _resolve_font()
            if UNICODE_FONT_PATH and os.path.exists(UNICODE_FONT_PATH):
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

def _reset_left(pdf: Report):
    pdf.set_x(pdf.l_margin)

def _font(pdf: Report, style: str = "", size: int = 11):
    pdf.set_text_color(0, 0, 0)
    if pdf.unicode_ok:
        pdf.set_font(UNICODE_FONT_NAME, "", size)  # regular only
    else:
        pdf.set_font("Helvetica", style, size)

def _h1(pdf: Report, text: str):
    _font(pdf, "B", 18 if not pdf.unicode_ok else 19)
    _reset_left(pdf)
    pdf.cell(0, 10, _sanitize(text, pdf.unicode_ok), ln=1)

def _h2(pdf: Report, text: str):
    _font(pdf, "B", 13 if not pdf.unicode_ok else 14)
    _reset_left(pdf)
    pdf.cell(0, 8, _sanitize(text, pdf.unicode_ok), ln=1)

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

def _hyperlink(pdf: Report, text: str, url: Optional[str]):
    label = _sanitize(text, pdf.unicode_ok)
    _reset_left(pdf)
    if url:
        if pdf.unicode_ok:
            pdf.set_text_color(0, 0, 200); pdf.set_font(UNICODE_FONT_NAME, "", 11)
        else:
            pdf.set_text_color(0, 0, 200); pdf.set_font("Helvetica", "U", 11)
        pdf.multi_cell(0, 6, label, align="L", link=url)
        pdf.set_text_color(0, 0, 0)
    else:
        _body(pdf, 11)
        pdf.multi_cell(0, 6, label, align="L")
    _reset_left(pdf)

def _key_val_rows(data: List[Tuple[str, Union[str, float, int]]], key_w: float, val_w: float):
    def render(pdf: Report):
        _reset_left(pdf)
        for k, v in data:
            _font(pdf, "B", 11 if not pdf.unicode_ok else 12)
            pdf.cell(key_w, 6, _sanitize(k, pdf.unicode_ok), align="L")
            _body(pdf, 11)
            pdf.multi_cell(val_w, 6, _sanitize(v, pdf.unicode_ok), align="L")
            _reset_left(pdf)
    return render

# ---------- Charts ----------
def _add_series_plot(pdf: Report,
                     series_dict: Dict[str, Sequence[float]],
                     title: str = "Repair Yield per Energy"):
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
        _reset_left(pdf)
        pdf.image(tmp.name, w=pdf.content_w)
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass
    _reset_left(pdf)

# ---------- Builder ----------
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
    if logo_path:
        _add_logo(pdf, logo_path, w=22)

    # Title
    _h1(pdf, "RYE Report")
    pdf.ln(1); _reset_left(pdf)

    # Dataset metadata
    _h2(pdf, "Dataset metadata")
    key_w = 42
    val_w = pdf.content_w - key_w

    ds_link_raw = str(metadata.get("dataset_link", "") or "").strip()
    if ds_link_raw:
        url = _normalize_doi_or_url(ds_link_raw)
        _hyperlink(pdf, f"Dataset link: {url or ds_link_raw}", url or "")

    meta_rows: List[Tuple[str, Union[str, float, int]]] = []
    skip = {"generated", "timestamp", "dataset_link", "columns", "notes", "sample_n"}
    for k, v in metadata.items():
        if k in skip:
            continue
        meta_rows.append((str(k), _fmt_val(v)))
    if meta_rows:
        _key_val_rows(meta_rows, key_w=key_w, val_w=val_w)(pdf)
    pdf.ln(1); _reset_left(pdf)

    # Summary statistics
    _h2(pdf, "Summary statistics")
    items = [(str(k), _fmt_val(v)) for k, v in summary.items()]
    _key_val_rows(items, key_w=key_w, val_w=val_w)(pdf)
    pdf.ln(1); _reset_left(pdf)

    # Plots
    if plot_series:
        if isinstance(plot_series, dict):
            _add_series_plot(pdf, plot_series)
        elif isinstance(plot_series, list):
            for ps in plot_series:
                if isinstance(ps, dict):
                    _add_series_plot(pdf, ps)
        pdf.ln(1); _reset_left(pdf)

    # Sample values (two columns)
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
        _reset_left(pdf)
    pdf.ln(1); _reset_left(pdf)

    # Columns
    cols = metadata.get("columns")
    if isinstance(cols, (list, tuple)) and len(cols) > 0:
        _h2(pdf, "Columns in dataset")
        _body(pdf, 11)
        pdf.multi_cell(0, 6, _sanitize(", ".join(map(str, cols)), pdf.unicode_ok), align="L")
        pdf.ln(1); _reset_left(pdf)

    # Interpretation
    if interpretation:
        _h2(pdf, "Interpretation")
        _body(pdf, 11)
        pdf.multi_cell(0, 6, _sanitize(interpretation, pdf.unicode_ok), align="L")
        pdf.ln(1); _reset_left(pdf)

    # Notes
    notes = metadata.get("notes")
    if notes:
        _h2(pdf, "Notes")
        _body(pdf, 11)
        pdf.multi_cell(0, 6, _sanitize(str(notes), pdf.unicode_ok), align="L")
        pdf.ln(1); _reset_left(pdf)

    # Footer attribution
    _body(pdf, 9)
    pdf.multi_cell(
        0, 5,
        _sanitize(
            "Open science by Cody Ryan Jenkins (CC BY 4.0). "
            "Metric: RYE (Repair Yield per Energy). TGRM.",
            pdf.unicode_ok
        ),
        align="L"
    )

    return bytes(pdf.output(dest="S"))
