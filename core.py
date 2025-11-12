# report.py
# RYE Report generator — Unicode-safe, link-safe, and section-rich.
# Drop-in replacement compatible with your app_streamlit.py.

from __future__ import annotations
import os, io, tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from fpdf import FPDF  # fpdf2 or classic fpdf
except Exception as e:
    raise RuntimeError("fpdf2 is required. Add 'fpdf2>=2.7.9' to requirements.txt") from e


# -----------------------------
# Font resolution (NotoSans preferred, DejaVuSans fallback)
# -----------------------------
FONT_DIR = "fonts"
NOTO_PATH = os.path.join(FONT_DIR, "NotoSans-Regular.ttf")
DEJAVU_PATH = os.path.join(FONT_DIR, "DejaVuSans.ttf")

UNICODE_FONT_NAME: Optional[str] = None
UNICODE_FONT_PATH: Optional[str] = None

def _resolve_font() -> None:
    """Pick a local Unicode-capable TTF if available."""
    global UNICODE_FONT_NAME, UNICODE_FONT_PATH
    if os.path.exists(NOTO_PATH):
        UNICODE_FONT_NAME, UNICODE_FONT_PATH = "NotoSans", NOTO_PATH
    elif os.path.exists(DEJAVU_PATH):
        UNICODE_FONT_NAME, UNICODE_FONT_PATH = "DejaVu", DEJAVU_PATH
    else:
        UNICODE_FONT_NAME = UNICODE_FONT_PATH = None  # will fall back to core fonts


# -----------------------------
# Text sanitizers & helpers
# -----------------------------
_ZW = ("\u200b", "\u200e", "\u200f", "\xad")  # zero-width & soft hyphen

def _strip_zero_width(s: str) -> str:
    for z in _ZW:
        s = s.replace(z, "")
    return s

def _fmt_num(v: Union[str, int, float]) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

def _sanitize(txt: Union[str, float, int], unicode_ok: bool) -> str:
    """Return safe text for current font; remove zero-width characters."""
    s = _strip_zero_width(_fmt_num(txt))
    if unicode_ok:
        return s
    # Core fonts only support latin-1; protect to avoid crashes.
    return s.encode("latin-1", "replace").decode("latin-1")

def _normalize_doi_or_url(val: str) -> Optional[str]:
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


# -----------------------------
# PDF shell
# -----------------------------
class Report(FPDF):
    """A4 portrait, clean margins, page numbers."""
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_text_color(0, 0, 0)
        self.unicode_ok = False

        _resolve_font()
        try:
            if UNICODE_FONT_PATH and os.path.exists(UNICODE_FONT_PATH):
                # Register both regular and bold “faces” for the same family name
                self.add_font(UNICODE_FONT_NAME, "", UNICODE_FONT_PATH, uni=True)
                # If you later add a bold TTF, call add_font(UNICODE_FONT_NAME, "B", <path>, uni=True)
                self.unicode_ok = True
            self.alias_nb_pages()
        except Exception:
            self.unicode_ok = False

    @property
    def content_w(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def footer(self):
        self.set_y(-12)
        self.set_text_color(0, 0, 0)
        if self.unicode_ok:
            self.set_font(UNICODE_FONT_NAME, "", 9)
        else:
            self.set_font("Helvetica", "I", 9)
        self.cell(0, 6, _sanitize(f"Page {self.page_no()}/{{nb}}", self.unicode_ok), align="R")


def _font(pdf: Report, style: str = "", size: int = 11) -> None:
    pdf.set_text_color(0, 0, 0)
    if pdf.unicode_ok:
        # We only registered the regular face; emulate bold via style if available
        try:
            pdf.set_font(UNICODE_FONT_NAME, style, size)
        except Exception:
            pdf.set_font(UNICODE_FONT_NAME, "", size)
    else:
        pdf.set_font("Helvetica", style, size)

def _h1(pdf: Report, text: str) -> None:
    _font(pdf, "B", 18)
    pdf.cell(0, 10, _sanitize(text, pdf.unicode_ok), ln=1)

def _h2(pdf: Report, text: str) -> None:
    _font(pdf, "B", 13)
    pdf.cell(0, 8, _sanitize(text, pdf.unicode_ok), ln=1)

def _body(pdf: Report, size: int = 11) -> None:
    _font(pdf, "", size)

def _add_logo(pdf: Report, path: str = "logo.png", w: float = 22.0) -> None:
    try:
        if os.path.exists(path):
            x = pdf.w - pdf.r_margin - w
            y = pdf.t_margin
            pdf.image(path, x=x, y=y, w=w)
    except Exception:
        pass

def _hyperlink(pdf: Report, label: str, url: Optional[str]) -> None:
    """Clickable link line; robust even if unicode font is unavailable."""
    url = (url or "").strip()
    text = f"{label}: {url}" if url else label
    if not url:
        pdf.multi_cell(0, 6, _sanitize(text, pdf.unicode_ok), align="L")
        return
    pdf.set_text_color(0, 0, 200)
    # underline if possible
    if pdf.unicode_ok:
        try:
            pdf.set_font(UNICODE_FONT_NAME, "U", 11)
        except Exception:
            pdf.set_font(UNICODE_FONT_NAME, "", 11)
    else:
        pdf.set_font("Helvetica", "U", 11)
    pdf.multi_cell(0, 6, _sanitize(text, pdf.unicode_ok), align="L", link=url)
    pdf.set_text_color(0, 0, 0)
    _body(pdf, 11)

def _key_val_rows(rows: List[Tuple[str, Union[str, float, int]]], key_w: float, val_w: float):
    """Wrapped key/value rows that respect the page width."""
    def render(pdf: Report) -> None:
        for k, v in rows:
            _font(pdf, "B", 11)
            pdf.cell(key_w, 6, _sanitize(k, pdf.unicode_ok), align="L")
            _body(pdf, 11)
            pdf.multi_cell(val_w, 6, _sanitize(v, pdf.unicode_ok), align="L")
    return render


# -----------------------------
# Small plots into the PDF
# -----------------------------
def _add_series_plot(pdf: Report,
                     series_dict: Dict[str, Sequence[float]],
                     title: str = "Repair Yield per Energy") -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=180)
    for label, series in series_dict.items():
        ax.plot(range(len(series)), list(series), label=str(label), linewidth=1.8)
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
    metadata: Dict[str, Union[str, int, float, list, dict]],
    plot_series: Optional[Union[Dict[str, Sequence[float]], List[Dict[str, Sequence[float]]]]] = None,
    interpretation: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> bytes:
    """
    Build a multi-section PDF and return raw bytes.

    metadata accepted keys (all optional):
      - 'generated' | 'timestamp' : shown under title if present
      - 'dataset_link'            : DOI or URL (clickable)
      - 'rolling_window'          : int
      - 'rows'                    : int
      - 'preset', 'repair_col', 'energy_col', 'time_col', 'domain_col'
      - 'columns'                 : list of column names
      - 'sample_n'                : how many RYE values to print (default 100)
      - 'notes'                   : freeform text
      - advanced (if available): 'regimes' (list of dicts), 'correlation' (dict),
                                 'noise_floor' (dict), 'bands' (dict with 'low','mid','high' Series)
    """
    pdf = Report()
    pdf.add_page()
    if logo_path:
        _add_logo(pdf, logo_path, w=22)

    # Title & when
    _h1(pdf, "RYE Report")
    when = str(metadata.get("generated") or metadata.get("timestamp") or "").strip()
    if when:
        _body(pdf, 10)
        pdf.cell(0, 6, _sanitize(f"Generated: {when}", pdf.unicode_ok), ln=1)
    pdf.ln(2)

    # Dataset metadata
    _h2(pdf, "Dataset metadata")
    key_w = 42
    val_w = pdf.content_w - key_w
    # Link first (if any)
    raw_link = str(metadata.get("dataset_link", "") or "").strip()
    if raw_link:
        url = _normalize_doi_or_url(raw_link)
        _hyperlink(pdf, "Dataset link", url)

    skip_keys = {"generated", "timestamp", "dataset_link", "columns", "notes",
                 "sample_n", "regimes", "correlation", "noise_floor", "bands"}
    rows = []
    for k, v in metadata.items():
        if k in skip_keys:
            continue
        rows.append((str(k), _fmt_num(v)))
    if rows:
        _key_val_rows(rows, key_w=key_w, val_w=val_w)(pdf)
    pdf.ln(2)

    # Summary statistics (whatever you passed in, e.g., mean/median/resilience)
    _h2(pdf, "Summary statistics")
    sum_rows = [(str(k), _fmt_num(v)) for k, v in summary.items()]
    _key_val_rows(sum_rows, key_w=key_w, val_w=val_w)(pdf)
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
    pdf.ln(2)

    # Optional: columns in dataset
    cols = metadata.get("columns")
    if isinstance(cols, (list, tuple)) and cols:
        _h2(pdf, "Columns in dataset")
        _body(pdf, 11)
        pdf.multi_cell(0, 6, _sanitize(", ".join(map(str, cols)), pdf.unicode_ok), align="L")
        pdf.ln(2)

    # Interpretation (auto text if not provided)
    if interpretation:
        interp_text = str(interpretation)
    else:
        mean = float(summary.get("mean", 0.0) or 0.0)
        minv = float(summary.get("min", 0.0) or 0.0)
        maxv = float(summary.get("max", 0.0) or 0.0)
        resil = summary.get("resilience", None)
        level = "high" if mean > 0.6 else ("moderate" if mean > 0.3 else "modest")
        extra = f" Resilience index {resil:.3f} indicates stability." if isinstance(resil, (int, float)) else ""
        interp_text = (
            f"Average efficiency (RYE mean) is {mean:.3f}, spanning [{minv:.3f}, {maxv:.3f}]. "
            f"Overall efficiency is {level}.{extra} "
            "Use a rolling window to smooth short-term noise, map spikes/dips to interventions, "
            "and iterate TGRM loops (detect → minimal fix → verify) to lift mean RYE and compress variance."
        )
    _h2(pdf, "Interpretation")
    _body(pdf, 11)
    pdf.multi_cell(0, 6, _sanitize(interp_text, pdf.unicode_ok), align="L")
    pdf.ln(2)

    # Advanced analytics (render only if present)
    regimes = metadata.get("regimes")
    if isinstance(regimes, list) and regimes:
        _h2(pdf, "Regimes (sustained zones)")
        _body(pdf, 11)
        for r in regimes[:200]:
            line = f"[{int(r.get('start', 0))} – {int(r.get('end', 0))}]  {str(r.get('label',''))}"
            pdf.multi_cell(0, 6, _sanitize(line, pdf.unicode_ok), align="L")
        pdf.ln(2)

    corr = metadata.get("correlation")
    if isinstance(corr, dict) and corr:
        _h2(pdf, "Energy ↔ ΔPerformance correlation")
        crows = []
        if "pearson" in corr: crows.append(("Pearson", _fmt_num(corr["pearson"])))
        if "spearman" in corr: crows.append(("Spearman", _fmt_num(corr["spearman"])))
        if crows:
            _key_val_rows(crows, key_w=key_w, val_w=val_w)(pdf)
            pdf.ln(2)

    noise = metadata.get("noise_floor")
    if isinstance(noise, dict) and noise:
        _h2(pdf, "Noise floor estimate")
        nrows = []
        for k, v in noise.items():
            nrows.append((str(k), _fmt_num(v)))
        _key_val_rows(nrows, key_w=key_w, val_w=val_w)(pdf)
        pdf.ln(2)

    bands = metadata.get("bands")
    if isinstance(bands, dict) and set(bands.keys()) & {"low","mid","high"}:
        _h2(pdf, "Bootstrap bands (rolling mean)")
        # Summarize as quantiles to save space
        def _qstat(series) -> List[Tuple[str, str]]:
            try:
                import numpy as _np
                a = _np.array(series, dtype=float)
                qs = _np.nanquantile(a, [0.1, 0.5, 0.9])
                return [("p10", f"{qs[0]:.3f}"), ("p50", f"{qs[1]:.3f}"), ("p90", f"{qs[2]:.3f}")]
            except Exception:
                return []
        for label in ("low","mid","high"):
            if label in bands and bands[label] is not None:
                pdf.cell(0, 6, _sanitize(f"{label}:", pdf.unicode_ok), ln=1)
                _key_val_rows(_qstat(bands[label]), key_w=18, val_w=pdf.content_w - 18)(pdf)
        pdf.ln(2)

    # Notes
    notes = metadata.get("notes")
    if isinstance(notes, str) and notes.strip():
        _h2(pdf, "Notes")
        _body(pdf, 11)
        pdf.multi_cell(0, 6, _sanitize(notes, pdf.unicode_ok), align="L")
        pdf.ln(2)

    # Footer
    _body(pdf, 9)
    pdf.multi_cell(
        0, 5,
        _sanitize("Open science by Cody Ryan Jenkins (CC BY 4.0). Metric: RYE (Repair Yield per Energy). TGRM.", pdf.unicode_ok),
        align="L"
    )

    # Return bytes (fpdf2 may return bytes-like already)
    out = pdf.output(dest="S")
    return out if isinstance(out, (bytes, bytearray)) else str(out).encode("latin-1", errors="replace")
