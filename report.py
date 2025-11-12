# report.py
# Robust PDF builder for RYE Analyzer using fpdf2.
# Keeps all sections: title → summary → interpretation → metadata → series previews.
# Tries to render compact line plots; falls back to text previews if plotting fails.

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Any
from fpdf import FPDF
import io

# Optional plotting (safe fallback if matplotlib isn't present)
try:
    import matplotlib.pyplot as plt  # no external styles/colors
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# --------------------- Small helpers ---------------------
def _latin1(s: str) -> str:
    """fpdf core fonts are latin-1; replace unsupported chars gracefully."""
    if not isinstance(s, str):
        return s
    # Map common Unicode punctuation to ASCII so we avoid "?" in the PDF
    replacements = {
        "\u2014": "-",    # em dash —
        "\u2013": "-",    # en dash –
        "\u2212": "-",    # minus sign −
        "\u2026": "...",  # ellipsis …
        "\u2018": "'",    # ‘
        "\u2019": "'",    # ’
        "\u201c": '"',    # “
        "\u201d": '"',    # ”
        "\u2192": "->",   # →
        "\u2190": "<-",   # ←
        "\u00a0": " ",    # non-breaking space
        "\u2248": "~",    # ≈
        "\u2264": "<=",   # ≤
        "\u2265": ">=",   # ≥
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    try:
        return s.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return s


def _fmt_num(v: Any) -> str:
    try:
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
                return f"{v:.6f}"
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def _wrap(pdf: FPDF, text: str, w: float, line_h: float = 5.0) -> None:
    if not text:
        return
    pdf.multi_cell(w, line_h, txt=_latin1(text))


def _kv(pdf: FPDF, title: str, value: str, w: float) -> None:
    # key/value pair, constrained to total width w, starting at left margin
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(w * 0.3, 6, _latin1(title))
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(w * 0.7, 6, _latin1(value))


def _section_title(pdf: FPDF, title: str, w: float) -> None:
    # always start section titles at left margin
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(w, 7, _latin1(title))
    pdf.ln(8)


def _small_gap(pdf: FPDF):
    pdf.ln(2)


def _med_gap(pdf: FPDF):
    pdf.ln(4)


def _as_rows(name: str, seq: Iterable[float], max_rows: int = 120) -> List[str]:
    vals = list(seq)
    n = len(vals)
    if n == 0:
        return [f"{name}: (empty)"]
    rows = [f"{name} (first {min(n, max_rows)} of {n})"]
    chunk = 10
    for i in range(0, min(n, max_rows), chunk):
        part = ", ".join(_fmt_num(vals[j]) for j in range(i, min(i + chunk, min(n, max_rows))))
        rows.append(part)
    if n > max_rows:
        rows.append("... (truncated)")
    return rows


def _image_from_series(
    series_dict: Dict[str, List[float]],
    width_px: int = 1000,
    height_px: int = 400
) -> Optional[bytes]:
    """Return PNG bytes of a simple line plot, or None if plotting fails or matplotlib absent."""
    if not _HAS_MPL:
        return None
    try:
        fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
        ax = fig.add_subplot(111)
        for name, ys in series_dict.items():
            if ys and any(isinstance(t, (int, float)) for t in ys):
                ax.plot(range(len(ys)), ys, label=name)  # no explicit colors or styles
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Series preview")
        if len(series_dict) > 1:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _ensure_space(pdf: FPDF, needed_mm: float) -> None:
    """
    Ensure there is at least `needed_mm` of vertical space left.
    If not, start a new page. This avoids ugly splits.
    """
    page_h = getattr(pdf, "h", 297)  # A4 ~297mm
    bottom_margin = getattr(pdf, "b_margin", 12)
    y = pdf.get_y()
    if y + needed_mm > (page_h - bottom_margin):
        pdf.add_page()


# --------------------- Main builder ---------------------
def build_pdf(
    rye_series: List[float],
    summary: Dict,
    *,
    metadata: Optional[Dict] = None,
    plot_series: Optional[Dict[str, List[float]]] = None,
    interpretation: str = ""
) -> bytes:
    """
    Returns PDF bytes.
    """
    metadata = metadata or {}
    plot_series = plot_series or {}

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # Compute printable width dynamically from margins with a tiny safety buffer
    W = pdf.w - pdf.l_margin - pdf.r_margin - 2

    # Header
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(W, 10, _latin1("RYE Analyzer Report"), ln=1)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(W, 6, _latin1("Repair Yield per Energy - portable summary"), ln=1)
    _med_gap(pdf)

    # Summary stats (grid-like listing) — use 2 columns for more room
    _section_title(pdf, "Summary stats", W)
    pdf.set_font("Helvetica", "", 11)
    keys = ["mean", "median", "min", "max", "std", "resilience", "count", "p10", "p50", "p90", "iqr"]
    cols_per_row = 2
    colw = W / cols_per_row
    pdf.set_x(pdf.l_margin)
    for i, k in enumerate(keys):
        v = _fmt_num(summary.get(k, ""))
        pdf.cell(colw, 6, _latin1(f"{k}: {v}"))
        if (i + 1) % cols_per_row == 0:
            pdf.ln(6)
            pdf.set_x(pdf.l_margin)
    if len(keys) % cols_per_row != 0:
        pdf.ln(6)
    _med_gap(pdf)

    # Interpretation (prominent)
    _section_title(pdf, "Interpretation", W)
    pdf.set_font("Helvetica", "", 11)
    if interpretation:
        _wrap(pdf, interpretation, W)
    else:
        _wrap(pdf, "No interpretation supplied by the app.", W)
    _med_gap(pdf)

    # Metadata (priority keys first, then the rest)
    if metadata:
        _section_title(pdf, "Metadata", W)
        priority = [
            "rows", "preset", "repair_col", "energy_col", "time_col",
            "domain_col", "rolling_window", "dataset_link"
        ]
        shown = set()
        for k in priority:
            if k in metadata:
                _kv(pdf, f"{k}:", _fmt_num(metadata[k]), W)
                shown.add(k)
        # Extra structured analytics if present
        for k in ("regimes", "correlation", "noise_floor", "bands"):
            if k in metadata and k not in shown:
                _kv(pdf, f"{k}:", _fmt_num(metadata[k]), W)
                shown.add(k)
        # Columns list
        if "columns" in metadata and "columns" not in shown:
            cols_str = ", ".join(map(str, metadata["columns"]))
            _kv(pdf, "columns:", cols_str, W)
            shown.add("columns")
        # The rest
        for k, v in metadata.items():
            if k in shown:
                continue
            _kv(pdf, f"{k}:", _fmt_num(v), W)
        _med_gap(pdf)

    # Series previews — attempt small plot, else text tables
    if plot_series:
        _ensure_space(pdf, needed_mm=20)
        _section_title(pdf, "Series previews", W)
        img_bytes = _image_from_series(plot_series)
        if img_bytes:
            try:
                _ensure_space(pdf, needed_mm=10)
                pdf.set_x(pdf.l_margin)
                pdf.image(io.BytesIO(img_bytes), w=W, type="PNG")
                _small_gap(pdf)
            except Exception:
                pdf.set_font("Helvetica", "", 10)
                for name, seq in plot_series.items():
                    for line in _as_rows(name, seq):
                        _wrap(pdf, line, W)
                    _small_gap(pdf)
        else:
            pdf.set_font("Helvetica", "", 10)
            for name, seq in plot_series.items():
                for line in _as_rows(name, seq):
                    _wrap(pdf, line, W)
                _small_gap(pdf)

    # Base RYE sequence as last section
    if rye_series:
        _ensure_space(pdf, needed_mm=25)
        _section_title(pdf, "RYE sequence", W)
        pdf.set_font("Helvetica", "", 10)
        for line in _as_rows("RYE", rye_series):
            _wrap(pdf, line, W)

    out = io.BytesIO()
    pdf.output(out)
    return out.getvalue()
