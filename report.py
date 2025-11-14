# report.py
# Robust PDF builder for RYE Analyzer using fpdf2.
# Keeps all sections: title -> quick summary -> summary -> interpretation -> metadata -> series previews.
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


# --------------------- Optional preset descriptor ---------------------
# You can import/use this in your Streamlit app when you define presets.
MARKETING_PRESET: Dict[str, Any] = {
    "label": "Marketing",
    # Typical column roles for a marketing RYE dataset.
    # Adjust to match your actual CSV headers.
    "repair_col": "conversion_rate",   # or "ROAS", "revenue_per_click", etc.
    "energy_col": "spend",             # money or effort being spent
    "time_col": "date",                # time dimension
    "domain_col": "campaign",          # segments / campaigns / channels
    "rolling_window": 7,               # 7-day smoothing often makes sense
}


# --------------------- Small helpers ---------------------
def _latin1(s: str) -> str:
    """fpdf core fonts are latin-1; replace unsupported chars gracefully."""
    if not isinstance(s, str):
        return s
    # Map common Unicode punctuation to ASCII so we avoid "?" in the PDF
    replacements = {
        "\u2014": "-",    # em dash
        "\u2013": "-",    # en dash
        "\u2212": "-",    # minus sign
        "\u2026": "...",  # ellipsis
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2192": "->",   # right arrow
        "\u2190": "<-",   # left arrow
        "\u00a0": " ",    # non-breaking space
        "\u2248": "~",    # approximately
        "\u2264": "<=",   # less or equal
        "\u2265": ">=",   # greater or equal
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


def _fmt_pct(v: Any) -> str:
    """Format a fraction like 0.1234 as '12.34 %'."""
    try:
        if v is None:
            return ""
        v = float(v)
        return f"{v * 100.0:.2f} %"
    except Exception:
        return str(v)


def _wrap(pdf: FPDF, text: str, w: float, line_h: float = 5.0) -> None:
    """Wrapper around multi_cell that always starts at the left margin."""
    if not text:
        return
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(w, line_h, txt=_latin1(text))


def _kv(pdf: FPDF, title: str, value: str, w: float) -> None:
    """Key/value pair, constrained to total width w, starting at left margin."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(w * 0.3, 6, _latin1(title))
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(w * 0.7, 6, _latin1(value))


def _section_title(pdf: FPDF, title: str, w: float) -> None:
    """Always start section titles at left margin."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(w, 7, _latin1(title))
    pdf.ln(8)


def _small_gap(pdf: FPDF) -> None:
    pdf.ln(2)


def _med_gap(pdf: FPDF) -> None:
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


def _marketing_kpis_section(pdf: FPDF, metadata: Dict[str, Any], W: float) -> None:
    """
    Render a Marketing KPIs block when preset == 'Marketing'.
    It looks for common keys but degrades gracefully if some are missing.
    """
    preset_name = str(metadata.get("preset", "")).lower()
    if preset_name != "marketing":
        return

    _section_title(pdf, "Marketing KPIs", W)
    pdf.set_font("Helvetica", "", 11)

    # Core volume metrics
    impressions = metadata.get("impressions")
    clicks = metadata.get("clicks")
    conversions = metadata.get("conversions")
    spend = metadata.get("spend")
    revenue = metadata.get("revenue")

    if impressions is not None:
        _kv(pdf, "Impressions", _fmt_num(impressions), W)
    if clicks is not None:
        _kv(pdf, "Clicks", _fmt_num(clicks), W)
    if conversions is not None:
        _kv(pdf, "Conversions", _fmt_num(conversions), W)
    if spend is not None:
        _kv(pdf, "Spend", _fmt_num(spend), W)
    if revenue is not None:
        _kv(pdf, "Revenue", _fmt_num(revenue), W)

    # Efficiency metrics (pre-computed or derived)
    ctr = metadata.get("ctr") or metadata.get("click_through_rate")
    cvr = metadata.get("cvr") or metadata.get("conversion_rate")
    cpc = metadata.get("cpc")
    cpa = metadata.get("cpa") or metadata.get("cost_per_acquisition")
    roas = metadata.get("roas") or metadata.get("return_on_ad_spend")

    if ctr is not None:
        _kv(pdf, "CTR", _fmt_pct(ctr), W)
    if cvr is not None:
        _kv(pdf, "CVR", _fmt_pct(cvr), W)
    if cpc is not None:
        _kv(pdf, "CPC", _fmt_num(cpc), W)
    if cpa is not None:
        _kv(pdf, "CPA", _fmt_num(cpa), W)
    if roas is not None:
        _kv(pdf, "ROAS", _fmt_num(roas), W)

    # Experimental / before-vs-after uplift fields
    uplift_conv_rate = metadata.get("uplift_conversion_rate")
    uplift_roas = metadata.get("uplift_roas")
    uplift_revenue = metadata.get("uplift_revenue")

    if uplift_conv_rate is not None:
        _kv(pdf, "Conversion rate uplift", _fmt_pct(uplift_conv_rate), W)
    if uplift_roas is not None:
        _kv(pdf, "ROAS uplift", _fmt_pct(uplift_roas), W)
    if uplift_revenue is not None:
        _kv(pdf, "Incremental revenue", _fmt_num(uplift_revenue), W)

    _med_gap(pdf)


def _headline_from_summary(summary: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """
    Build a short, two-line style headline that appears near the top of the PDF.
    It is preset-aware (generic, marketing, or marine biology).
    """
    preset_name = str(metadata.get("preset", "")).lower()
    mean_v = float(summary.get("mean", 0) or 0)
    resil = summary.get("resilience", None)
    p10 = summary.get("p10", None)
    p90 = summary.get("p90", None)

    try:
        resil_val = float(resil) if resil is not None else None
    except Exception:
        resil_val = None

    # Flags
    low_band_cross = False
    if isinstance(p10, (int, float)) and isinstance(p90, (int, float)):
        low_band_cross = p10 < 0.3 < p90

    # Simple labels for efficiency and stability
    if mean_v > 0.6:
        eff_label = "high efficiency"
    elif mean_v > 0.3:
        eff_label = "moderate efficiency"
    elif mean_v > 0.05:
        eff_label = "low efficiency"
    else:
        eff_label = "near zero efficiency"

    if resil_val is None:
        stab_label = "stability not computed"
    elif resil_val < 0.1:
        stab_label = "no stable regulation"
    elif resil_val < 0.4:
        stab_label = "partial stability"
    else:
        stab_label = "strong stability"

    # Preset specific text
    if preset_name == "marketing":
        base = f"Campaign RYE shows {eff_label} with {stab_label}."
        if low_band_cross:
            tail = (
                "There are clear periods where efficiency falls into a weak band; those segments "
                "are prime targets for budget repair."
            )
        else:
            tail = (
                "Most cycles cluster in a consistent efficiency band; use high RYE segments as a "
                "guide for scaling spend."
            )
        return f"{base} {tail}"

    if preset_name == "marine biology":
        base = f"Marine RYE shows {eff_label} with {stab_label} across the observed seasons."
        if low_band_cross:
            tail = (
                "Cycles dip through the low efficiency band, hinting at stress periods where gross "
                "primary production gains are weak for the respiratory cost."
            )
        else:
            tail = (
                "Efficiency stays in a relatively stable band, suggesting a repeatable coupling "
                "between production and respiration."
            )
        return f"{base} {tail}"

    # Generic headline
    base = f"RYE shows {eff_label} with {stab_label} across the dataset."
    if low_band_cross:
        tail = (
            "The series crosses the low efficiency band around 0.3, which acts as an early "
            "warning for unstable repair."
        )
    else:
        tail = (
            "Most cycles stay within a narrow efficiency band, pointing to a characteristic "
            "operating regime."
        )
    return f"{base} {tail}"


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

    # Compute printable width dynamically from margins with a safety buffer
    SAFE_MARGIN = 8.0  # extra buffer so even strict viewers do not clip
    W = pdf.w - pdf.l_margin - pdf.r_margin - SAFE_MARGIN

    # Header
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(W, 10, _latin1("RYE Analyzer Report"), ln=1)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 10)

    # Optional use_case line (e.g. "marketing_efficiency")
    use_case = str(metadata.get("use_case", "") or "").strip()
    if use_case:
        pdf.cell(W, 6, _latin1(f"Repair Yield per Energy - use case: {use_case}"), ln=1)
    else:
        pdf.cell(W, 6, _latin1("Repair Yield per Energy - portable summary"), ln=1)

    _med_gap(pdf)

    # Quick summary headline (preset aware)
    headline = _headline_from_summary(summary, metadata)
    if headline:
        _section_title(pdf, "Quick summary", W)
        pdf.set_font("Helvetica", "", 11)
        _wrap(pdf, headline, W)

        # If the app supplied a separate quick_summary string in metadata, show it just under the headline.
        meta_qs = str(metadata.get("quick_summary", "") or "").strip()
        if meta_qs:
            _small_gap(pdf)
            pdf.set_font("Helvetica", "I", 10)
            _wrap(pdf, meta_qs, W)

        _med_gap(pdf)

    # Summary stats (grid-like listing) use 2 columns for more room
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

    # Interpretation
    _section_title(pdf, "Interpretation", W)
    pdf.set_font("Helvetica", "", 11)
    if interpretation:
        _wrap(pdf, interpretation, W)
    else:
        _wrap(pdf, "No interpretation supplied by the app.", W)

    # Extra ecological hint for Marine Biology preset
    preset_name = str(metadata.get("preset", "")).lower()
    if preset_name == "marine biology":
        extra = (
            "In ecological terms, repeated peaks and troughs in RYE can mark coupling "
            "between gross primary production and ecosystem respiration, or periods of "
            "stress where metabolic cost rises faster than repair."
        )
        _small_gap(pdf)
        _wrap(pdf, extra, W)

    _med_gap(pdf)

    # Marketing-specific KPIs (only if preset == "Marketing")
    _marketing_kpis_section(pdf, metadata, W)

    # Metadata
    if metadata:
        _section_title(pdf, "Metadata", W)
        priority = [
            "rows",
            "preset",
            "use_case",          # added so it also appears in the metadata table
            "repair_col",
            "energy_col",
            "time_col",
            "domain_col",
            "rolling_window",
            "dataset_link",
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

    # Series previews
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

    # Base RYE sequence
    if rye_series:
        _ensure_space(pdf, needed_mm=25)
        _section_title(pdf, "RYE sequence", W)
        pdf.set_font("Helvetica", "", 10)
        for line in _as_rows("RYE", rye_series):
            _wrap(pdf, line, W)

    out = io.BytesIO()
    pdf.output(out)
    return out.getvalue()
