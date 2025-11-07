# report.py — Absolute-safe PDF generator (no width errors, UTF-8 guard, clean fallback)
from datetime import datetime
from fpdf import FPDF
import io, matplotlib.pyplot as plt, textwrap, re

PAGE_MARGIN = 15
MAX_LINE_LEN = 100  # shorter width to guarantee fit

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "RYE Report", ln=True, align="L")
        self.ln(4)

def _make_plot_png(rye_series, rye_roll=None) -> bytes:
    fig = plt.figure(figsize=(7, 3))
    ax = plt.gca()
    ax.plot(rye_series, label="RYE", linewidth=1.5)
    if rye_roll is not None:
        ax.plot(rye_roll, label="RYE (rolling)", linewidth=1.5)
    ax.set_xlabel("Index")
    ax.set_ylabel("RYE")
    ax.set_title("Repair Yield per Energy")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _clean_text(txt: str) -> str:
    """Strip control chars, wrap long lines, ensure Latin-1 safety."""
    if not txt:
        return ""
    txt = str(txt)
    txt = re.sub(r"[\x00-\x1F\x7F\u200B\u200C\u200D\uFEFF]", "", txt)
    txt = txt.replace("\t", " ").strip()
    lines = []
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for wrapped in textwrap.wrap(line, MAX_LINE_LEN):
            # encode to latin-1 ignoring unsupported chars
            safe = wrapped.encode("latin-1", errors="ignore").decode("latin-1")
            if safe:
                lines.append(safe)
    return "\n".join(lines)

def _safe_multicell(pdf: FPDF, text: str):
    """Wrapper that never throws width errors."""
    for part in _clean_text(text).split("\n"):
        if not part:
            continue
        try:
            pdf.multi_cell(0, 6, part, align="L")
        except Exception:
            # if still fails, fall back to truncated single cell
            safe = part[:MAX_LINE_LEN].encode("latin-1", errors="ignore").decode("latin-1")
            pdf.cell(0, 6, safe, ln=True)

def build_pdf(
    rye_series,
    summary: dict,
    title: str = "RYE Report",
    metadata: dict | None = None,
    plot_series: dict | None = None,
) -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=PAGE_MARGIN)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, title, ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Generated: {datetime.utcnow().isoformat()}Z", ln=True)
    pdf.ln(3)

    if metadata:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Dataset metadata", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for k, v in metadata.items():
            _safe_multicell(pdf, f"{k}: {v}")
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary statistics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k in ["mean", "median", "max", "min", "count"]:
        v = summary.get(k, "")
        _safe_multicell(pdf, f"{k}: {v}")
    pdf.ln(2)

    try:
        png_bytes = _make_plot_png(
            plot_series.get("rye", rye_series),
            plot_series.get("rye_roll") if plot_series else None,
        )
        img_buf = io.BytesIO(png_bytes)
        pdf.image(img_buf, w=pdf.w - 2 * PAGE_MARGIN)
        pdf.ln(2)
    except Exception as e:
        _safe_multicell(pdf, f"[Chart rendering failed: {e}]")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "RYE sample values (first 100)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, val in enumerate(list(rye_series)[:100]):
        _safe_multicell(pdf, f"{i}: {val}")
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 10)
    footer = (
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics — RYE (Repair Yield per Energy)."
    )
    _safe_multicell(pdf, footer)

    buf = io.BytesIO()
    pdf.output(buf)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
