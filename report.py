# report.py — Final hardened version for Streamlit Cloud (handles all text safely)
from datetime import datetime
from fpdf import FPDF
import io
import matplotlib.pyplot as plt
import textwrap
import re

PAGE_MARGIN = 15  # mm
MAX_LINE_LEN = 120  # characters per line before wrapping

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "RYE Report", ln=True, align="L")
        self.ln(4)

def _make_plot_png(rye_series, rye_roll=None) -> bytes:
    """Render chart as PNG for embedding."""
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

def _sanitize_text(text: str) -> str:
    """Remove zero-width and control characters, and wrap safely."""
    if text is None:
        return ""
    text = str(text)
    # Remove non-printable and zero-width chars
    text = re.sub(r"[\x00-\x1F\x7F\u200B\u200C\u200D\uFEFF]", "", text)
    # Replace tabs with spaces
    text = text.replace("\t", " ").strip()
    # Break long lines manually
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue  # skip empty
        lines.extend(textwrap.wrap(line, MAX_LINE_LEN))
    return "\n".join(lines)

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

    # Header
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, title, ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Generated: {datetime.utcnow().isoformat()}Z", ln=True)
    pdf.ln(3)

    # Metadata
    if metadata:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Dataset metadata", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for k, v in metadata.items():
            clean_text = _sanitize_text(f"{k}: {v}")
            if clean_text:
                pdf.multi_cell(0, 6, clean_text, align="L")
        pdf.ln(2)

    # Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary statistics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k in ["mean", "median", "max", "min", "count"]:
        v = summary.get(k, "")
        clean_text = _sanitize_text(f"{k}: {v}")
        if clean_text:
            pdf.multi_cell(0, 6, clean_text, align="L")
    pdf.ln(2)

    # Chart
    try:
        png_bytes = _make_plot_png(
            plot_series.get("rye", rye_series),
            plot_series.get("rye_roll") if plot_series else None,
        )
        img_buf = io.BytesIO(png_bytes)
        page_width = pdf.w - 2 * PAGE_MARGIN
        pdf.image(img_buf, w=page_width)
        pdf.ln(2)
    except Exception as e:
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, f"[Chart rendering failed: {e}]")

    # RYE values
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "RYE sample values (first 100)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, val in enumerate(list(rye_series)[:100]):
        text = _sanitize_text(f"{i}: {val}")
        if text:
            pdf.cell(0, 6, text, ln=True)
    pdf.ln(4)

    # Footer
    pdf.set_font("Helvetica", "I", 10)
    footer = (
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics — RYE (Repair Yield per Energy)."
    )
    pdf.multi_cell(0, 6, _sanitize_text(footer), align="L")

    # Export to bytes
    output = io.BytesIO()
    pdf.output(output)
    data = output.getvalue()
    output.close()
    return data
