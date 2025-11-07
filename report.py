# report.py — Final Safe Version (auto-wrap + width limit + UTF-8 guard)
from datetime import datetime
from fpdf import FPDF
import io
import matplotlib.pyplot as plt
import textwrap

PAGE_MARGIN = 15  # mm
MAX_LINE_LEN = 120  # wrap long metadata lines manually

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

def _safe_text(txt: str) -> str:
    """Ensure safe encoding and manual wrapping."""
    txt = str(txt)
    lines = []
    for line in txt.splitlines():
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

    # Title
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
            text = _safe_text(f"{k}: {v}")
            pdf.multi_cell(0, 6, text, align="L")
        pdf.ln(2)

    # Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary statistics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k in ["mean", "median", "max", "min", "count"]:
        v = summary.get(k, "")
        pdf.multi_cell(0, 6, _safe_text(f"{k}: {v}"), align="L")
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

    # Values
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "RYE sample values (first 100)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, val in enumerate(list(rye_series)[:100]):
        pdf.cell(0, 6, f"{i}: {val}", ln=True)
    pdf.ln(4)

    # Footer
    pdf.set_font("Helvetica", "I", 10)
    footer = (
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics – RYE (Repair Yield per Energy)."
    )
    pdf.multi_cell(0, 6, _safe_text(footer), align="L")

    out = io.BytesIO()
    pdf.output(out)
    data = out.getvalue()
    out.close()
    return data
