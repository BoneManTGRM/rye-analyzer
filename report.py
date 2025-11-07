# report.py
from fpdf import FPDF
import io
from datetime import datetime

LEFT = 12
RIGHT = 12
TOP = 12
LINE = 6

def _safe_text(x):
    # Avoid non latin-1 characters for FPDF
    return str(x).encode("latin-1", "replace").decode("latin-1")

def _kv_block(pdf, pairs, col_gap=6):
    """
    Render key-value pairs in two columns with wrapping.
    pairs is a list of (key, value)
    """
    content_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = (content_w - col_gap) / 2.0

    left_pairs = pairs[0::2]
    right_pairs = pairs[1::2]
    rows = max(len(left_pairs), len(right_pairs))

    for i in range(rows):
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        # left cell
        if i < len(left_pairs):
            k, v = left_pairs[i]
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(col_w, LINE, _safe_text(k))
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(col_w, LINE, _safe_text(v))
        # compute left cell height used
        h_left = pdf.get_y() - y_start

        # right cell
        pdf.set_xy(x_start + col_w + col_gap, y_start)
        if i < len(right_pairs):
            k, v = right_pairs[i]
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(col_w, LINE, _safe_text(k))
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(col_w, LINE, _safe_text(v))
        h_right = pdf.get_y() - y_start

        # move to next row
        pdf.set_xy(x_start, y_start + max(h_left, h_right) + 2)

def build_pdf(
    rye_series,
    summary: dict,
    title: str = "RYE Report",
    metadata: dict | None = None,
    plot_series: dict | None = None,
    interpretation_text: str | None = None,
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=TOP)
    pdf.add_page()
    pdf.set_margins(LEFT, TOP, RIGHT)

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _safe_text(title), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, _safe_text(f"Generated: {datetime.utcnow().isoformat()}Z"), ln=True)
    pdf.ln(2)

    # Dataset metadata
    if metadata:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Dataset metadata", ln=True)
        pdf.set_font("Arial", "", 11)
        pairs = [(k.replace("_"," "), str(v)) for k, v in metadata.items()]
        _kv_block(pdf, pairs)
        pdf.ln(2)

    # Summary stats
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary statistics", ln=True)
    pdf.set_font("Arial", "", 11)
    pairs = [
        ("mean", f"{summary.get('mean', 0):.6f}"),
        ("median", f"{summary.get('median', 0):.6f}"),
        ("max", f"{summary.get('max', 0):.6f}"),
        ("min", f"{summary.get('min', 0):.6f}"),
        ("count", str(summary.get('count', 0))),
    ]
    _kv_block(pdf, pairs)
    pdf.ln(2)

    # Interpretation
    if interpretation_text:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Interpretation", ln=True)
        pdf.set_font("Arial", "", 11)
        content_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.multi_cell(content_w, LINE, _safe_text(interpretation_text))
        pdf.ln(2)

    # Sample values (first 100)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "RYE sample values (first 100)", ln=True)
    pdf.set_font("Arial", "", 11)
    content_w = pdf.w - pdf.l_margin - pdf.r_margin
    text = "\n".join([f"{i}: {v}" for i, v in list(enumerate(rye_series))[:100]])
    pdf.multi_cell(content_w, LINE, _safe_text(text))
    pdf.ln(2)

    # Footer
    pdf.ln(2)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        content_w,
        LINE,
        _safe_text(
            "Open science by Cody Ryan Jenkins (CC BY 4.0). "
            "Learn more: Reparodynamics  RYE  Repair Yield per Energy"
        ),
    )

    # Output to bytes, compatible with fpdf and fpdf2
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
    except Exception:
        pdf_bytes = pdf.output(dest="S")
    return pdf_bytes
