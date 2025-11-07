# report.py — PDF builder compatible with fpdf 1.x and fpdf2
from datetime import datetime
from fpdf import FPDF

def build_pdf(rye_series, summary: dict, title: str = "RYE Report") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="L")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.utcnow().isoformat()}Z", ln=True)
    pdf.ln(4)

    # Summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary statistics", ln=True)
    pdf.set_font("Arial", "", 11)
    for k in ["mean", "median", "max", "min", "count"]:
        v = summary.get(k, "")
        pdf.cell(0, 6, f"{k}: {v}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "RYE sample values", ln=True)
    pdf.set_font("Arial", "", 11)
    for i, val in enumerate(list(rye_series)[:120]):  # keep it short
        pdf.cell(0, 6, f"{i}: {val}", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0, 6,
        "Open science by Cody Ryan Jenkins (CC BY 4.0). "
        "Learn more: Reparodynamics — RYE (Repair Yield per Energy)."
    )

    # -------- Output handling (works for both fpdf and fpdf2) --------
    # In fpdf 1.x, output(dest='S') returns a *str* that must be latin-1 encoded.
    # In fpdf2, output(dest='S') returns *bytes* already.
    try:
        out = pdf.output(dest="S")
    except TypeError:
        # Very old fpdf versions sometimes behave oddly; last-resort fallback
        from io import BytesIO
        buf = BytesIO()
        # fpdf2 can also accept a file-like object
        pdf.output(buf)
        return buf.getvalue()

    if isinstance(out, bytes):
        return out
    # str -> bytes (fpdf 1.x)
    return out.encode("latin-1")
