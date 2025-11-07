from fpdf import FPDF
import io
from datetime import datetime

def build_pdf(rye_series, summary: dict, title="RYE Report"):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    # Summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary statistics", ln=True)
    pdf.set_font("Arial", "", 11)
    for k in ["mean", "median", "max", "min", "count"]:
        v = summary.get(k, "")
        pdf.cell(0, 7, f"{k}: {v}", ln=True)

    # RYE sample values
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "RYE sample values", ln=True)
    pdf.set_font("Arial", "", 11)
    for i, val in enumerate(list(rye_series)[:15]):
        pdf.cell(0, 6, f"{i}: {val}", ln=True)

    # Footer
    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 6, "Open science by Cody Ryan Jenkins. CC BY 4.0")

    # Output to bytes (this fixes your Streamlit error)
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes
