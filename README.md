# RYE Analyzer

**Compute Repair Yield per Energy (RYE)** from any time series — an open-science tool built with Streamlit.

This app measures how efficiently a system converts effort or energy into repair or performance improvements.  
It implements the Reparodynamics framework developed by **Cody Ryan Jenkins (Open Science / CC-BY-4.0)**.

---

## 🚀 Features

- Single CSV analysis  
- Compare two datasets (before/after)  
- Multi-domain plotting (AI, Bio, Robotics, etc.)  
- Rolling window smoothing  
- Energy simulator (test ΔEnergy scenarios)  
- Automatic summary metrics and RYE scorecard  
- Downloadable CSV, JSON, and PDF reports  
- Built-in example dataset  
- Streamlit UI — fully interactive and mobile-friendly

---

## 🧩 How to Use

1. Upload a CSV file containing:
   - `performance` — repair or output metric  
   - `energy` — effort, time, or energy input  
   - Optional: `time`, `domain`
2. Adjust the rolling window and energy multiplier in the sidebar.
3. View plots, summary stats, and download your results.
4. Optionally upload a second CSV to compare two datasets.

If you don’t have data yet, click **“Download example CSV”** to start testing.

---

## 📦 Installation (Local)

```bash
git clone https://github.com/BoneManTGRM/rye-analyzer.git
cd rye-analyzer
pip install -r requirements.txt
streamlit run app_streamlit.py
