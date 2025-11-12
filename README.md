# 🧠 RYE Analyzer

**Compute Repair Yield per Energy (RYE)** — an open-science metric for measuring how efficiently a system converts effort or energy into self-repair or performance improvement.

The RYE Analyzer implements the **Reparodynamics** framework developed by **Cody Ryan Jenkins** (Open Science / CC-BY-4.0).

---

## 🚀 Features

- Single CSV analysis
- Compare two datasets (before/after)
- Multi-domain presets: **AI**, **Biology**, and **Robotics**
- Energy simulator (test ΔEnergy scenarios)
- Rolling window smoothing
- Automatic summary metrics and RYE scorecard
- Downloadable **CSV**, **JSON**, and **Unicode-safe PDF reports**
- Clickable **Zenodo DOI / dataset link** in the PDF
- Built-in example dataset
- Streamlit UI — fully interactive and mobile-friendly

---

## 🧩 How to Use

1. **Upload a CSV file** containing:
   - `performance` — repair or output metric  
   - `energy` — effort, time, or energy input  
   - *(Optional)*: `time`, `domain`
2. Adjust the **rolling window** and **energy multiplier** in the sidebar.
3. View **plots**, **summary statistics**, and **download** your results.
4. *(Optional)* Upload a second CSV to compare two datasets.
5. *(Optional)* Enter a **Zenodo DOI or dataset URL** to embed a clickable link inside your PDF report.

If you don’t have data yet, click **“Download example CSV”** to start testing.

---

#git clone https://github.com/BoneManTGRM/rye-analyzer.git
cd rye-analyzer
pip install -r requirements.txt
streamlit run app_streamlit.py
