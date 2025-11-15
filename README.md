# 🧠 RYE Analyzer  
Repair Yield per Energy (RYE) is a universal metric for measuring how efficiently a system converts energy or effort into repair or improvement.  
The RYE Analyzer implements the Reparodynamics framework created by Cody Ryan Jenkins.

![banner](banner.png)

🔗 Live App  
https://rye-analyzer-live.streamlit.app/

![License](https://img.shields.io/badge/License-Dual-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Ready-orange.svg)
![OpenScience](https://img.shields.io/badge/Open%20Science-Supported-brightgreen.svg)

---

## 🚀 Features
- Single file analysis with automatic column detection  
- A and B dataset comparison  
- Multi domain presets for AI, Biology, Robotics, Marketing, and General Systems  
- Energy simulator for hypothetical energy changes  
- Rolling, EMA, and cumulative smoothing  
- RYE scorecard and Reparodynamics or TGRM gauge  
- Phase classification (high efficiency, stable, mixed, decreasing, collapse)  
- Collapse prediction based on linear trend  
- Efficiency frontier regression  
- Noise floor and stability diagnostics  
- Enriched CSV, JSON, and Unicode safe PDF export  
- PDF reports include clickable DOI or dataset link  
- Supports CSV, TSV, Excel, JSON, ZIP, GZ, Darwin Core archives  
- Interactive Streamlit UI that works on mobile and desktop  
- Comes with a built in example dataset

---

## 🧩 How to Use

### 1. Upload your dataset  
Required columns:  
- performance  
- energy  

Optional:  
- time  
- domain  

The Analyzer will attempt to infer columns automatically.

### 2. Adjust settings  
- Rolling window  
- EMA smoothing  
- Energy multiplier  
- Domain preset selection  

### 3. Inspect results  
- RYE curves  
- Rolling and EMA smoothing  
- Efficiency stability and frontier  
- Noise diagnostics  
- TGRM gauge  
- Phase classification  
- Collapse prediction  

### 4. Compare datasets  
Upload a second file to evaluate changes in resilience, collapse risk, efficiency, and RYE deltas.

### 5. Export results  
Download:  
- enriched CSV  
- summary JSON  
- extended JSON with phase and collapse information  
- full Unicode safe PDF with metadata  

### 6. Try without data  
Use the built in example dataset.

---

## ⚙️ Installation

```sh
git clone https://github.com/BoneManTGRM/rye-analyzer.git
cd rye-analyzer
pip install -r requirements.txt
streamlit run app_streamlit.py