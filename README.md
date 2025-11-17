# 🧠 RYE Analyzer  
### *Reparodynamics Open Science Initiative*  
**Created by Cody Ryan Jenkins — Founder of Reparodynamics**

A high-precision analytical engine for measuring **self-repair efficiency**, **resilience**, and **adaptive stability** in real systems using the Reparodynamics framework.

At its core is **RYE — Repair Yield per Energy**, a universal metric that quantifies how effectively a system converts energy into verified improvement across biology, AI, robotics, engineering, and ecological sciences.

---

## 🔖 Scientific Badges

![License](https://img.shields.io/badge/License-Apache_2.0_%2B_Attribution-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Framework](https://img.shields.io/badge/Streamlit-Ready-FE560A)
![Stability](https://img.shields.io/badge/Stability-Production-success)
![Open Science](https://img.shields.io/badge/Open_Science-Enabled-9D4EDD)
![Domains](https://img.shields.io/badge/Domains-AI%20%7C%20Biology%20%7C%20Robotics%20%7C%20Marketing-informational)

---

## 🔗 Live App  
Run instantly in your browser (no installation required):

**https://rye-analyzer-live.streamlit.app/**

Optimized for desktop, tablet, and mobile.

---

# 🚀 Core Capabilities

## 🔬 1. Single-Dataset RYE Analysis
- Automatic semantic column detection  
- Multi-domain presets  
- Time-based or index-based RYE curves  
- Δ-performance computation  
- Rolling, EMA, and cumulative smoothing  

---

## 🔄 2. A ↔ B Comparative Analysis
Compare stability, drift, and repair efficiency between two datasets:

- ΔRYE difference  
- Resilience and stability comparison  
- Collapse prediction  
- Overlayed curves (rolling, EMA, cumulative)  
- Domain-aware diagnostics  

Ideal for: A/B experiments, biological perturbations, firmware tuning, robotics drift, omics, marine/ecological data, and marketing analytics.

---

## ⚙️ 3. Energy Simulator
Modify hypothetical energy/resource usage and instantly view predicted RYE impacts.

---

## 📊 4. Reparodynamic Analytics
- TGRM self-repair gauge  
- Repair-phase classification  
- Collapse prediction  
- Efficiency frontier regression  
- Noise-floor + resilience diagnostics  

---

## 📦 5. Multi-Format Data Support
Accepts virtually all scientific dataset types:

- CSV  
- TSV  
- Excel  
- JSON / NDJSON  
- ZIP / GZ  
- Parquet  
- HDF5  
- Arrow  
- NetCDF  
- Darwin Core Archives  
- Many omics & ecological formats  

---

## 📤 6. Unified Export Suite
Generate publication-ready outputs:

- Enriched CSV  
- Summary JSON  
- Extended JSON (phases + collapse + diagnostics)  
- Unicode-safe PDF (optionally with dataset DOI)  

---

# 🧩 How to Use

## 1. Upload your dataset
Minimum required columns:
- `performance`  
- `energy`

Optional but recommended:
- `time`  
- `domain`

The analyzer will automatically infer column roles.

---

## 2. Adjust your analysis
Sidebar settings include:

- Rolling window  
- EMA smoothing  
- Energy multiplier  
- Domain presets  

---

## 3. Interpret RYE and Reparodynamics
View:

- Base RYE  
- Rolling + EMA  
- Stability phases  
- Efficiency frontier  
- Noise floor  
- TGRM self-repair gauge  
- Collapse modeling  

---

## 4. Add a second file for A↔B comparison
Instantly visualize:

- ΔRYE  
- Stability shift  
- Regime change  
- Collapse-risk differences  

---

## 5. Export scientific outputs
Download CSV, JSON, or PDF for research or publication.

---

# ⚙️ Local Installation

```bash
git clone https://github.com/BoneManTGRM/rye-analyzer.git
cd rye-analyzer
pip install -r requirements.txt
streamlit run app_streamlit.py
