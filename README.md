# 🧠 RYE Analyzer  
### *Reparodynamics Open Science Initiative*  
**Created by Cody Ryan Jenkins**

A high-precision analytical engine for measuring **self-repair efficiency**, **resilience**, and **adaptive stability** in real systems using the Reparodynamics framework.  
At its core is **RYE — Repair Yield per Energy**, a universal metric that quantifies how effectively a system converts energy or effort into verified improvement.

---

## 🔖 Scientific Badges

![License](https://img.shields.io/badge/License-Reparodynamics_Dual-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Framework](https://img.shields.io/badge/Streamlit-Ready-FE560A)
![Stability](https://img.shields.io/badge/Stability-Production-success)
![Open Science](https://img.shields.io/badge/Open_Science-Supported-9D4EDD)
![Domains](https://img.shields.io/badge/Domains-AI%20%7C%20Biology%20%7C%20Robotics%20%7C%20Marketing-informational)

---

## 🔗 Live App  
**Run instantly in your browser:**  
https://rye-analyzer-live.streamlit.app/

No installation required. Fully optimized for desktop and mobile.

---

# 🚀 Core Capabilities

### 🔬 **1. Single-Dataset RYE Analysis**
- Automatic column detection  
- Multi-domain presets  
- Time-based or index-based RYE curves  
- Delta-performance engine  
- Rolling, EMA, and cumulative smoothing

### 🔄 **2. Full A ↔ B Dataset Comparison**
Evaluate improvement or degradation between two conditions:

- ΔRYE difference  
- Resilience and stability comparison  
- Collapse risk estimation for A and B  
- Overlayed curves (rolling, EMA, cumulative)  
- Domain-aware diagnostics  
- Combined enriched output  

Ideal for:
- Before/after experiments  
- A/B marketing tests  
- Biological perturbations  
- Firmware or robotics tuning  
- Omics analysis  
- Ecological and marine datasets  

### ⚙️ **3. Energy Simulator**
Test hypothetical changes in energy or resource usage and see the RYE impact instantly.

### 📊 **4. Advanced Reparodynamic Analytics**
- TGRM self-repair gauge  
- Phase classification  
  - High efficiency  
  - Stable  
  - Mixed  
  - Decreasing  
  - Collapse  
- Collapse prediction via trend modeling  
- Efficiency frontier regression  
- Noise floor, stability, and resilience diagnostics  

### 📦 **5. Multi-Format Support**
Accepts virtually any scientific dataset:

- CSV  
- TSV  
- Excel  
- JSON  
- NDJSON  
- ZIP / GZ bundles  
- Parquet  
- HDF5  
- Arrow  
- NetCDF  
- Darwin Core Archives  
- Many omics and ecological formats  

### 📤 **6. Unified Export Suite**
- Enriched CSV  
- Summary JSON  
- Extended JSON (phase + collapse + diagnostics)  
- Unicode-safe PDF with optional DOI/dataset link  

---

# 🧩 How to Use

### 1. **Upload your dataset**
Provide at minimum:
- `performance`  
- `energy`  

Optional:
- `time`  
- `domain`

The analyzer will automatically infer column roles using semantic heuristics.

---

### 2. **Tune your analysis**
Adjust in the sidebar:

- Rolling window  
- EMA smoothing  
- Energy multiplier  
- Domain preset  

---

### 3. **Explore RYE and Reparodynamics**
Understand system behavior through:

- Base RYE curves  
- Rolling/EMA smoothed trends  
- Efficiency frontier  
- Noise floor  
- Phase classification  
- Collapse prediction  
- Self-repair gauge (TGRM)  

---

### 4. **Upload a second file for A↔B comparison**
Instantly visualize:

- ΔRYE  
- Resilience shift  
- Regime change  
- Stability differences  
- Collapse risk for each dataset  

---

### 5. **Export everything**
Generate publication-ready outputs:

- CSV  
- JSON  
- Extended JSON  
- Scientific PDF report  

---

# ⚙️ Local Installation

```bash
git clone https://github.com/BoneManTGRM/rye-analyzer.git
cd rye-analyzer
pip install -r requirements.txt
streamlit run app_streamlit.py
