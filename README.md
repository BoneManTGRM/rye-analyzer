# 🧠 RYE Analyzer

**Reparodynamics Open Science Initiative**  
Tools for measuring self repair and efficiency in real systems.

Repair Yield per Energy (RYE) is a universal metric for measuring how efficiently a system converts energy or effort into repair or improvement.  
The RYE Analyzer implements the Reparodynamics framework created by **Cody Ryan Jenkins**.

---

## 🔖 Badges

![License](https://img.shields.io/badge/License-Reparodynamics_Dual-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Framework](https://img.shields.io/badge/Streamlit-Ready-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![Open Science](https://img.shields.io/badge/Open_Science-Supported-purple)

---

## 🔗 Live App

https://rye-analyzer-live.streamlit.app/

Use the hosted Streamlit app to run RYE analyses without installing anything.

---

## 🚀 Features

- Single file analysis with automatic column detection  
- A and B dataset comparison  
- Multi domain presets for AI, Biology, Robotics, Marketing, Marine Biology, Omics, and General Systems  
- Energy simulator for hypothetical energy changes  
- Rolling, EMA, cumulative, and windowed smoothing  
- RYE scorecard and Reparodynamics (TGRM) self repair gauge  
- Phase classification (high efficiency, stable, mixed, decreasing, collapse)  
- Collapse prediction based on linear trend  
- Efficiency frontier regression for energy vs delta performance  
- Noise floor and stability diagnostics  
- Enriched CSV, JSON, and Unicode safe PDF export  
- PDF reports include a clickable DOI or dataset link  
- Supports CSV, TSV, Excel, JSON, ZIP, GZ, Darwin Core Archives, Parquet, HDF5, Arrow, and NetCDF  
- Interactive Streamlit UI that works on mobile and desktop  
- Built in example dataset to explore RYE without your own data  

---

## 🧩 How to Use

### 1. Upload your dataset

Required columns:

- `performance`  
- `energy`  

Optional:

- `time`  
- `domain`  

The analyzer will try to infer column roles automatically using presets and smart heuristics.

---

### 2. Adjust settings

In the sidebar you can tune:

- Rolling window size  
- EMA smoothing strength  
- Energy multiplier for what if experiments  
- Domain preset (AI, Biology, Marketing, Marine, General Systems and others)

---

### 3. Inspect results

The app provides:

- RYE curves over time or index  
- Rolling and EMA smoothed RYE  
- Stability and resilience metrics  
- Reparodynamics self repair gauge  
- Phase classification of the system  
- Collapse prediction based on trend toward RYE zero  
- Energy vs delta performance scatter with efficiency frontier  
- Noise floor and variability diagnostics  

---

### 4. Compare datasets

Upload a second file to compare:

- Mean RYE and resilience between A and B  
- Relative improvement or degradation in efficiency  
- Phase and collapse risk for each dataset  
- RYE distributions and smoothed curves side by side  

---

### 5. Export results

Download:

- Enriched CSV with RYE, rolling RYE, EMA, and cumulative RYE  
- Summary JSON  
- Extended JSON including phase and collapse information  
- Full Unicode safe PDF report with metadata and optional DOI or dataset link  

---

### 6. Try without data

If you do not have data ready, click **Download example CSV** in the sidebar and test the full workflow with the built in example dataset.

---

## ⚙️ Local Installation

```bash
git clone https://github.com/BoneManTGRM/rye-analyzer.git
cd rye-analyzer
pip install -r requirements.txt
streamlit run app_streamlit.py
