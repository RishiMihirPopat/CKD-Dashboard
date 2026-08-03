# CKD Clinical Intelligence & Merck R&D Streamlit Dashboard

This folder contains all isolated code, assets, model cache, and data required to run the **Chronic Kidney Disease (CKD) Streamlit Dashboard**.

---

## 📁 Included Files

- **`app.py`**: Main Streamlit application file containing UI layouts, tabs, Plotly charts, and the live patient risk simulator.
- **`model_pipeline.py`**: Machine learning and biostatistical pipeline module for hypothesis testing, model training, cross-validation, and SHAP explainability.
- **`Chronic_Kidney_Dsease_data.csv`**: Raw dataset containing patient demographic, lab, and clinical biomarker records.
- **`ckd_pipeline_cache.pkl`**: Pre-computed pipeline cache containing trained models, metrics, and SHAP value matrices for instant dashboard loading.
- **`requirements.txt`**: List of required Python packages.
- **`README.md`**: Project documentation and quickstart instructions.

---

## 🚀 How to Run the Dashboard

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.
