# Smart Leaf Analyzer

The Smart Leaf Analyzer is a local Streamlit application that classifies Butterhead lettuce leaves as **Healthy** or **Tipburn** using visible-near-infrared spectral measurements.

The dashboard applies the locked preprocessing and Logistic Regression pipeline developed in the accompanying notebook. It performs inference only and does not retrain or retune the model.

## Project Structure

```text
smart_leaf_analyzer/
├── core/                          # Shared preprocessing, model and validation code
├── dashboard/                     # Streamlit application
├── dataset/                       # Four source spectral datasets
├── notebook/
│   ├── tipburn_pipeline_notebook.ipynb
├── outputs/
│       ├── development_reference.npz
│       ├── final_tipburn_model_metadata.json
│       ├── final_tipburn_pipeline.joblib
│       └── tipburn_model_validation_report.html
├── README.md
└── requirements.txt
```

## Main Files

- `notebook/tipburn_pipeline_notebook.ipynb` contains the complete model-development, validation and export workflow.
- `dashboard/app.py` contains the Streamlit dashboard.
- `outputs` contains the locked model artefacts required by the dashboard.
- `tipburn_model_validation_report.html` contains the complete standalone validation results.

## Run the Dashboard

Python 3.11 or above is required.

From the project root:

```bash
python -m streamlit run dashboard/app.py
```

Required Python packages are listed in `requirements.txt`.

## Locked Model

The deployed classifier uses:

- Logistic Regression
- `C = 10`
- decision threshold `0.5`
- positive class: `Tipburn`
- prediction unit: one lettuce leaf
- grouping unit: one lettuce plant
- five input features:
  - `REIP_nm`
  - `NDRE`
  - `Red_Edge_Slope`
  - `NDVI`
  - `NIR_Mean_SmoothedRaw`
