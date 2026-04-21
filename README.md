# Predicting Post-Graduation Earnings with Machine Learning

**COMP 560 — Artificial Intelligence | UNC Chapel Hill**
**Authors:** Gili Horwitz, Ethan Ahdout, John Sebastian, Daniel Kane

## Overview
This project predicts median student earnings 10 years after enrollment using the U.S. Department of Education College Scorecard dataset. We train a Linear Regression baseline and a Feedforward Neural Network (FFN), compare their performance, and deploy the FFN behind a Gradio web interface that calls the Claude API (Anthropic) to generate natural-language career insights.

## Methods
- **Linear Regression** — MSE loss, standardized features
- **Feedforward Neural Network** — 3 hidden layers, ReLU, Dropout, trained via backpropagation for 150 epochs
- **Frontend** — Gradio UI + Claude API for salary prediction and career advice

## Results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $6,596 | $8,512 | 0.587 |
| FFN | $6,148 | $8,101 | **0.626** |

## Repository Files
- `college_salary_prediction.ipynb` — full pipeline: load, preprocess, train, evaluate, demo
- `college_scorecard_clean.csv` — cleaned 950-institution dataset (bundled for fast reproducibility)
- `ffn_model.pt` — saved FFN weights (load with `torch.load()`)
- `lr_model.pkl` — saved Linear Regression model
- `scaler.pkl` — fitted StandardScaler for preprocessing new inputs
- `requirements.txt` — pinned dependency versions
- `outputs/` — generated plots and figures
- `report.pdf` — 4-page NeurIPS-style conference paper
- `slides.pdf` — 3-slide presentation

## Running the Code

**Option 1 — Google Colab:** Open the notebook, upload `college_scorecard_clean.csv` via the file browser, then Runtime → Run all.

**Option 2 — Local Jupyter:**
```bash
git clone <this-repo>
pip install -r requirements.txt
jupyter notebook college_salary_prediction.ipynb
```

Saved models let you skip training entirely — run evaluation cells only. To rebuild from raw data, download from [collegescorecard.ed.gov/data](https://collegescorecard.ed.gov/data/).
