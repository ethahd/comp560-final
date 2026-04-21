# Predicting Post-Graduation Earnings with Machine Learning

**COMP 560 — Artificial Intelligence | UNC Chapel Hill**
**Authors:** Gili Horwitz

## Overview
This project predicts median student earnings 10 years after
enrollment using the U.S. Department of Education College Scorecard
dataset. We train a Linear Regression baseline and a Feedforward
Neural Network (FFN) with backpropagation, compare their performance,
and deploy the FFN behind a Gradio web interface that calls the Claude
API (Anthropic) to generate natural-language career insights.

## Methods
- **Linear Regression** — MSE loss, standardized features (W8)
- **Feedforward Neural Network** — 3 hidden layers, ReLU, Dropout,
  trained via backpropagation for 150 epochs (W10, W11)
- **Frontend** — Gradio UI + Claude API for salary prediction and
  natural language career advice (W13)

## Results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $6,596 | $8,512 | 0.587 |
| FFN | $6,148 | $8,101 | **0.626** |

## Running the Code

### Option 1 — Google Colab (recommended)
1. Open `college_salary_prediction.ipynb` in Colab
2. Upload `college_scorecard_clean.csv` to the Colab file browser
3. Runtime → Run all

### Option 2 — Local Jupyter
```bash
git clone <this-repo>
cd COMP560-Project
pip install -r requirements.txt
jupyter notebook college_salary_prediction.ipynb
```

The bundled CSV contains the cleaned 950-institution dataset. To
rebuild from raw data, download from
[collegescorecard.ed.gov/data](https://collegescorecard.ed.gov/data/)
and place `college-scorecard-institution.csv.gz` in the repo root.
