# ⚡ Electricity Theft Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A machine learning system that detects electricity theft from smart meter consumption data by identifying anomalous usage patterns using classification algorithms.

---

## 📌 Problem Statement

Electricity theft is a widespread problem in India and globally. Consumers tamper with meters, bypass them entirely, or use illegal connections — causing significant revenue losses to distribution companies and increasing electricity costs for honest consumers. Manual inspection is slow and expensive.

This project applies Machine Learning to automatically flag suspicious consumers based on their daily consumption patterns extracted from smart meter data.

---

## 🎯 Approach

1. **Data** — Synthetic smart meter dataset simulating 600 consumers over 60 days (70% normal, 30% theft)
2. **Feature Engineering** — 15 statistical and anomaly features derived from time-series readings
3. **Models** — Logistic Regression, Random Forest, Gradient Boosting, and SVM compared
4. **Evaluation** — Accuracy, ROC-AUC, Confusion Matrix, and 5-fold Cross Validation

---

## 📁 Project Structure

```
electricity-theft-detection/
│
├── electricity_theft_detection.py   # Main ML pipeline (data → features → models → plots)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/                            # (Generated at runtime — no external dataset needed)
├── models/                          # Saved model artifacts (optional)
│
└── outputs/
    ├── eda_plots.png                # Exploratory Data Analysis visualisations
    ├── results_plots.png            # Confusion matrices, ROC curves, comparison
    └── feature_importance.png       # Random Forest feature importances
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/divyanshdhimole/electricity-theft-detection.git
cd electricity-theft-detection

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
python electricity_theft_detection.py
```

That's it! The script will:
- Generate the synthetic smart meter dataset
- Print EDA summary and model evaluation metrics to the console
- Save three plot files: `eda_plots.png`, `results_plots.png`, `feature_importance.png`

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `mean_consumption` | Average daily kWh usage |
| `std_consumption` | Variability in daily usage |
| `coeff_of_variation` | Relative variability (std / mean) |
| `negative_readings_count` | Impossible readings indicating tampering |
| `zero_readings_count` | Days with zero consumption |
| `skewness` / `kurtosis` | Shape of consumption distribution |
| `consumption_trend` | Rising or falling trend over time |
| `low_consumption_days` | Days far below baseline |
| `high_consumption_days` | Days far above baseline |
| `max_single_day_drop` | Largest sudden decrease |
| `max_single_day_spike` | Largest sudden increase |

---

## 🤖 Models & Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~97% | ~0.99 |
| Random Forest | ~98% | ~1.00 |
| Gradient Boosting | ~98% | ~1.00 |
| SVM | ~97% | ~0.99 |

> **Note:** Results may vary slightly on each run due to random data generation.
> On real-world data, expect lower scores — the dataset here is synthetic.

---

## 📈 Sample Output

After running, you will see:
- `eda_plots.png` — Side-by-side histograms showing how normal and theft consumers differ
- `results_plots.png` — Confusion matrices + ROC curves + bar chart comparing all models
- `feature_importance.png` — Which features the Random Forest found most useful

---

## 🛠️ Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Key Concepts Applied

- Binary Classification
- Feature Engineering from time-series data
- Model Evaluation (Accuracy, Precision, Recall, F1, AUC-ROC)
- Cross-Validation
- Data Visualization with Matplotlib & Seaborn
- Scikit-learn Pipelines

---

## 🔮 Future Improvements

- Use real SGCC (State Grid Corporation of China) public dataset
- Add LSTM / deep learning model for raw time-series input
- Build a simple web dashboard using Streamlit
- Integrate alert system for flagging suspicious consumers

---

## 👤 Author

**Divyansh Dhimole**  
B.Tech CSE | VIT  
Subject: Artificial Intelligence & Machine Learning  

---
