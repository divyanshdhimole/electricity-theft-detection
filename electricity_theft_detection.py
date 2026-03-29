"""
Electricity Theft Detection using Machine Learning
===================================================
Author: [Your Name]
Subject: Artificial Intelligence & Machine Learning
Institution: VIT

Problem Statement:
    Electricity theft causes billions of dollars in losses globally every year.
    Distribution companies struggle to identify which consumers are illegally
    tampering with meters or bypassing them entirely. This project applies
    ML classification techniques to detect anomalous consumption patterns
    that indicate potential electricity theft.

Approach:
    - Generate synthetic smart meter data simulating normal and theft behavior
    - Apply feature engineering on consumption time-series data
    - Train multiple classifiers (Random Forest, Logistic Regression, SVM)
    - Evaluate and compare model performance
    - Visualize results and key features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_smart_meter_data(n_customers=500, n_days=60, random_state=42):
    """
    Generate synthetic smart meter data for normal and theft consumers.

    Normal consumers  → consistent, slightly seasonal daily usage patterns.
    Theft consumers   → sudden drops, irregular spikes, unusual night usage,
                        negative readings (meter tamper).

    Returns a DataFrame with one row per customer and aggregated features.
    """
    np.random.seed(random_state)
    records = []

    for cust_id in range(n_customers):
        # 70% normal, 30% theft
        is_theft = 1 if cust_id >= int(n_customers * 0.70) else 0

        # Base daily consumption (kWh) — residential profile
        base_consumption = np.random.uniform(5, 20)

        daily_readings = []
        for day in range(n_days):
            if is_theft:
                theft_type = np.random.choice(['tamper', 'bypass', 'irregular'])
                if theft_type == 'tamper':
                    # Meter under-records: consumption suddenly drops
                    reading = base_consumption * np.random.uniform(0.1, 0.4)
                elif theft_type == 'bypass':
                    # Completely bypassed — near-zero reading
                    reading = np.random.uniform(0, 1.5)
                else:
                    # Irregular: random spikes and drops
                    reading = base_consumption * np.random.uniform(0, 2.5)
                    if np.random.random() < 0.15:
                        reading = -reading  # Negative (impossible for honest meter)
            else:
                # Normal: slight day-of-week variation + Gaussian noise
                dow_factor = 1.0 if day % 7 < 5 else 1.3  # Higher on weekends
                reading = base_consumption * dow_factor + np.random.normal(0, 1.5)
                reading = max(reading, 0)  # Can't be negative for honest consumer

            daily_readings.append(reading)

        daily = np.array(daily_readings)

        # ── Feature Engineering ──────────────────────────────────────────────
        record = {
            'customer_id': cust_id,
            'label': is_theft,  # 0 = Normal, 1 = Theft

            # Statistical features
            'mean_consumption': daily.mean(),
            'std_consumption': daily.std(),
            'min_consumption': daily.min(),
            'max_consumption': daily.max(),
            'median_consumption': np.median(daily),
            'skewness': pd.Series(daily).skew(),
            'kurtosis': pd.Series(daily).kurt(),

            # Anomaly indicators
            'negative_readings_count': (daily < 0).sum(),
            'zero_readings_count': (daily == 0).sum(),
            'coeff_of_variation': daily.std() / (daily.mean() + 1e-6),

            # Trend features
            'consumption_trend': np.polyfit(range(n_days), daily, 1)[0],
            'max_single_day_drop': np.diff(daily).min(),
            'max_single_day_spike': np.diff(daily).max(),

            # Ratio features
            'low_consumption_days': (daily < base_consumption * 0.3).sum(),
            'high_consumption_days': (daily > base_consumption * 1.8).sum(),
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Dataset created: {len(df)} customers, "
          f"{df['label'].sum()} theft cases ({df['label'].mean()*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda(df, save_path="eda_plots.png"):
    """Generate EDA visualisations comparing normal vs theft consumers."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Electricity Theft Detection — EDA", fontsize=16, fontweight='bold')

    normal = df[df['label'] == 0]
    theft  = df[df['label'] == 1]
    colors = ['#2196F3', '#F44336']  # blue=normal, red=theft

    features_to_plot = [
        ('mean_consumption',        'Mean Daily Consumption (kWh)'),
        ('std_consumption',         'Std Dev of Consumption'),
        ('coeff_of_variation',      'Coefficient of Variation'),
        ('negative_readings_count', 'Negative Readings Count'),
        ('low_consumption_days',    'Low Consumption Days'),
        ('consumption_trend',       'Consumption Trend Slope'),
    ]

    for ax, (feat, title) in zip(axes.flatten(), features_to_plot):
        ax.hist(normal[feat], bins=25, alpha=0.7, color=colors[0], label='Normal', density=True)
        ax.hist(theft[feat],  bins=25, alpha=0.7, color=colors[1], label='Theft',  density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"EDA plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'mean_consumption', 'std_consumption', 'min_consumption', 'max_consumption',
    'median_consumption', 'skewness', 'kurtosis', 'negative_readings_count',
    'zero_readings_count', 'coeff_of_variation', 'consumption_trend',
    'max_single_day_drop', 'max_single_day_spike',
    'low_consumption_days', 'high_consumption_days',
]


def build_models():
    """Return dict of sklearn Pipeline models."""
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=42))
        ]),
    }


def evaluate_models(df):
    """Train, evaluate, and return results for all models."""
    X = df[FEATURE_COLS].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = build_models()
    results = {}

    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cv  = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc').mean()

        results[name] = {
            'model': pipe, 'y_pred': y_pred, 'y_proba': y_proba,
            'y_test': y_test, 'accuracy': acc, 'auc': auc, 'cv_auc': cv
        }

        print(f"\n{name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  CV AUC   : {cv:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Theft']))

    return results, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESULTS VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results, save_path="results_plots.png"):
    """Confusion matrices, ROC curves, and model comparison bar chart."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Electricity Theft Detection — Model Results", fontsize=16, fontweight='bold')

    model_names = list(results.keys())
    n = len(model_names)
    palette = plt.cm.tab10.colors

    # ── Confusion Matrices (top row) ─────────────────────────────────────────
    for i, name in enumerate(model_names):
        ax = fig.add_subplot(3, n, i + 1)
        cm = confusion_matrix(results[name]['y_test'], results[name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Theft'],
                    yticklabels=['Normal', 'Theft'])
        ax.set_title(f'{name}\nAcc={results[name]["accuracy"]:.3f}', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    # ── ROC Curves (middle row) ──────────────────────────────────────────────
    ax_roc = fig.add_subplot(3, 1, 2)
    for i, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(results[name]['y_test'], results[name]['y_proba'])
        ax_roc.plot(fpr, tpr, color=palette[i], lw=2,
                    label=f"{name} (AUC={results[name]['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves — All Models')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)

    # ── Model Comparison Bar Chart (bottom row) ──────────────────────────────
    ax_bar = fig.add_subplot(3, 1, 3)
    x = np.arange(n)
    width = 0.25
    acc_vals = [results[m]['accuracy'] for m in model_names]
    auc_vals = [results[m]['auc']      for m in model_names]
    cv_vals  = [results[m]['cv_auc']   for m in model_names]
    ax_bar.bar(x - width, acc_vals, width, label='Accuracy',  color='#42A5F5')
    ax_bar.bar(x,         auc_vals, width, label='ROC-AUC',   color='#66BB6A')
    ax_bar.bar(x + width, cv_vals,  width, label='CV AUC',    color='#FFA726')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(model_names, rotation=10)
    ax_bar.set_ylabel('Score')
    ax_bar.set_ylim(0.5, 1.05)
    ax_bar.set_title('Model Performance Comparison')
    ax_bar.legend()
    ax_bar.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved → {save_path}")


def plot_feature_importance(results, save_path="feature_importance.png"):
    """Feature importance from the Random Forest model."""
    rf_pipe = results['Random Forest']['model']
    rf_clf  = rf_pipe.named_steps['clf']
    importances = pd.Series(rf_clf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#F44336' if importances[f] > importances.median() else '#2196F3'
              for f in importances.index]
    importances.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    red_patch  = mpatches.Patch(color='#F44336', label='Above Median (High Impact)')
    blue_patch = mpatches.Patch(color='#2196F3', label='Below Median (Lower Impact)')
    ax.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ELECTRICITY THEFT DETECTION — AI/ML PROJECT")
    print("=" * 60)

    # Step 1: Generate data
    df = generate_smart_meter_data(n_customers=600, n_days=60)

    # Step 2: EDA
    plot_eda(df)

    # Step 3: Train & evaluate
    results, X_train, X_test, y_train, y_test = evaluate_models(df)

    # Step 4: Visualise results
    plot_results(results)
    plot_feature_importance(results)

    # Step 5: Summary
    best_model = max(results, key=lambda m: results[m]['auc'])
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model}")
    print(f"  ROC-AUC  : {results[best_model]['auc']:.4f}")
    print(f"  Accuracy : {results[best_model]['accuracy']:.4f}")
    print("="*60)
    print("\nAll plots saved. Project complete!")


if __name__ == '__main__':
    main()
