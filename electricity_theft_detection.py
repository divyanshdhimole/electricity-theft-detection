# Electricity Theft Detection
# Name: Divyansh Dhimole

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# 1. Generate Simple Data
# -----------------------------
def generate_data(n=200):
    data = []

    for i in range(n):
        if i < int(0.7 * n):
            # Normal user
            consumption = np.random.normal(15, 2, 30)  # stable usage
            label = 0
        else:
            # Theft user
            consumption = np.random.normal(5, 5, 30)  # irregular usage
            if np.random.rand() < 0.3:
                consumption[np.random.randint(0, 30)] = -5  # fake reading
            label = 1

        data.append([
            consumption.mean(),
            consumption.std(),
            consumption.min(),
            label
        ])

    df = pd.DataFrame(data, columns=[
        'mean', 'std', 'min', 'label'
    ])

    return df


# -----------------------------
# 2. Prepare Data
# -----------------------------
df = generate_data()

X = df[['mean', 'std', 'min']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 3. Train Models
# -----------------------------
lr = LogisticRegression()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -----------------------------
# 4. Predictions
# -----------------------------
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# -----------------------------
# 5. Results
# -----------------------------
print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_pred))


# -----------------------------
# 6. Simple Plot
# -----------------------------
plt.scatter(df['mean'], df['std'], c=df['label'])
plt.xlabel("Mean Consumption")
plt.ylabel("Standard Deviation")
plt.title("Electricity Theft Detection")
plt.show()
