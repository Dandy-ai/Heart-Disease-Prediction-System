# =============================================================================
# Heart Disease Prediction System - Model Training Script
# =============================================================================
# Run this script FIRST before starting the web application.
# It trains all 6 ML algorithms, evaluates them, prints the comparison table,
# and saves the best-performing model to the models/ folder.
#
# Usage:  python train_models.py
# =============================================================================

import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

# ── 1. Load Dataset ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  HEART DISEASE PREDICTION SYSTEM - MODEL TRAINING")
print("="*65)

DATA_PATH = os.path.join('data', 'heart.csv')

if not os.path.exists(DATA_PATH):
    print("\n[ERROR] Dataset not found at data/heart.csv")
    print("\n  Please download the dataset:")
    print("  1. Go to: https://www.kaggle.com/datasets/ronitf/heart-disease-uci")
    print("  2. Download heart.csv")
    print("  3. Place it in the 'data' folder of this project")
    print("  4. Re-run this script\n")
    exit(1)

print(f"\n[1/6] Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"      Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Preprocessing ────────────────────────────────────────────────────────
print("\n[2/6] Preprocessing data...")

# Drop rows with missing values
df.dropna(inplace=True)
print(f"      After dropping missing values: {df.shape[0]} records")

# Ensure target column is named 'target' (binary: 0 or 1)
# Some versions of the dataset use 'condition' or have values > 1
if 'target' not in df.columns:
    # Try common alternative column names
    for col in ['condition', 'num', 'class']:
        if col in df.columns:
            df.rename(columns={col: 'target'}, inplace=True)
            break

# Binarise target: some versions have 0-4 instead of 0-1
df['target'] = (df['target'] > 0).astype(int)

# Separate features and target
FEATURE_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = df[FEATURE_COLS]
y = df['target']

class_counts = y.value_counts()
print(f"      Class distribution: No Disease={class_counts[0]}, "
      f"Disease={class_counts[1]}")

# ── 3. Train-Test Split ──────────────────────────────────────────────────────
print("\n[3/6] Splitting dataset (70% train / 30% test, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"      Training set: {len(X_train)} records")
print(f"      Test set:     {len(X_test)} records")

# ── 4. Feature Scaling ───────────────────────────────────────────────────────
print("\n[4/6] Applying StandardScaler (fit on training set only)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save scaler immediately
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
print("      Scaler saved to models/scaler.pkl")

# ── 5. Define Models ─────────────────────────────────────────────────────────
models = {
    'Logistic Regression':    LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree':          DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors':    KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes':            GaussianNB(),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
}

# ── 6. Train and Evaluate ────────────────────────────────────────────────────
print("\n[5/6] Training and evaluating all 6 algorithms...\n")

results = []
best_accuracy = 0
best_model_name = ''
best_model_obj  = None

SEP = "─" * 95

header = (f"{'Algorithm':<28} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10} {'Specificity':>12} {'CV Mean':>10}")
print(header)
print(SEP)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp = cm[0][0], cm[0][1]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10)
    cv_mean = cv_scores.mean()

    results.append({
        'Algorithm':   name,
        'Accuracy':    acc,
        'Precision':   prec,
        'Recall':      rec,
        'F1-Score':    f1,
        'Specificity': spec,
        'CV Mean':     cv_mean,
        'Model':       model,
    })

    row = (f"{name:<28} {acc*100:>9.2f}% {prec*100:>9.2f}% "
           f"{rec*100:>9.2f}% {f1*100:>9.2f}% {spec*100:>11.2f}% "
           f"{cv_mean*100:>9.2f}%")
    print(row)

    if acc > best_accuracy:
        best_accuracy    = acc
        best_model_name  = name
        best_model_obj   = model

print(SEP)

# ── 7. Save Best Model ───────────────────────────────────────────────────────
print(f"\n  Best Model : {best_model_name}")
print(f"  Accuracy   : {best_accuracy*100:.2f}%")

joblib.dump(best_model_obj, os.path.join('models', 'best_model.pkl'))

# Also save a mapping of algorithm name → model file for user-selectable predictions
all_models_dict = {r['Algorithm']: r['Model'] for r in results}
joblib.dump(all_models_dict, os.path.join('models', 'all_models.pkl'))

# Save feature names for validation
joblib.dump(FEATURE_COLS, os.path.join('models', 'feature_cols.pkl'))

# Save best model name for Flask to reference
with open(os.path.join('models', 'best_model_name.txt'), 'w') as f:
    f.write(best_model_name)

print(f"\n  Saved to  : models/best_model.pkl")
print(f"  Saved to  : models/all_models.pkl")
print(f"  Saved to  : models/scaler.pkl")

# ── 8. Confusion Matrix Plots ────────────────────────────────────────────────
print("\n[6/6] Generating confusion matrix plots...")
os.makedirs(os.path.join('static', 'img'), exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, r in enumerate(results):
    model = r['Model']
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[i].set_title(f"{r['Algorithm']}\nAccuracy: {r['Accuracy']*100:.2f}%",
                      fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.suptitle('Confusion Matrices – All 6 Algorithms', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('static', 'img', 'confusion_matrices.png'),
            dpi=120, bbox_inches='tight')
plt.close()
print("      Saved to static/img/confusion_matrices.png")

# ── Accuracy Comparison Bar Chart ──────────────────────────────────────────
names = [r['Algorithm'] for r in results]
accs  = [r['Accuracy'] * 100 for r in results]
colors = ['#3498DB' if n != best_model_name else '#2ECC71' for n in names]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=0.8)
ax.set_ylim(60, 100)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Algorithm Accuracy Comparison – Cleveland Heart Disease Dataset',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=20)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(y=best_accuracy*100, color='#E74C3C', linestyle='--',
           linewidth=1.2, alpha=0.7, label=f'Best: {best_accuracy*100:.2f}%')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join('static', 'img', 'accuracy_comparison.png'),
            dpi=120, bbox_inches='tight')
plt.close()
print("      Saved to static/img/accuracy_comparison.png")

# ── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TRAINING COMPLETE")
print("="*65)
print(f"\n  Best performing algorithm : {best_model_name}")
print(f"  Test set accuracy          : {best_accuracy*100:.2f}%")
print(f"\n  You can now run the web application:")
print(f"  > python app.py")
print(f"  Then open your browser to: http://127.0.0.1:5000\n")
