"""
Crop Recommendation — EDA + Model Training + Pickle Export
Run: python3.11 eda_train.py
"""
import os, pickle, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, "data", "crop_recommendation.xlsx")
EDA_DIR    = os.path.join(BASE, "eda_outputs")
MODEL_DIR  = os.path.join(BASE, "models")
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

 
print("=" * 60)
print("  CROP RECOMMENDATION — EDA & MODEL TRAINING")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"\n[1] Raw shape: {df.shape}")
print(f"    Columns  : {df.columns.tolist()}")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"\n[2] Missing values before imputation:\n{df.isnull().sum()}")
for col in FEATURES:
    if df[col].isnull().any():
        median = df[col].median()
        df[col] = df[col].fillna(median)
        print(f"    Filled '{col}' missing with median={median:.3f}")

print(f"\n[3] Missing values after imputation:\n{df.isnull().sum()}")
print(f"\n[4] Descriptive statistics:\n{df[FEATURES].describe().round(3)}")
print(f"\n[5] Class distribution:\n{df[TARGET].value_counts()}")

# ─── 2. EDA PLOTS ──────────────────────────────────────────────────────────
print("\n[6] Generating EDA plots...")

# 6a. Class distribution
fig, ax = plt.subplots(figsize=(14, 5))
df[TARGET].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Crop Class Distribution", fontsize=14)
ax.set_xlabel("Crop"); ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "1_class_distribution.png"), dpi=120)
plt.close(fig)

# 6b. Feature histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    axes[i].hist(df[col], bins=30, color="mediumseagreen", edgecolor="white")
    axes[i].set_title(col, fontsize=12)
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "2_feature_histograms.png"), dpi=120)
plt.close(fig)

# 6c. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "3_correlation_heatmap.png"), dpi=120)
plt.close(fig)

# 6d. Boxplots — top 8 crops, 4 key features
top8 = df[TARGET].value_counts().index[:8].tolist()
sub  = df[df[TARGET].isin(top8)]
key_feats = ["N", "K", "temperature", "rainfall"]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for i, feat in enumerate(key_feats):
    sub.boxplot(column=feat, by=TARGET, ax=axes[i])
    axes[i].set_title(f"{feat} by Crop (top 8)", fontsize=11)
    axes[i].set_xlabel(""); axes[i].tick_params(axis="x", rotation=45)
plt.suptitle("", fontsize=1)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "4_boxplots_by_crop.png"), dpi=120)
plt.close(fig)

# 6e. Pairplot (sampled 400 rows, 4 features)
sample = df.sample(400, random_state=42)[["N", "temperature", "humidity", "rainfall", TARGET]]
pair_fig = sns.pairplot(sample, hue=TARGET, plot_kws={"alpha": 0.5, "s": 20})
pair_fig.fig.suptitle("Pairplot (sample 400)", y=1.02, fontsize=12)
pair_fig.savefig(os.path.join(EDA_DIR, "5_pairplot.png"), dpi=90)
plt.close("all")

print(f"    EDA plots saved to: {EDA_DIR}")

# ─── 3. PREPROCESSING ───────────────────────────────────────────────────────
print("\n[7] Preprocessing...")
le = LabelEncoder()
y  = le.fit_transform(df[TARGET])
X  = df[FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"    Classes ({len(le.classes_)}): {list(le.classes_)}")

# ─── 4. TRAIN & EVALUATE ────────────────────────────────────────────────────
models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "KNN":                  KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)":            SVC(C=10, probability=True, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = []
print(f"\n[8] Training & evaluating {len(models)} models...\n")
fmt = "{:<22} {:>8} {:>10} {:>9} {:>8}  {:>15}"
print(fmt.format("Model", "Acc", "Precision", "Recall", "F1", "CV F1 (5-fold)"))
print("-" * 80)

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="f1_macro", n_jobs=-1)
    cv_str = f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"

    results.append({"name": name, "clf": clf, "acc": acc, "prec": prec,
                    "rec": rec, "f1": f1, "cv_mean": cv_scores.mean()})
    print(fmt.format(name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", cv_str))

print("-" * 80)

# ─── 5. BEST MODEL ──────────────────────────────────────────────────────────
best = max(results, key=lambda r: r["f1"])
print(f"\n[9] Best model: {best['name']}  (Test F1-macro = {best['f1']:.4f})")

print(f"\n    Classification Report — {best['name']}:")
y_pred_best = best["clf"].predict(X_test)
print(classification_report(y_test, y_pred_best,
                             target_names=le.classes_, zero_division=0))

# Confusion matrix
cm  = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title(f"Confusion Matrix — {best['name']}", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "6_confusion_matrix.png"), dpi=120)
plt.close(fig)
print(f"    Confusion matrix saved.")

# ─── 6. SAVE ARTIFACTS ──────────────────────────────────────────────────────
model_path  = os.path.join(MODEL_DIR, "crop_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
le_path     = os.path.join(MODEL_DIR, "label_encoder.pkl")

with open(model_path,  "wb") as f: pickle.dump(best["clf"], f)
with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
with open(le_path,     "wb") as f: pickle.dump(le, f)

print(f"\n[10] Artifacts saved:")
print(f"     {model_path}")
print(f"     {scaler_path}")
print(f"     {le_path}")
print("\n Done! Run app.py to start the Flask server.\n")
