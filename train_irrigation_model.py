# ============================================================
# Irrigation Type Classifier - Full Training Pipeline
# Real-Time AI Crop Advisory System
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import warnings, joblib
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv('irrigation_dataset.csv')
print("=" * 55)
print("IRRIGATION TYPE CLASSIFICATION - TRAINING PIPELINE")
print("=" * 55)
print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nTarget distribution:")
print(df['irrigation_type'].value_counts())

# ── 2. PREPROCESSING ──────────────────────────────────────────
cat_cols = ['crop_type', 'soil_type', 'region', 'season', 'groundwater_availability']
num_cols = ['farm_size_acres', 'temperature_C', 'rainfall_mm',
            'soil_moisture_percent', 'humidity_percent']

le = {}
df_enc = df.copy()
for col in cat_cols:
    le[col] = LabelEncoder()
    df_enc[col] = le[col].fit_transform(df[col])

le_target = LabelEncoder()
df_enc['irrigation_type'] = le_target.fit_transform(df['irrigation_type'])

X = df_enc[cat_cols + num_cols]
y = df_enc['irrigation_type']

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── 4. MODELS ─────────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_leaf=5, class_weight='balanced',
        random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=5,
        learning_rate=0.1, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    cv  = cross_val_score(model, X, y, cv=StratifiedKFold(5),
                          scoring='accuracy', n_jobs=-1)
    results[name] = {'model': model, 'y_pred': y_pred,
                     'acc': acc, 'f1': f1, 'cv_mean': cv.mean(), 'cv_std': cv.std()}
    print(f"\n{name}:")
    print(f"  Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Weighted F1    : {f1:.4f}")
    print(f"  5-Fold CV      : {cv.mean():.4f} ± {cv.std():.4f}")

# ── 5. BEST MODEL ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best      = results[best_name]
print(f"\n✅ Best Model: {best_name} (F1={best['f1']:.4f})")

print("\nDetailed Classification Report:")
print(classification_report(y_test, best['y_pred'],
      target_names=le_target.classes_))

# ── 6. FEATURE IMPORTANCE ─────────────────────────────────────
fi = pd.Series(best['model'].feature_importances_,
               index=cat_cols + num_cols).sort_values(ascending=False)
print("\nTop Feature Importances:")
print(fi.to_string())

# ── 7. PLOTS ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_, ax=axes[0])
axes[0].set_title(f'{best_name} - Confusion Matrix')
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

# Feature importance
fi.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Feature Importance'); axes[1].invert_yaxis()
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('irrigation_model_results.png', dpi=120, bbox_inches='tight')
print("\n📊 Plot saved: irrigation_model_results.png")

# ── 8. SAVE MODEL & ENCODERS ──────────────────────────────────
joblib.dump(best['model'], 'irrigation_model.pkl')
joblib.dump(le, 'label_encoders.pkl')
joblib.dump(le_target, 'target_encoder.pkl')
print("💾 Model saved: irrigation_model.pkl")

# ── 9. PREDICTION FUNCTION (drop into your app) ───────────────
print("\n" + "="*55)
print("SAMPLE PREDICTION (ready to use in your app):")
print("="*55)

def predict_irrigation(crop, soil, region, season, farm_size,
                        temp, rainfall, soil_moisture, humidity, groundwater):
    row = {}
    for col, val in zip(cat_cols,
                        [crop, soil, region, season, groundwater]):
        row[col] = le[col].transform([val])[0]
    for col, val in zip(num_cols,
                        [farm_size, temp, rainfall, soil_moisture, humidity]):
        row[col] = val
    x = pd.DataFrame([row])[cat_cols + num_cols]
    pred = best['model'].predict(x)[0]
    proba = best['model'].predict_proba(x)[0]
    label = le_target.inverse_transform([pred])[0]
    conf  = proba.max() * 100
    return label, conf

# Example predictions
examples = [
    ("Rice",    "Clay",  "Humid",     "Kharif", 5.0, 30, 900, 60, 80, "High"),
    ("Cotton",  "Sandy", "Arid",      "Kharif", 8.0, 38, 80,  35, 45, "Low"),
    ("Wheat",   "Loam",  "Semi-Arid", "Rabi",   12.0,22, 200, 45, 55, "Medium"),
    ("Vegetable","Loam", "Humid",     "Zaid",   1.5, 28, 600, 50, 70, "High"),
]

for args in examples:
    label, conf = predict_irrigation(*args)
    print(f"  {args[0]:10s} | {args[2]:10s} | {args[9]:6s} gw → {label:12s} ({conf:.1f}% confident)")

print("\n✅ Training complete. All files ready.")
