# Irrigation Type Classifier — Model Training Module

Part of **Real-Time AI Crop Advisory System for Farmers**.
This module trains a model that predicts the recommended **irrigation type** (Drip / Flood / Sprinkler / Manual) from crop and field conditions.

## Files in this folder

| File | Purpose |
|---|---|
| `train_irrigation_model.py` | Full training pipeline — run this to (re)train |
| `irrigation_dataset.csv` | Training data (6,000 rows) |
| `irrigation_model.pkl` | **Trained model** (Random Forest) — load this in the app |
| `label_encoders.pkl` | Encoders for categorical inputs (crop, soil, region, season, groundwater) |
| `target_encoder.pkl` | Encoder/decoder for the irrigation type labels |
| `irrigation_model_results.png` | Confusion matrix + feature importance chart |
| `requirements.txt` | Python dependencies |

## How to run / retrain

```bash
pip install -r requirements.txt
python train_irrigation_model.py
```

This regenerates the `.pkl` files and the results chart from scratch.

## Current model performance

- **Algorithm:** Random Forest (200 trees, max_depth=12, class_weight=balanced) — selected over Gradient Boosting after comparison
- **Test Accuracy:** 61.0%
- **Weighted F1:** 0.612
- **5-Fold CV Accuracy:** 61.8% ± 1.8%

**Per-class performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Drip | 0.76 | 0.68 | 0.71 |
| Flood | 0.72 | 0.69 | 0.70 |
| Sprinkler | 0.42 | 0.57 | 0.48 |
| Manual | 0.20 | 0.14 | 0.16 |

The model is strong on Drip and Flood, weaker on Sprinkler, and weakest on Manual. This is mainly a **data problem, not a tuning problem**: Manual only has 515 of 6,000 rows (~8.6%), so the model rarely sees enough examples to learn it well. See "Next steps" below.

**Most important features for the prediction:** `crop_type` (33%), `farm_size_acres` (15%), then `rainfall_mm`, `soil_moisture_percent`, `humidity_percent`, `temperature_C` (~10% each).

## How to use the model in the app

```python
import joblib
import pandas as pd

model = joblib.load('irrigation_model.pkl')
encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

cat_cols = ['crop_type', 'soil_type', 'region', 'season', 'groundwater_availability']
num_cols = ['farm_size_acres', 'temperature_C', 'rainfall_mm', 'soil_moisture_percent', 'humidity_percent']

def predict_irrigation(crop, soil, region, season, farm_size, temp, rainfall, soil_moisture, humidity, groundwater):
    row = {}
    for col, val in zip(cat_cols, [crop, soil, region, season, groundwater]):
        row[col] = encoders[col].transform([val])[0]
    for col, val in zip(num_cols, [farm_size, temp, rainfall, soil_moisture, humidity]):
        row[col] = val
    x = pd.DataFrame([row])[cat_cols + num_cols]
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    return target_encoder.inverse_transform([pred])[0], proba.max() * 100

label, confidence = predict_irrigation("Rice", "Clay", "Humid", "Kharif", 5.0, 30, 900, 60, 80, "High")
print(label, confidence)
```

Valid category values come from the dataset itself, e.g.:
- `crop_type`: Rice, Wheat, Cotton, Vegetable, ... (check `irrigation_dataset.csv` for the full list)
- `groundwater_availability`: Low, Medium, High

## Next steps to improve accuracy

1. **More Manual-irrigation examples** — this is the single biggest lever. The class is underrepresented; real or synthetically augmented Manual-irrigation rows would help most.
2. **Try SMOTE or oversampling** on the Manual/Sprinkler classes before training.
3. **Hyperparameter search** (e.g., `GridSearchCV` / `Optuna`) on Random Forest / try XGBoost — likely a few extra points, not a breakthrough, given current data.
4. **Add more discriminating features** if available from the field app — e.g., soil pH, irrigation infrastructure already installed, water source distance — these often separate Manual vs Sprinkler more cleanly than weather alone.
