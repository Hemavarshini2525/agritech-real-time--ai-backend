import os
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weather import get_weather
from database import init_db, save_advisory, get_all_history
from soil import get_soil_data
from models import AdvisoryRecord, AdvisoryInput, PredictionInput, FertilizerInput

import io
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
import torch
import pandas as pd

app = FastAPI(title="AgriTech Backend API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_recommendation_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "disease_model.pkl")
DISEASE_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "disease_label_encoder.pkl")

_loaded_model = None
_loaded_encoder = None
_disease_model = None
_disease_encoder = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()





# ─── MODEL LOADERS ─────────────────────────────────────────

def load_model():
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    _loaded_model = joblib.load(MODEL_PATH)
    return _loaded_model

def load_encoder():
    global _loaded_encoder
    if _loaded_encoder is not None:
        return _loaded_encoder
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder file not found: {ENCODER_PATH}")
    _loaded_encoder = joblib.load(ENCODER_PATH)
    return _loaded_encoder

def load_disease_model():
    global _disease_model
    if _disease_model is not None:
        return _disease_model
    if not os.path.exists(DISEASE_MODEL_PATH):
        raise FileNotFoundError(f"Disease model not found: {DISEASE_MODEL_PATH}")
    try:
        # Load the full saved PyTorch model object.
        _disease_model = torch.load(DISEASE_MODEL_PATH, map_location="cpu", weights_only=False)
        if hasattr(_disease_model, "eval"):
            _disease_model.eval()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load disease model: {e}")
    return _disease_model

def load_disease_encoder():
    global _disease_encoder
    if _disease_encoder is not None:
        return _disease_encoder
    if not os.path.exists(DISEASE_ENCODER_PATH):
        raise FileNotFoundError(f"Disease label encoder not found: {DISEASE_ENCODER_PATH}")
    _disease_encoder = joblib.load(DISEASE_ENCODER_PATH)
    return _disease_encoder


# ─── IMAGE PREPROCESSING ───────────────────────────────────

def preprocess_image(image_data: bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((256, 256))

    width, height = image.size
    left = (width - target_size[0]) / 2
    top = (height - target_size[1]) / 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    image = image.crop((left, top, right, bottom))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


# ─── ROUTES ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AgriTech Backend is running ✅"}


@app.get("/weather")
def weather(location: str):
    result = get_weather(location)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/save-advisory")
def save(data: AdvisoryInput):
    weather_data = get_weather(data.location)
    weather_info = str(weather_data)
    record = {
        "farmer_query": data.farmer_query,
        "disease": data.disease,
        "fertilizer": data.fertilizer,
        "irrigation": data.irrigation,
        "weather_info": weather_info,
        "advisory_text": data.advisory_text
    }
    save_advisory(record)
    return {"message": "Advisory saved successfully ✅"}


@app.get("/history")
def history():
    rows = get_all_history()
    return [
        {
            "id": row[0],
            "farmer_query": row[1],
            "disease": row[2],
            "fertilizer": row[3],
            "irrigation": row[4],
            "weather_info": row[5],
            "advisory_text": row[6],
            "created_at": row[7]
        }
        for row in rows
    ]


@app.post("/predict")
def predict(input: PredictionInput):
    weather_data = get_weather(input.location)
    if "error" in weather_data:
        raise HTTPException(status_code=400, detail=weather_data["error"])

    latitude = weather_data.get("latitude")
    longitude = weather_data.get("longitude")
    if latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="Unable to get coordinates from weather data")

    soil_data = get_soil_data(latitude, longitude)
    soil_error = soil_data.get("error")

    # Prefer user-provided N and pH when available; otherwise use soil lookup or defaults
    N_val = input.nitrogen if input.nitrogen is not None else (soil_data.get("nitrogen") if soil_data.get("nitrogen") is not None else 50.0)
    ph_val = input.ph if input.ph is not None else (soil_data.get("ph") if soil_data.get("ph") is not None else 6.5)

    feature_dict = {
        "N": N_val,
        "P": input.phosphorus,
        "K": input.potassium,
        "temperature": weather_data.get("temperature"),
        "humidity": weather_data.get("humidity"),
        "ph": ph_val,
        "rainfall": weather_data.get("rain") if weather_data.get("rain") is not None else weather_data.get("precipitation")
    }

    try:
        model = load_model()
        encoder = load_encoder()
        X = [
            feature_dict["N"], feature_dict["P"], feature_dict["K"],
            feature_dict["temperature"], feature_dict["humidity"],
            feature_dict["ph"], feature_dict["rainfall"]
        ]
        prediction = model.predict([X])
        predicted_encoded = int(prediction[0])
        predicted_crop = str(encoder.inverse_transform([predicted_encoded])[0])

        response = {
            "prediction": predicted_crop,
            "prediction_encoded": predicted_encoded,
            "features": feature_dict,
            "weather": weather_data,
            "soil": soil_data
        }
        if soil_error:
            response["soil_error"] = soil_error
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/soil")
def soil(lat: float, lon: float):
    result = get_soil_data(lat, lon)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await file.read()
    try:
        model = load_disease_model()
        encoder = load_disease_encoder()
        X = preprocess_image(image_bytes, target_size=(224, 224))

        input_tensor = torch.from_numpy(X).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_index = int(np.argmax(probs))

        predicted_label = encoder.inverse_transform([predicted_index])[0]

        return {
            "disease_prediction": str(predicted_label),
            "prediction_index": predicted_index,
            "probabilities": probs.tolist()
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(exc)}")

@app.post("/fertilizer-recommendation")
def fertilizer_recommendation(data: FertilizerInput):
    model_path    = "fertilizer_model.pkl"
    encoders_path = "label_encoder_fertilizer.pkl"
    target_path   = "target_encoder.pkl"

    if not os.path.exists(model_path):
        return {
            "status": "model_not_ready",
            "message": "ML model not yet available.",
            "received_data": {
                "crop": data.crop_type,
                "soil": data.soil_type,
                "N": data.nitrogen,
                "P": data.phosphorous,
                "K": data.potassium
            }
        }

    model          = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)  # dictionary
    target_encoder = joblib.load(target_path)

    # Encode soil and crop using dictionary
    soil_encoded = label_encoders["Soil Type"].transform([data.soil_type])[0]
    crop_encoded = label_encoders["Crop Type"].transform([data.crop_type])[0]

    # 8 features with correct column names
    features = pd.DataFrame([[
        data.temperature,
        data.humidity,
        data.moisture,
        soil_encoded,
        crop_encoded,
        data.nitrogen,
        data.potassium,
        data.phosphorous,
        
    ]], columns=[
        "Temparature", "Humidity", "Moisture",
        "Soil Type", "Crop Type",
        "Nitrogen", "Potassium", "Phosphorous"
        
    ])

    prediction = model.predict(features)
    result     = target_encoder.inverse_transform(prediction)

    return {
        "status": "success",
        "recommended_fertilizer": result[0],
        "message": f"Apply {result[0]} based on your soil and crop data"
    }