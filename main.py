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

TN_RAINFALL = {
    "chennai": 1870.6,
    "kancheepuram": 1428.9,
    "chengalpattu": 1392.4,
    "tiruvallur": 1682.9,
    "cuddalore": 1454.4,
    "villupuram": 1622.7,
    "kallakurichi": 1290.6,
    "vellore": 1288.0,
    "ranipet": 1457.8,
    "tirupathur": 1143.3,
    "tiruvannamalai": 1470.9,
    "salem": 1245.4,
    "namakkal": 928.7,
    "dharmapuri": 1139.7,
    "krishnagiri": 1146.9,
    "coimbatore": 1989.6,
    "tiruppur": 846.7,
    "erode": 807.5,
    "tiruchirapalli": 952.6,
    "trichy": 952.6,
    "karur": 796.5,
    "perambalur": 922.8,
    "ariyalur": 1137.9,
    "pudukkottai": 1050.8,
    "thanjavur": 80.3,
    "tiruvarur": 1478.4,
    "nagapattinam": 1675.8,
    "mayiladuthurai": 1438.1,
    "madurai": 1038.9,
    "theni": 1087.2,
    "dindigul": 1175.3,
    "ramanathapuram": 1107.6,
    "virudhunagar": 998.9,
    "sivagangai": 1246.3,
    "tirunelveli": 1914.3,
    "tenkasi": 1246.1,
    "thoothukudi": 681.3,
    "nilgiris": 2382.9,
    "kanniyakumari": 1566.5,
}

def get_annual_rainfall(location: str):
    location_lower = location.lower().strip()
    for district, rainfall in TN_RAINFALL.items():
        if district in location_lower:
            return rainfall
    return 1299.5  # State average default



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

    print("Loading disease model...")

    import torch

    if not os.path.exists(DISEASE_MODEL_PATH):
        raise FileNotFoundError(
            f"Disease model not found: {DISEASE_MODEL_PATH}"
        )

    _disease_model = torch.load(
        DISEASE_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    _disease_model.eval()

    print("Disease model loaded.")

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


@app.get("/soil")
def soil(lat: float, lon: float):
    result = get_soil_data(lat, lon)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


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
        "rainfall": get_annual_rainfall(input.location)  
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







@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await file.read()

    try:
        import tempfile
        import os
        from gradio_client import Client, handle_file

        # Save image temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # Call Hugging Face Space
        client = Client("Hemavarshini2525/agritech-disease-detection")
        result = client.predict(
            image=handle_file(tmp_path),
            api_name="/predict_disease"
        )

        # Clean up temp file
        os.unlink(tmp_path)

        return {
            "disease_prediction": result,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Disease prediction failed: {str(e)}"
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Disease detection timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")
    
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

