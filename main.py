import os
import joblib
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weather import get_weather
from database import init_db, save_advisory, get_all_history
from soil import get_soil_data
from models import AdvisoryRecord, AdvisoryInput, PredictionInput, FertilizerInput, IrrigationInput
from tn_rainfall import get_seasonal_rainfall

import io
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
import pandas as pd
import httpx
import google.generativeai as genai

app = FastAPI(title="AgriTech Backend API")

load_dotenv()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_ensemble_model.pkl")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5")


def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(GEMINI_MODEL)


ENCODER_PATH = os.path.join(os.path.dirname(__file__), "crop_label_encoder.pkl")
DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "disease_model.pkl")
DISEASE_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "disease_label_encoder.pkl")

IRRIGATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "irrigation_model.pkl") 

_irrigation_model = None
_loaded_model = None
_loaded_encoder = None
_disease_model = None
_disease_encoder = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://localhost:5173"],  # frontend dev server
    allow_credentials=True,
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


def load_irrigation_model():
    global _irrigation_model

    if _irrigation_model is not None:
        return _irrigation_model

    if not os.path.exists(IRRIGATION_MODEL_PATH):
        raise FileNotFoundError(
            f"Irrigation model not found: {IRRIGATION_MODEL_PATH}"
        )

    _irrigation_model = joblib.load(IRRIGATION_MODEL_PATH)

    return _irrigation_model
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

    N_val = input.nitrogen if input.nitrogen is not None else (soil_data.get("nitrogen") if soil_data.get("nitrogen") is not None else 50.0)
    ph_val = input.ph if input.ph is not None else (soil_data.get("ph") if soil_data.get("ph") is not None else 6.5)

    feature_dict = {
        "N": N_val,
        "P": input.phosphorus,
        "K": input.potassium,
        "temperature": weather_data.get("temperature"),
        "humidity": weather_data.get("humidity"),
        "ph": ph_val,
        "rainfall": get_seasonal_rainfall(input.location, months=3)
    }

    try:
        model = load_model()
        encoder = load_encoder()
        scaler = joblib.load(os.path.join(os.path.dirname(__file__), "crop_scaler.pkl"))

        X = [[
            feature_dict["N"], feature_dict["P"], feature_dict["K"],
            feature_dict["temperature"], feature_dict["humidity"],
            feature_dict["ph"], feature_dict["rainfall"]
        ]]

        X_scaled = scaler.transform(X)          # ← scale cheyyuka

        # Try to provide top-3 crop recommendations when model supports probabilities
        top_3_crops = []
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            # model.classes_ should contain encoded labels used with the encoder
            classes = getattr(model, "classes_", None)
            if classes is not None:
                top_idx = np.argsort(proba)[::-1][:3]
                top_labels = np.array(classes)[top_idx]

                for lbl in top_labels:
                    # If the model's classes_ are encoded integers
                    if isinstance(lbl, (int, np.integer)):
                        enc = int(lbl)
                        try:
                            crop_name = encoder.inverse_transform([enc])[0]
                        except Exception:
                            crop_name = str(lbl)
                        top_3_crops.append(str(crop_name))
                    # If the model's classes_ are already crop names (strings)
                    elif isinstance(lbl, str):
                        top_3_crops.append(lbl)
                    else:
                        # Fallback: try to coerce to int then inverse transform
                        try:
                            enc = int(lbl)
                            crop_name = encoder.inverse_transform([enc])[0]
                            top_3_crops.append(str(crop_name))
                        except Exception:
                            top_3_crops.append(str(lbl))
        else:
            # Fallback: single prediction
            prediction = model.predict(X_scaled)
            predicted_encoded = int(prediction[0])
            predicted_crop = str(encoder.inverse_transform([predicted_encoded])[0])
            top_3_crops = [predicted_crop]

        # Use the top-1 crop as the primary prediction
        predicted_crop = top_3_crops[0] if top_3_crops else "Unknown"
        try:
            # Find the encoded value for the top-1 crop
            predicted_encoded = int(np.where(encoder.classes_ == predicted_crop)[0][0])
        except Exception:
            predicted_encoded = 0

        response = {
            "prediction": predicted_crop,
            "prediction_encoded": predicted_encoded,
            "top_3": top_3_crops,
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

    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Disease detection timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")



@app.post("/ai-query")
def ai_query(payload: dict):

    query = payload.get("query")

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Missing query"
        )

    try:
        model = get_gemini_model()

        prompt = f"""
                You are an agricultural expert.

                Answer the farmer's question in:
                1. Simple language
                2. Maximum 5 points
                3. No technical jargon
                4. Practical recommendations only

                Question:
                {query}
                """
        response = model.generate_content(prompt)

        return {
            "success": True,
            "query": query,
            "answer": response.text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
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
@app.post("/irrigation-recommendation")
def irrigation_recommendation(data: IrrigationInput):
    model_path    = "irrigation_model.pkl"
    encoders_path = "label_encoder_irrigation.pkl"
    target_path   = "target_encoder_irrigation.pkl"

    


    if not os.path.exists(model_path):
        return {
            "status": "model_not_ready",
            "message": "ML model not yet available.",
            "received_data": data.dict()
        }

    try:
        model          = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)  # dictionary
        target_encoder = joblib.load(target_path)

        # Encode categorical columns using dictionary
        crop_encoded = label_encoders["crop_type"].transform([data.crop_type])[0]
        soil_encoded = label_encoders["soil_type"].transform([data.soil_type])[0]
        region_encoded = label_encoders["region"].transform([data.region])[0]
        season_encoded = label_encoders["season"].transform([data.season])[0]
        groundwater_encoded = label_encoders["groundwater_availability"].transform([data.groundwater_availability])[0]

        # Build feature dataframe with exact column order from training
        features = pd.DataFrame([[
            crop_encoded,
            soil_encoded,
            region_encoded,
            season_encoded,
            groundwater_encoded,
            data.farm_size_acres,
            data.temperature_C,
            data.rainfall_mm,
            data.soil_moisture_percent,
            data.humidity_percent
            
        ]], columns=[
            "crop_type", "soil_type", "region", "season","groundwater_availability",
            "farm_size_acres", "temperature_C", "rainfall_mm",
            "soil_moisture_percent", "humidity_percent"
            
        ])

        prediction = model.predict(features)
        result = target_encoder.inverse_transform(prediction)

        return {
            "status": "success",
            "recommended_irrigation": result[0],
            "message": f"Recommended irrigation type: {result[0]}"
        }

    except KeyError as e:
        return {
            "status": "error",
            "message": f"Label encoder key mismatch: {str(e)}. Check exact column names with Member 3."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Irrigation prediction failed: {str(e)}")


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




    