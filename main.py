"""
AgriTech Backend API
=====================

FastAPI backend for the AgroAI Real-Time AI Crop Advisory System.

Responsibilities:
    - Crop recommendation (StackingClassifier)
    - Fertilizer recommendation (Random Forest)
    - Irrigation recommendation (ExtraTreesClassifier, served via Hugging Face Space)
    - Plant disease detection (MobileNetV3, served via Hugging Face Space)
    - Gemini-powered farmer advisory Q&A
    - Weather + soil data lookups (external APIs)
    - Tamil Nadu district/taluk lookups
    - Persistent history of all predictions, stored in Firestore

Persistence:
    All history (crop / disease / fertilizer / irrigation / advisory) is stored
    exclusively in Firebase Firestore under the "history" collection. There is
    no local/SQLite storage in this service.
"""

import os
import joblib
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weather import get_weather
from soil import get_soil_data
from models import AdvisoryInput, PredictionInput, FertilizerInput, IrrigationInput
from tn_rainfall import get_seasonal_rainfall
from tn_taluks import district_taluks, get_taluks
import firebase_admin
from firebase_admin import credentials, firestore

import numpy as np
from fastapi import UploadFile, File
import pandas as pd
import httpx
import google.generativeai as genai

app = FastAPI(title="AgriTech Backend API")



#  LOGGING 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Path to the trained crop-recommendation ensemble model (StackingClassifier).
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_ensemble_model.pkl")

# Gemini model name used for the /ai-query advisory endpoint.
# Override via GEMINI_MODEL env var if needed (e.g. when a model is
# deprecated or a newer/cheaper one should be used instead).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5")


def get_gemini_model():
    """
    Build a configured Gemini GenerativeModel client for the advisory endpoint.

    Raises:
        RuntimeError: if GEMINI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(GEMINI_MODEL)


# Paths to the crop model's label encoder.
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "crop_label_encoder.pkl")

# Lazily-loaded singletons for the crop model + encoder (populated on first use).
_loaded_model = None
_loaded_encoder = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins; adjust for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  FIREBASE INIT 
# Firestore is the single source of truth for all prediction history.
# firebase-key.json must be present alongside this file (kept out of
# version control â€” see .gitignore / Render secret files).
cred = credentials.Certificate(
    os.path.join(os.path.dirname(__file__), "firebase-key.json")
)
firebase_admin.initialize_app(cred)
db = firestore.client()




#  MODEL LOADERS 

def load_model():
    """
    Lazily load and cache the crop-recommendation StackingClassifier.

    Raises:
        FileNotFoundError: if the .pkl model file is missing on disk.
    """
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    _loaded_model = joblib.load(MODEL_PATH)
    return _loaded_model

def load_encoder():
    """
    Lazily load and cache the label encoder used to decode crop predictions
    back into human-readable crop names.

    Raises:
        FileNotFoundError: if the .pkl encoder file is missing on disk.
    """
    global _loaded_encoder
    if _loaded_encoder is not None:
        return _loaded_encoder
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder file not found: {ENCODER_PATH}")
    _loaded_encoder = joblib.load(ENCODER_PATH)
    return _loaded_encoder


def save_to_history(feature_type: str, input_data: dict, result_data: dict):
    """
    Persist a single prediction/advisory event to Firestore.

    Args:
        feature_type: one of "crop", "disease", "fertilizer", "irrigation", "advisory".
        input_data: the request payload that produced this result.
        result_data: the response payload returned to the client.
    """
    db.collection("history").add({
        "feature_type": feature_type,
        "input_data": input_data,
        "result_data": result_data,
        "timestamp": firestore.SERVER_TIMESTAMP
    })



#  ROUTES 

@app.get("/")
def root():
    """Health check / liveness endpoint."""
    return {"message": "AgriTech Backend is running …"}


@app.get("/weather")
def weather(location: str):
    """Return current weather data for a given location name."""
    result = get_weather(location)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/soil")
def soil(lat: float, lon: float):
    """Return SoilGrids soil data for given coordinates."""
    result = get_soil_data(lat, lon)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/predict")
def predict(input: PredictionInput):
    """
    Recommend a crop based on soil nutrients (N/P/K), pH, weather, and rainfall.

    Missing nitrogen/pH values fall back to SoilGrids data, then to hardcoded
    defaults (N=50.0, pH=6.5) if SoilGrids has no data for the coordinates
    (common for urban/coastal Tamil Nadu locations).

    Returns the top-1 prediction plus a top-3 list (when the model supports
    predict_proba), and saves the result to Firestore history.
    """
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

        X_scaled = scaler.transform(X)          # â† scale cheyyuka

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

        
        save_to_history("crop", input.dict(), response) 
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")





@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    """
    Detect plant disease from an uploaded leaf image.

    Inference is delegated to the MobileNetV3 model hosted on the
    Hugging Face Space (Hemavarshini2525/agritech-disease-detection)
    via gradio_client. The uploaded image is written to a temp file
    because the gradio_client API expects a file path, not raw bytes.

    Saves the result to Firestore history under feature_type="disease".
    """
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await file.read()

    try:
        import tempfile
        import os
        from gradio_client import Client, handle_file

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        client = Client("Hemavarshini2525/agritech-disease-detection")
        raw_result = client.predict(
            image=handle_file(tmp_path),
            api_name="/predict_disease"
        )

        os.unlink(tmp_path)

        confidence = None
        result = raw_result

        if isinstance(raw_result, dict):
            confidence = raw_result.get("confidence") or raw_result.get("probability") or raw_result.get("score") or raw_result.get("prob")
            result = raw_result.get("label") or raw_result.get("prediction") or next(iter(raw_result.values()), "")
        elif isinstance(raw_result, (list, tuple)) and len(raw_result) > 0:
            if len(raw_result) > 1 and isinstance(raw_result[1], (int, float)):
                confidence = raw_result[1]
            result = raw_result[0]

        if isinstance(result, str):
            result = result.strip()
            if result.lower().startswith("disease:"):
                result = result.split(":", 1)[1].strip()
            result = result.splitlines()[0].strip()

        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None

        if isinstance(confidence, (int, float)):
            try:
                confidence = float(confidence)
            except Exception:
                confidence = None

        advice = DISEASE_ADVICE.get(result, "Consult a local agricultural expert for specific treatment advice.")

        response = {
            "disease_prediction": result,
            "advice": advice,
            "confidence_score": confidence if confidence is not None else 0.0,
            "status": "success"
        }

        save_to_history("disease", {"filename": file.filename}, response) 

        return response

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Disease detection timed out")
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="Disease prediction requires the gradio_client package. Install dependencies with `pip install -r requirements.txt` and restart the service."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")

@app.post("/ai-query")
def ai_query(payload: dict):
    """
    Answer a free-form farmer question using Gemini, constrained to a
    simple, jargon-free, max-5-point response format.

    Expects payload = {"query": "<question text>"}.
    Saves the result to Firestore history under feature_type="advisory".
    """
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

        response2 =  {
            "success": True,
            "query": query,
            "answer": response.text
        }

        save_to_history("advisory", {"query": query}, response2)

        return response2
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
@app.post("/fertilizer-recommendation")
def fertilizer_recommendation(data: FertilizerInput):
    """
    Recommend a fertilizer based on crop/soil type, NPK values, and
    environmental conditions, using a Random Forest model.

    Returns status="model_not_ready" with the received payload echoed
    back if the model file isn't present on disk yet.
    Saves the result to Firestore history under feature_type="fertilizer".
    """
    try:
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


        response =  {
            "status": "success",
            "recommended_fertilizer": result[0],
            "message": f"Apply {result[0]} based on your soil and crop data"
        }

        save_to_history("fertilizer", data.dict(), response) 
        return response
    except Exception as e:
        logger.exception("Fertilizer recommendation failed")
        raise HTTPException(status_code=500, detail=f"Fertilizer error: {str(e)}")


@app.post("/irrigation-recommendation")
def irrigation_recommendation(data: IrrigationInput):
    """
    Recommend an irrigation type based on crop/soil/region/season and
    environmental readings, using an ExtraTreesClassifier model hosted
    on the Hugging Face Space (Hemavarshini2525/agritech-irrigation-detection).

    Saves the result to Firestore history under feature_type="irrigation".
    """
    try:
        from gradio_client import Client

        client = Client("Hemavarshini2525/agritech-irrigation-detection")
        result = client.predict(
            crop_type=data.crop_type,
            soil_type=data.soil_type,
            region=data.region,
            season=data.season,
            farm_size_acres=data.farm_size_acres,
            temperature_C=data.temperature_C,
            rainfall_mm=data.rainfall_mm,
            soil_moisture_percent=data.soil_moisture_percent,
            humidity_percent=data.humidity_percent,
            groundwater_availability=data.groundwater_availability,
            api_name="/predict_irrigation"
        )

        response = {
            "status": "success",
            "recommended_irrigation": result,
            "message": f"Recommended irrigation type: {result}"
        }

        save_to_history("irrigation", data.dict(), response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Irrigation prediction failed: {str(e)}")
@app.post("/save-advisory")
def save(data: AdvisoryInput):
    """
    Save a combined advisory record (farmer query + disease/fertilizer/
    irrigation results + weather context) as a single document.

    Previously persisted to a local SQLite database; now stored in
    Firestore under feature_type="advisory_record" so it's visible
    via /history alongside everything else.
    """
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
    save_to_history("advisory_record", data.dict(), record)
    return {"message": "Advisory saved successfully âœ…"}


@app.get("/history")
def history(feature_type: str = None, limit: int = 50):
    """
    Fetch prediction/advisory history from Firestore, newest first.

    Args:
        feature_type: optional filter â€” "crop", "disease", "fertilizer",
            "irrigation", "advisory", or "advisory_record". Omit to get all.
        limit: max number of records to return (default 50).
    """
    query = db.collection("history")

    if feature_type:
        query = query.where("feature_type", "==", feature_type)

    query = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)

    docs = query.stream()

    results = []
    for doc in docs:
        record = doc.to_dict()
        record["id"] = doc.id
        results.append(record)

    return results


@app.get("/tn-districts")
def get_tn_districts():
    """Return the list of all Tamil Nadu district names."""
    return {"districts": list(district_taluks.keys())}

@app.get("/tn-taluks/{district}")
def get_taluks_for_district(district: str):
    """Return the list of taluks for a given Tamil Nadu district."""
    taluks = get_taluks(district)
    if not taluks:
        raise HTTPException(status_code=404, detail=f"No taluks found for district: {district}")
    return {"district": district, "taluks": taluks}



