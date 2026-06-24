import os
import joblib
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weather import get_weather
from database import init_db, save_advisory, get_all_history
from soil import get_soil_data
from models import AdvisoryRecord, AdvisoryInput, PredictionInput, FertilizerInput, IrrigationInput
from tn_rainfall import get_seasonal_rainfall
from tn_taluks import district_taluks, get_taluks
import firebase_admin
from firebase_admin import credentials, firestore

import io
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
import pandas as pd
import httpx
import google.generativeai as genai

app = FastAPI(title="AgriTech Backend API")


DISEASE_ADVICE = {
    "Pepper__bell___Bacterial_spot": "Apply copper-based bactericide. Avoid overhead irrigation, use drip instead. Remove infected leaves immediately.",
    "Pepper__bell___healthy": "Crop is healthy! Continue regular care and monitoring.",
    "Potato___Early_blight": "Apply fungicide containing chlorothalonil. Improve air circulation between plants. Avoid wetting foliage.",
    "Potato___Late_blight": "Apply copper-based fungicide immediately. Remove and destroy infected plants. Avoid overhead watering.",
    "Potato___healthy": "Crop is healthy! Continue regular care and monitoring.",
    "Tomato_Bacterial_spot": "Apply copper-based bactericide. Use disease-free seeds. Avoid working with wet plants.",
    "Tomato_Early_blight": "Apply fungicide with chlorothalonil or copper. Remove lower infected leaves. Mulch to prevent soil splash.",
    "Tomato_Late_blight": "Apply fungicide immediately, this spreads fast. Remove infected plants completely. Improve drainage.",
    "Tomato_Leaf_Mold": "Improve ventilation in greenhouse/field. Reduce humidity. Apply fungicide if severe.",
    "Tomato_Septoria_leaf_spot": "Remove infected lower leaves. Apply fungicide. Avoid overhead watering, water at soil level.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Apply miticide or neem oil. Increase humidity around plants. Remove heavily infested leaves.",
    "Tomato__Target_Spot": "Apply fungicide. Improve air circulation. Remove plant debris after harvest.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Remove infected plants to prevent spread. Control whitefly population (vector). Use resistant varieties next season.",
    "Tomato__Tomato_mosaic_virus": "Remove and destroy infected plants. Disinfect tools between plants. Control aphids (vector).",
    "Tomato_healthy": "Crop is healthy! Continue regular care and monitoring."
}

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

_loaded_model = None
_loaded_encoder = None
_disease_model = None
_disease_encoder = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins; adjust for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ─── FIREBASE INIT ──────────────────────────────────────────
cred = credentials.Certificate(
    os.path.join(os.path.dirname(__file__), "firebase-key.json")
)
firebase_admin.initialize_app(cred)
db = firestore.client()




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

def save_to_history(feature_type: str, input_data: dict, result_data: dict):
    db.collection("history").add({
        "feature_type": feature_type,
        "input_data": input_data,
        "result_data": result_data,
        "timestamp": firestore.SERVER_TIMESTAMP
    })



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

        
        save_to_history("crop", input.dict(), response) 
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

        save_to_history("advisory", {"query": query}, response)

        return response2
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
@app.post("/fertilizer-recommendation")
def fertilizer_recommendation(data: FertilizerInput):
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

        return {
            "status": "success",
            "recommended_irrigation": result,
            "message": f"Recommended irrigation type: {result}"
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
def history(feature_type: str = None, limit: int = 50):
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
    return {"districts": list(district_taluks.keys())}

@app.get("/tn-taluks/{district}")
def get_taluks_for_district(district: str):
    taluks = get_taluks(district)
    if not taluks:
        raise HTTPException(status_code=404, detail=f"No taluks found for district: {district}")
    return {"district": district, "taluks": taluks}

if __name__ == "__main__":
    print(DISEASE_ADVICE.get("Pepper__bell___Bacterial_spot"))