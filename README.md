# AgroAI Backend — Real-Time AI Crop Advisory System

FastAPI backend powering AgroAI: crop, fertilizer, and irrigation
recommendations, plant disease detection, and a Gemini-powered farming
assistant — backed by live weather and soil data.

Built for the SASTRA TBI + Innovaisz Corp research internship
(Track 01: AI & Data Science).

---

## 🔗 Live Deployment

**Backend (Render):** https://agritech-real-time-ai-backend.onrender.com

Interactive API docs: `/docs` (Swagger UI)

> Render free-tier services spin down when idle — the first request
> after inactivity may take 30–60s to respond.

---

##  What This Backend Does

- **Crop Recommendation** — predicts the best crop using soil
  nutrients (N/P/K), pH, temperature, humidity, and seasonal rainfall.
- **Fertilizer Recommendation** — suggests fertilizer based on crop
  type, soil type, and NPK levels.
- **Irrigation Recommendation** — recommends an irrigation method
  based on crop, soil, region, season, and farm size.
- **Plant Disease Detection** — detects disease from a leaf photo and
  returns treatment advice.
- **AI Farming Assistant** — answers free-form farming questions in
  simple language (powered by Gemini).
- **History** — every prediction/query is saved to Firestore and
  retrievable by feature type.
- **Tamil Nadu District & Taluk Lookup** — location reference data
  used by the frontend's location inputs.

---

## 🏗️ Architecture

Frontend (separate repo) connects to this FastAPI backend (Render),
which connects to:
- Firebase Firestore (all history)
- Gemini API (`/ai-query`)
- OpenWeatherMap API (weather)
- SoilGrids API (soil)
- Hugging Face Spaces:
  - Disease Detection (MobileNetV3)
  - Irrigation Detection (ExtraTreesClassifier)

All history is persisted exclusively in **Firebase Firestore** — there
is no local/SQLite database in this service.

---

## 🧰 Tech Stack

- **Framework:** FastAPI (Python)
- **ML models:**
  - Crop → StackingClassifier (scikit-learn)
  - Fertilizer → Random Forest (scikit-learn)
  - Irrigation → ExtraTreesClassifier, served via Hugging Face Spaces
  - Disease → MobileNetV3, served via Hugging Face Spaces
- **Storage:** Firebase Admin SDK / Firestore
- **AI assistant:** Google Gemini API (`google-generativeai`)
- **External APIs:** OpenWeatherMap (weather), SoilGrids (soil)
- **Inference client:** `gradio_client` (for Hugging Face Spaces)
- **Deployment:** Render

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/weather?location=` | Current weather for a location |
| GET | `/soil?lat=&lon=` | Soil data for coordinates |
| POST | `/predict` | Crop recommendation |
| POST | `/predict-disease` | Plant disease detection (image upload) |
| POST | `/ai-query` | Ask the Gemini-powered farming assistant |
| POST | `/fertilizer-recommendation` | Fertilizer recommendation |
| POST | `/irrigation-recommendation` | Irrigation type recommendation |
| POST | `/save-advisory` | Save a combined advisory record |
| GET | `/history?feature_type=&limit=` | Fetch prediction/advisory history |
| GET | `/tn-districts` | List Tamil Nadu districts |
| GET | `/tn-taluks/{district}` | List taluks for a district |

---

## ⚙️ Local Setup

```bash
git clone https://github.com/<your-username>/agritech-real-time--ai-backend.git
cd agritech-real-time--ai-backend
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENWEATHER_API_KEY=33658c69e218d762161d43a43ea34a50
GEMINI_API_KEY=AQ.Ab8RN6LyDMShPaCNpGQ9zFeF8C37bMmx-kgsflCvwNydQr04uw
GEMINI_MODEL=gemini-2.5-flash
```


Run the server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`, with docs at
`http://localhost:8000/docs`.

---

## 🔒 Security Notes

- `firebase-key.json` contains a Firebase service-account private key
  and must never be committed to GitHub. Add it to `.gitignore` and
  provide it to Render via a secret file or environment variable
  instead.
- CORS is currently set to allow all origins (`allow_origins=["*"]`)
  for development convenience — restrict this to the frontend's
  actual domain before treating this as production-ready.

---

## 📁 Project Structure

- main.py — FastAPI app & all routes
- models.py — Pydantic request/response schemas
- weather.py — OpenWeatherMap integration
- soil.py — SoilGrids integration
- tn_rainfall.py — Seasonal rainfall lookup
- tn_taluks.py — Tamil Nadu district/taluk data
- crop_ensemble_model.pkl — Crop recommendation model
- crop_label_encoder.pkl — Crop label encoder
- crop_scaler.pkl — Feature scaler for crop model
- firebase-key.json (not committed) — Firebase service account
- requirements.txt

---

## 🗺️ Roadmap

- [ ] Move CORS to an explicit allow-list before production
- [ ] Add automated tests for prediction endpoints
- [ ] Expand disease advice database with more crop/disease pairs
