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

## ✅ Features Completed

| Feature | Status | Notes |
|---|---|---|
| Crop Recommendation | ✅ Done | StackingClassifier; uses live weather + soil + seasonal rainfall, with fallback defaults when soil data is missing |
| Fertilizer Recommendation | ✅ Done | Random Forest model; returns `model_not_ready` gracefully if model file isn't deployed |
| Irrigation Recommendation | ✅ Done | ExtraTreesClassifier served via Hugging Face Space; saved to history |
| Plant Disease Detection | ✅ Done | MobileNetV3 served via Hugging Face Space; returns treatment advice + confidence score |
| AI Farming Assistant (Gemini) | ✅ Done | Free-form Q&A, constrained to simple, max-5-point answers |
| Prediction/Advisory History | ✅ Done | All five feature types saved to Firestore, filterable by `feature_type` |
| Weather Lookup | ✅ Done | OpenWeatherMap integration |
| Soil Data Lookup | ✅ Done | SoilGrids integration (gaps for some urban/coastal TN coordinates — see Known Limitations) |
| Tamil Nadu District/Taluk Reference | ✅ Done | Used by frontend location inputs |
| Combined Advisory Save | ✅ Done | `/save-advisory` now writes to Firestore (migrated off SQLite) |

---

## 🏗️ Architecture

```
                     ┌───────────────────────┐
   Frontend  ───────▶│   FastAPI Backend      │
   (separate repo)◀───│   (this repo, Render)  │
                     └───────────┬───────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       Firebase Firestore   Gemini API          OpenWeatherMap /
       (all history)        (/ai-query)         SoilGrids APIs
                                  │
                                  ▼
                       Hugging Face Spaces
                       ├── Disease Detection (MobileNetV3)
                       └── Irrigation Detection (ExtraTreesClassifier)
```

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
git clone https://github.com/Hemavarshini2525/agritech-real-time--ai-backend.git
cd agritech-real-time--ai-backend
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5        # optional, defaults to gemini-1.5
```

Placed our Firebase service account key as `firebase-key.json` in the
project root (**do not commit this file** — see [Security
Notes](#-security-notes)).

Run the server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`, with docs at
`http://localhost:8000/docs`.

---

## 🔒 Security Notes

- `firebase-key.json` contains a Firebase service-account private key
  and **must never be committed to GitHub**. Add it to `.gitignore`
  and provide it to Render via a secret file or environment variable
  instead.
- CORS is currently set to allow all origins (`allow_origins=["*"]`)
  for development convenience — restrict this to the frontend's
  actual domain before treating this as production-ready.

---

## ⚠️ Known Limitations

- **No per-user history isolation.** All predictions/queries from all
  users are written to a single shared Firestore `history` collection
  with no `user_id` field. Anyone using the app currently sees
  everyone's history mixed together. Planned fix: see Future
  Enhancements below.
- **Soil data gaps.** SoilGrids returns null values for some
  urban/coastal Tamil Nadu coordinates; the backend falls back to
  default N (50.0) and pH (6.5) values in these cases, flagged via a
  `soil_error` field in the `/predict` response.
- **External model dependency.** Disease and irrigation predictions
  depend on Hugging Face Spaces staying awake/available; both Spaces
  spin down when idle, so the first call after inactivity can be slow
  or time out (handled via `httpx.TimeoutException`).
- **Gemini free-tier quota.** `/ai-query` can return a 500 error if
  the configured Gemini model's free-tier quota is exhausted.

---

## 📁 Project Structure

```
.
main.py                     — FastAPI app & all routes
models.py                    — Pydantic request/response schemas
weather.py                    — OpenWeatherMap integration
soil.py                         — SoilGrids integration
tn_rainfall.py                   — Seasonal rainfall lookup
tn_taluks.py                       — Tamil Nadu district/taluk data
crop_ensemble_model.pkl             — Crop recommendation model
crop_label_encoder.pkl               — Crop label encoder
crop_scaler.pkl                        — Feature scaler for crop model
firebase-key.json (not committed)       — Firebase service account
requirements.txt
```

---

## 🚀 Future Enhancements

- [ ] **Per-user history.** Add Firebase Anonymous Auth (or full
  sign-in) to generate a stable `uid` per device, attach it to every
  Firestore history write, and filter `/history` by the requesting
  user.
- [ ] **CORS allow-list.** Replace the development-only wildcard with
  the frontend's actual deployed domain.
- [ ] **Automated tests.** Add unit/integration tests for all
  prediction endpoints, including model-not-ready and external-API
  failure paths.
- [ ] **Expanded disease advice.** Grow `DISEASE_ADVICE` to cover more
  crop/disease pairs as the detection model's class list grows.
- [ ] **Caching for weather/soil lookups.** Avoid redundant external
  API calls for repeated locations within a short time window.
- [ ] **Health checks for external dependencies.** Surface Hugging
  Face Space / Gemini availability in the `/` health check rather
  than only failing at request time.
- [ ] **Rate limiting.** Protect Gemini and external model calls from
  abuse given they consume paid/free-tier quota.