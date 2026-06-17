from pydantic import BaseModel
from typing import Optional

class AdvisoryRecord(BaseModel):
    farmer_query: str
    disease: str
    fertilizer: str
    irrigation: str
    weather_info: str
    advisory_text: str

class AdvisoryInput(BaseModel):
    farmer_query: str
    location: str
    disease: str = ""
    fertilizer: str = ""
    irrigation: str = ""
    advisory_text: str = ""

class PredictionInput(BaseModel):
    location: str
    phosphorus: float
    potassium: float
    nitrogen: Optional[float] = None
    ph: Optional[float] = None

class FertilizerInput(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    nitrogen: int
    potassium: int
    phosphorous: int
    soil_type: str
    crop_type: str


class IrrigationInput(BaseModel):
    location: str

    crop_type: str
    soil_type: str
    region: str
    season: str

    farm_size_acres: float

    soil_moisture_percent: float

    groundwater_availability: str

    temperature_C: Optional[float] = None
    rainfall_mm: Optional[float] = None
    humidity_percent: Optional[float] = None