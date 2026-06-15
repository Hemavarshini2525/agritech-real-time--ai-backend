from pydantic import BaseModel

class AdvisoryRecord(BaseModel):
    farmer_query: str
    disease: str
    fertilizer: str
    irrigation: str
    weather_info: str
    advisory_text: str