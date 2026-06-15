import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(location: str):
    if not WEATHER_API_KEY:
        return {"error": "OpenWeatherMap API key not found"}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "location": data["name"],
            "latitude": data["coord"]["lat"],
            "longitude": data["coord"]["lon"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure_msl": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"].get("deg"),
            "cloud_cover": data["clouds"]["all"],
            "visibility": data.get("visibility"),
            "weather_code": data["weather"][0]["id"],
            "rain": data.get("rain", {}).get("1h", 0.0),
            "precipitation": data.get("rain", {}).get("1h", 0.0),
            # Soil data from Open-Meteo separately
            "soil_temperature_0cm": None,
            "soil_moisture_0_1cm": None,
        }

    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return {"error": "Invalid API key"}
        elif response.status_code == 404:
            return {"error": f"Location '{location}' not found"}
        return {"error": f"Weather API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}