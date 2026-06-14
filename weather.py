import requests


def get_weather(location: str):
    """
    Fetch comprehensive weather data using Open-Meteo API (free, no key required).
    First geocodes the location to get coordinates, then fetches weather.
    """
    try:
        # Step 1: Geocode location to get latitude/longitude
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        geocode_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        geocode_response = requests.get(geocode_url, params=geocode_params)
        geocode_response.raise_for_status()
        geocode_data = geocode_response.json()
        
        if not geocode_data.get("results"):
            return {"error": f"Location '{location}' not found"}
        
        result = geocode_data["results"][0]
        latitude = result["latitude"]
        longitude = result["longitude"]
        location_name = f"{result.get('name', '')}, {result.get('country', '')}"
        
        # Step 2: Fetch weather data using Open-Meteo
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "precipitation,"
                "rain,"
                "showers,"
                "snowfall,"
                "weather_code,"
                "wind_speed_10m,"
                "wind_direction_10m,"
                "wind_gusts_10m,"
                "pressure_msl,"
                "surface_pressure,"
                "cloud_cover,"
                "visibility,"
                "dew_point_2m,"
                "soil_temperature_0cm,"
                "soil_temperature_6cm,"
                "soil_moisture_0_1cm,"
                "soil_moisture_1_3cm,"
                "soil_moisture_3_9cm"
            ),
            "timezone": "auto"
        }
        
        weather_response = requests.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        current = weather_data.get("current", {})
        
        return {
            "location": location_name,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": weather_data.get("timezone"),
            
            # Temperature data
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "dew_point": current.get("dew_point_2m"),
            
            # Humidity and pressure
            "humidity": current.get("relative_humidity_2m"),
            "pressure_msl": current.get("pressure_msl"),
            "surface_pressure": current.get("surface_pressure"),
            
            # Precipitation
            "precipitation": current.get("precipitation"),
            "rain": current.get("rain"),
            "showers": current.get("showers"),
            "snowfall": current.get("snowfall"),
            
            # Wind
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "wind_gusts": current.get("wind_gusts_10m"),
            
            # Cloud and visibility
            "cloud_cover": current.get("cloud_cover"),
            "visibility": current.get("visibility"),
            
            # Soil data
            "soil_temperature_0cm": current.get("soil_temperature_0cm"),
            "soil_temperature_6cm": current.get("soil_temperature_6cm"),
            "soil_moisture_0_1cm": current.get("soil_moisture_0_1cm"),
            "soil_moisture_1_3cm": current.get("soil_moisture_1_3cm"),
            "soil_moisture_3_9cm": current.get("soil_moisture_3_9cm"),
            
            "weather_code": current.get("weather_code")
        }
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Weather API error: {str(e)}"}
    except KeyError as e:
        return {"error": f"Error parsing weather data: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}