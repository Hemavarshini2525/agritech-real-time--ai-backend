import requests

def get_soil_data(lat, lon):
    try:
        # Open-Meteo soil data — free, no key, works globally
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "soil_temperature_0cm",
                "soil_moisture_0_to_1cm"
            ],
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})

        # SoilGrids for pH and nitrogen
        sg_url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query"
            f"?lon={lon}&lat={lat}"
            f"&property=nitrogen&property=phh2o"
            f"&depth=0-5cm&value=mean"
        )
        sg_response = requests.get(sg_url, timeout=10)
        nitrogen = 50.0
        ph = 6.5

        if sg_response.status_code == 200:
            sg_data = sg_response.json()
            layers = sg_data.get("properties", {}).get("layers", [])
            for layer in layers:
                name = layer["name"]
                value = layer["depths"][0]["values"]["mean"]
                if name == "nitrogen" and value:
                    nitrogen = value
                elif name == "phh2o" and value:
                    ph = round(value / 10, 1)

        return {
            "soil_temperature_0cm": current.get("soil_temperature_0cm"),
            "soil_moisture": current.get("soil_moisture_0_to_1cm"),
            "nitrogen": nitrogen,
            "ph": ph,
            "note": "pH and N are estimated values"
        }

    except Exception as e:
        return {
            "nitrogen": 50.0,
            "ph": 6.5,
            "error": str(e)
        }