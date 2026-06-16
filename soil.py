import requests

def get_soil_data(lat, lon):
    try:
        # SoilGrids for pH and nitrogen only
        sg_url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query"
            f"?lon={lon}&lat={lat}"
            f"&property=nitrogen&property=phh2o"
            f"&depth=0-5cm&value=mean"
        )
        response = requests.get(sg_url, timeout=10)
        nitrogen = 50.0
        ph = 6.5

        if response.status_code == 200:
            data = response.json()
            layers = data.get("properties", {}).get("layers", [])
            for layer in layers:
                name = layer["name"]
                value = layer["depths"][0]["values"]["mean"]
                if name == "nitrogen" and value:
                    nitrogen = value
                elif name == "phh2o" and value:
                    ph = round(value / 10, 1)

        return {
            "nitrogen": nitrogen,
            "ph": ph,
            "note": "Estimated values from SoilGrids"
        }

    except Exception as e:
        return {
            "nitrogen": 50.0,
            "ph": 6.5,
            "error": str(e)
        }