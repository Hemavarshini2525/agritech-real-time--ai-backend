# soil.py

import requests


def _query_isric_v2(lat, lon, depth="15-30cm"):
    url = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lon={lon}"
        f"&lat={lat}"
        f"&property=nitrogen"
        f"&property=phh2o"
        f"&depth={depth}"
        f"&value=mean"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    layers = data.get("properties", {}).get("layers", [])
    soil = {}
    for layer in layers:
        name = layer.get("name")
        depths = layer.get("depths") or []
        if depths:
            vals = depths[0].get("values", {})
            soil[name] = vals.get("mean")

    return {
        "nitrogen": soil.get("nitrogen"),
        "ph": (soil.get("phh2o") / 10 if soil.get("phh2o") is not None else None),
    }


def _query_soilgrids_legacy(lat, lon):
    # Legacy SoilGrids endpoint sometimes returns values when v2 returns null.
    url = f"https://rest.soilgrids.org/query?lat={lat}&lon={lon}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    props = data.get("properties", {})
    layers = props.get("layers", [])
    soil = {}
    for layer in layers:
        name = layer.get("name")
        depths = layer.get("depths") or []
        if depths:
            vals = depths[0].get("values", {})
            soil[name] = vals.get("mean")

    return {
        "nitrogen": soil.get("nitrogen"),
        "ph": (soil.get("phh2o") / 10 if soil.get("phh2o") is not None else None),
    }


def get_soil_data(lat, lon, depth="15-30cm"):
    """Get soil nitrogen and pH for a lat/lon.

    Tries ISRIC SoilGrids v2 first, then the legacy SoilGrids endpoint.
    Returns dict with keys: `nitrogen` (mg/kg or dataset units), `ph` (pH), and
    `error` if no provider returned data.
    """
    errors = []

    try:
        result = _query_isric_v2(lat, lon, depth=depth)
        if result.get("nitrogen") is not None or result.get("ph") is not None:
            return result
        errors.append("v2 returned no data")
    except Exception as e:
        errors.append(f"v2 error: {e}")

    try:
        result = _query_soilgrids_legacy(lat, lon)
        if result.get("nitrogen") is not None or result.get("ph") is not None:
            return result
        errors.append("legacy returned no data")
    except Exception as e:
        errors.append(f"legacy error: {e}")

    return {"nitrogen": None, "ph": None, "error": "; ".join(errors)}