# tn_rainfall.py — Final version
# TN Government Rainfall Data 2024-25 — Monthly Actual Rainfall (mm)
# Source: India Meteorological Department, Chennai (tn.gov.in)

from datetime import datetime

TN_MONTHLY_RAINFALL = {
    "chennai": {"jun": 200.4, "jul": 144.6, "aug": 114.8, "sep": 181.9, "oct": 371.7, "nov": 347.9, "dec": 358.0, "jan": 39.6, "feb": 0.0, "mar": 16.4, "apr": 39.5, "may": 55.8},
    "kancheepuram": {"jun": 175.5, "jul": 149.8, "aug": 219.9, "sep": 93.7, "oct": 163.7, "nov": 122.3, "dec": 358.5, "jan": 40.4, "feb": 0.0, "mar": 9.8, "apr": 7.0, "may": 88.3},
    "chengalpattu": {"jun": 158.9, "jul": 119.6, "aug": 162.5, "sep": 107.5, "oct": 218.6, "nov": 214.9, "dec": 290.9, "jan": 15.7, "feb": 0.0, "mar": 12.5, "apr": 17.9, "may": 73.4},
    "tiruvallur": {"jun": 188.5, "jul": 171.7, "aug": 145.5, "sep": 150.0, "oct": 260.2, "nov": 203.6, "dec": 379.8, "jan": 47.9, "feb": 0.0, "mar": 12.5, "apr": 23.7, "may": 99.5},
    "cuddalore": {"jun": 123.8, "jul": 53.3, "aug": 184.0, "sep": 45.1, "oct": 204.5, "nov": 204.9, "dec": 371.3, "jan": 4.3, "feb": 5.0, "mar": 73.0, "apr": 26.5, "may": 158.7},
    "villupuram": {"jun": 88.7, "jul": 72.1, "aug": 319.3, "sep": 58.5, "oct": 162.4, "nov": 142.5, "dec": 572.5, "jan": 6.3, "feb": 0.0, "mar": 37.0, "apr": 19.3, "may": 144.1},
    "kallakurichi": {"jun": 84.8, "jul": 33.1, "aug": 168.9, "sep": 48.0, "oct": 171.9, "nov": 70.6, "dec": 402.6, "jan": 1.7, "feb": 0.0, "mar": 93.8, "apr": 29.5, "may": 185.7},
    "vellore": {"jun": 153.1, "jul": 104.3, "aug": 205.1, "sep": 64.9, "oct": 212.0, "nov": 60.0, "dec": 243.7, "jan": 28.1, "feb": 0.0, "mar": 1.1, "apr": 39.1, "may": 176.6},
    "ranipet": {"jun": 206.7, "jul": 154.3, "aug": 223.4, "sep": 88.4, "oct": 165.4, "nov": 73.5, "dec": 369.1, "jan": 25.8, "feb": 0.0, "mar": 6.3, "apr": 3.3, "may": 141.6},
    "tirupathur": {"jun": 183.7, "jul": 31.7, "aug": 190.0, "sep": 28.4, "oct": 281.6, "nov": 23.8, "dec": 188.7, "jan": 12.3, "feb": 0.0, "mar": 0.9, "apr": 40.5, "may": 161.7},
    "tiruvannamalai": {"jun": 135.8, "jul": 81.4, "aug": 289.4, "sep": 47.7, "oct": 188.7, "nov": 89.8, "dec": 389.3, "jan": 16.6, "feb": 0.0, "mar": 22.9, "apr": 26.8, "may": 182.5},
    "salem": {"jun": 127.8, "jul": 40.3, "aug": 242.8, "sep": 45.5, "oct": 230.4, "nov": 65.1, "dec": 244.5, "jan": 4.8, "feb": 0.0, "mar": 37.2, "apr": 38.0, "may": 169.0},
    "namakkal": {"jun": 93.3, "jul": 17.5, "aug": 142.7, "sep": 27.4, "oct": 179.4, "nov": 78.4, "dec": 149.0, "jan": 10.0, "feb": 0.0, "mar": 28.1, "apr": 47.2, "may": 155.7},
    "dharmapuri": {"jun": 106.4, "jul": 32.1, "aug": 184.7, "sep": 13.5, "oct": 218.8, "nov": 44.4, "dec": 244.9, "jan": 2.9, "feb": 0.0, "mar": 13.4, "apr": 50.2, "may": 228.4},
    "krishnagiri": {"jun": 139.4, "jul": 29.8, "aug": 190.8, "sep": 16.6, "oct": 261.3, "nov": 40.5, "dec": 216.9, "jan": 3.9, "feb": 0.0, "mar": 13.5, "apr": 58.0, "may": 176.2},
    "coimbatore": {"jun": 242.0, "jul": 459.2, "aug": 176.9, "sep": 90.3, "oct": 331.7, "nov": 103.0, "dec": 69.4, "jan": 5.7, "feb": 0.0, "mar": 41.8, "apr": 101.8, "may": 367.8},
    "tiruppur": {"jun": 42.2, "jul": 32.8, "aug": 106.3, "sep": 25.6, "oct": 265.2, "nov": 77.1, "dec": 99.3, "jan": 14.3, "feb": 0.0, "mar": 30.6, "apr": 72.4, "may": 80.9},
    "erode": {"jun": 67.3, "jul": 38.9, "aug": 129.2, "sep": 26.2, "oct": 199.8, "nov": 66.1, "dec": 62.3, "jan": 9.4, "feb": 0.0, "mar": 30.9, "apr": 90.5, "may": 86.9},
    "tiruchirapalli": {"jun": 99.5, "jul": 10.5, "aug": 98.0, "sep": 14.2, "oct": 215.9, "nov": 101.7, "dec": 223.6, "jan": 13.5, "feb": 0.0, "mar": 25.5, "apr": 29.4, "may": 120.8},
    "trichy": {"jun": 99.5, "jul": 10.5, "aug": 98.0, "sep": 14.2, "oct": 215.9, "nov": 101.7, "dec": 223.6, "jan": 13.5, "feb": 0.0, "mar": 25.5, "apr": 29.4, "may": 120.8},
    "karur": {"jun": 70.9, "jul": 0.4, "aug": 126.9, "sep": 13.7, "oct": 203.5, "nov": 48.6, "dec": 172.0, "jan": 8.5, "feb": 0.0, "mar": 21.5, "apr": 64.3, "may": 66.2},
    "perambalur": {"jun": 39.9, "jul": 17.4, "aug": 196.2, "sep": 9.2, "oct": 169.2, "nov": 81.5, "dec": 242.1, "jan": 11.5, "feb": 0.0, "mar": 22.6, "apr": 24.7, "may": 108.5},
    "ariyalur": {"jun": 58.3, "jul": 17.3, "aug": 144.0, "sep": 33.0, "oct": 163.0, "nov": 147.8, "dec": 306.3, "jan": 20.5, "feb": 0.0, "mar": 44.1, "apr": 56.6, "may": 147.0},
    "pudukkottai": {"jun": 116.7, "jul": 37.5, "aug": 160.6, "sep": 23.3, "oct": 228.3, "nov": 162.2, "dec": 166.0, "jan": 8.1, "feb": 5.0, "mar": 41.5, "apr": 38.7, "may": 62.9},
    "thanjavur": {"jun": 80.3, "jul": 25.8, "aug": 108.7, "sep": 29.2, "oct": 222.3, "nov": 248.9, "dec": 272.5, "jan": 28.9, "feb": 6.4, "mar": 65.7, "apr": 57.4, "may": 130.2},
    "tiruvarur": {"jun": 69.5, "jul": 15.4, "aug": 97.6, "sep": 27.9, "oct": 199.7, "nov": 402.7, "dec": 291.7, "jan": 31.8, "feb": 15.8, "mar": 126.7, "apr": 61.2, "may": 138.4},
    "nagapattinam": {"jun": 38.9, "jul": 9.3, "aug": 56.9, "sep": 17.9, "oct": 136.4, "nov": 767.5, "dec": 321.0, "jan": 72.7, "feb": 10.0, "mar": 101.9, "apr": 10.7, "may": 132.6},
    "mayiladuthurai": {"jun": 58.5, "jul": 31.5, "aug": 97.2, "sep": 17.9, "oct": 195.4, "nov": 422.4, "dec": 306.8, "jan": 79.6, "feb": 4.9, "mar": 66.3, "apr": 35.9, "may": 121.7},
    "madurai": {"jun": 80.6, "jul": 27.0, "aug": 185.6, "sep": 18.5, "oct": 320.8, "nov": 77.5, "dec": 137.4, "jan": 3.3, "feb": 3.8, "mar": 30.3, "apr": 80.1, "may": 74.0},
    "theni": {"jun": 105.5, "jul": 99.8, "aug": 132.3, "sep": 38.5, "oct": 191.3, "nov": 92.8, "dec": 120.1, "jan": 17.3, "feb": 1.6, "mar": 26.9, "apr": 120.2, "may": 140.9},
    "dindigul": {"jun": 70.6, "jul": 38.7, "aug": 176.7, "sep": 46.7, "oct": 305.6, "nov": 131.8, "dec": 161.1, "jan": 37.6, "feb": 0.3, "mar": 39.2, "apr": 79.9, "may": 87.1},
    "ramanathapuram": {"jun": 48.1, "jul": 17.5, "aug": 64.3, "sep": 19.9, "oct": 195.3, "nov": 327.4, "dec": 134.8, "jan": 66.4, "feb": 14.6, "mar": 85.8, "apr": 72.2, "may": 61.3},
    "virudhunagar": {"jun": 79.3, "jul": 18.2, "aug": 161.9, "sep": 29.7, "oct": 152.4, "nov": 84.9, "dec": 224.3, "jan": 20.3, "feb": 4.4, "mar": 41.3, "apr": 105.0, "may": 77.2},
    "sivagangai": {"jun": 105.4, "jul": 74.4, "aug": 199.0, "sep": 26.0, "oct": 314.7, "nov": 126.4, "dec": 176.3, "jan": 5.9, "feb": 7.6, "mar": 62.0, "apr": 61.6, "may": 87.0},
    "tirunelveli": {"jun": 118.4, "jul": 109.7, "aug": 60.8, "sep": 48.9, "oct": 115.2, "nov": 314.6, "dec": 432.4, "jan": 226.4, "feb": 19.6, "mar": 167.2, "apr": 100.8, "may": 200.3},
    "tenkasi": {"jun": 71.6, "jul": 99.6, "aug": 45.5, "sep": 56.2, "oct": 98.5, "nov": 131.9, "dec": 333.8, "jan": 31.7, "feb": 2.7, "mar": 113.2, "apr": 74.3, "may": 187.1},
    "thoothukudi": {"jun": 6.2, "jul": 1.3, "aug": 42.9, "sep": 35.0, "oct": 71.1, "nov": 147.6, "dec": 189.1, "jan": 26.5, "feb": 2.4, "mar": 76.5, "apr": 64.6, "may": 18.1},
    "nilgiris": {"jun": 336.2, "jul": 617.0, "aug": 216.9, "sep": 103.8, "oct": 227.7, "nov": 157.2, "dec": 111.2, "jan": 11.8, "feb": 0.1, "mar": 88.4, "apr": 82.5, "may": 430.1},
    "kanniyakumari": {"jun": 209.6, "jul": 116.3, "aug": 128.5, "sep": 47.9, "oct": 270.2, "nov": 242.7, "dec": 27.1, "jan": 5.8, "feb": 0.7, "mar": 65.7, "apr": 221.1, "may": 230.9},
}



MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]

# Dataset's rainfall column average (from your training data) — used to scale
# Your dataset rainfall range was ~185-300mm, average ~240mm
DATASET_RAINFALL_MEAN = 240.0


def get_seasonal_rainfall(location: str, months: int = 3) -> float:
    """
    Returns rainfall (mm) for current month + next (months-1) months,
    for the given Tamil Nadu district. Falls back to dataset mean if
    district is not found.
    """
    location_lower = location.lower().strip()
    monthly_data = None

    for district, data in TN_MONTHLY_RAINFALL.items():
        if district in location_lower or location_lower in district:
            monthly_data = data
            break

    if monthly_data is None:
        return DATASET_RAINFALL_MEAN

    current_month_idx = datetime.now().month - 1
    selected = []
    for i in range(months):
        month_idx = (current_month_idx + i) % 12
        selected.append(monthly_data[MONTH_ORDER[month_idx]])

    return round(sum(selected), 1)


if __name__ == "__main__":
    for d in ["Thanjavur", "Chennai", "Coimbatore", "Madurai", "Salem", "Trichy", "Unknown City"]:
        print(f"{d}: {get_seasonal_rainfall(d)} mm")
