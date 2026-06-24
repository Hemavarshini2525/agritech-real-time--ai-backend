import json
import urllib.request
import traceback

url = 'http://127.0.0.1:8000/predict'
payload = {
    "location": "Tirunelveli",
    "ph": 6.5,
    "phosphorus": 82,
    "potassium": 52,
    "nitrogen": 105
}

req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(resp.read().decode())
except Exception:
    traceback.print_exc()
