#python client.py

import requests

# --- 1. API-Endpunkt
url = "http://127.0.0.1:8000/predict"

# --- 2. Pfad zum Bild (lokal)
image_path = "input/test_data_v2/sample.jpg"  # <--- Anpassen

# --- 3. Anfrage senden
with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# --- 4. Maske speichern
if response.status_code == 200:
    with open("predicted_mask.png", "wb") as f:
        f.write(response.content)
    print("✅ Maske gespeichert als predicted_mask.png")
else:
    print(f"❌ Fehler: {response.status_code}")
