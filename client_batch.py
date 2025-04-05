import os
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import pandas as pd

# --- 1. API-Endpunkt
url = "http://127.0.0.1:8000/predict"

# --- 2. Ordner mit Testbildern
image_folder = "input/test_data_v2"
output_folder = "output/masks_predicted"
os.makedirs(output_folder, exist_ok=True)

# --- 3. Vorbereitung für Logging
log = []

# --- 4. Dateitypen zulassen
valid_ext = [".jpg", ".png", ".jpeg"]
image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in valid_ext]

for i, filename in enumerate(image_files):
    image_path = os.path.join(image_folder, filename)
    print(f"[{i+1}/{len(image_files)}] Sende: {filename}")

    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        # Maske anzeigen
        mask_img = Image.open(BytesIO(response.content))
        orig_img = Image.open(image_path).convert("RGB").resize(mask_img.size)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title("Original")
        axes[1].imshow(mask_img, cmap="gray")
        axes[1].set_title("Vorhersage")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        # Maske speichern
        save_path = os.path.join(output_folder, f"mask_{filename}")
        mask_img.save(save_path)
        print(f"✅ Maske gespeichert: {save_path}\n")

        # Logging-Eintrag
        log.append({
            "original_image": image_path,
            "predicted_mask": save_path
        })

    else:
        print(f"❌ Fehler bei {filename}: {response.status_code}\n")

# --- 5. Logging speichern ---
df = pd.DataFrame(log)
log_path = "output/prediction_log.csv"
df.to_csv(log_path, index=False)
print(f"📝 Log gespeichert unter: {log_path}")
