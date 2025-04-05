import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.config import OUTPUT_DIR
from src.model import dice_score, iou_score

# --- 1. Modell dynamisch laden ---
MODE = os.environ.get("MODE", "dropout")
model_path = os.path.join(OUTPUT_DIR, f"best_model_{MODE}.keras")

print("Looking for model at:", model_path)
assert os.path.exists(model_path), f"Model file not found at: {model_path}"

model = load_model(
    model_path,  
    custom_objects={
        "dice_score": dice_score,
        "iou_score": iou_score
    },
    compile=False
)

print(f"Modell im Modus '{MODE}' geladen: {model_path}")

# --- 2. App initialisieren ---
app = FastAPI(title="Segmentation API", version="1.0")

# --- 3. Statische Dateien (HTML, CSS, JS) bereitstellen ---
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/upload")
def serve_upload_page():
    return FileResponse("api/static/upload.html")

# --- 4. Bildverarbeitung ---
def preprocess_image(image_bytes, target_size=(256, 256)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # (1, H, W, 3)

def postprocess_mask(mask_array):
    mask = (mask_array[0] > 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask.squeeze(), mode="L")
    return mask_image

# --- 5. API-Endpunkt: Vorhersage ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    prediction = model.predict(image_tensor)
    mask_image = postprocess_mask(prediction)

    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
