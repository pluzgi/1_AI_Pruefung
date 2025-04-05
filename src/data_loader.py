import os
import pandas as pd
import numpy as np
from PIL import Image

from src.config import SUBSET_CSV, TRAIN_IMAGE_DIR, BASE_DIR

def load_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalisierung
    return image_array

def load_mask(mask_path, target_size=(256, 256)):
    mask = Image.open(mask_path).convert("L")  # Graustufen
    mask = mask.resize(target_size)
    mask_array = np.array(mask) / 255.0
    return (mask_array > 0.5).astype(np.float32)[..., np.newaxis]

# force_dummy=False für Baseline Training, force_dummy=True für echtes Training
def load_dataset(csv_path=SUBSET_CSV, image_dir=TRAIN_IMAGE_DIR, target_size=(256, 256), force_dummy=False, unlabeled=False):
    df = pd.read_csv(csv_path)
    X, y = [], []

    mask_dir = os.path.join(BASE_DIR, "masks")

    for _, row in df.iterrows():
        file_name = os.path.basename(row["file_name"])
        image_path = os.path.join(image_dir, file_name)

        try:
            image = load_image(image_path, target_size=target_size)
            X.append(image)

            if not unlabeled:
                label = row.get("label", 0)  # fallback für unlabeled CSVs
                mask_file = file_name.replace(".jpg", "_mask.png")
                mask_path = os.path.join(mask_dir, mask_file)

                if force_dummy or not os.path.exists(mask_path):
                    mask = np.full((*target_size, 1), float(label), dtype=np.float32)
                else:
                    mask = load_mask(mask_path, target_size=target_size)

                y.append(mask)

        except Exception as e:
            print(f"❌ Fehler beim Laden von {file_name}: {e}")

    X = np.array(X, dtype=np.float32)

    if unlabeled:
        print(f"📦 Geladen: {len(X)} unlabeled Bilder mit Shape {X.shape}")
        return X, None

    y = np.array(y, dtype=np.float32)
    num_total = len(y)
    num_dummy = sum((mask == 0).all() or (mask == 1).all() for mask in y)
    num_real = num_total - num_dummy

    print(f"📦 Geladen: {num_total} Bilder mit Shape {X.shape}")
    print(f"✅ Davon echte Segmentierungsmasken: {num_real}")
    print(f"⚠️  Dummy-Masken: {num_dummy}")
    return X, y