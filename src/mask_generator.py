import os
import numpy as np
from PIL import Image
import pandas as pd
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

from config import BASE_DIR, SUBSET_CSV, TRAIN_IMAGE_DIR
MASK_DIR = os.path.join(BASE_DIR, "masks")
IMAGE_SIZE = (256, 256)

# --- Hilfsfunktionen ---
def load_and_resize(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img).astype("float32") / 255.0

def generate_difference_mask(real_img, ai_img, threshold=0.15):
    real_gray = rgb2gray(real_img)
    ai_gray = rgb2gray(ai_img)
    _, diff = ssim(real_gray, ai_gray, full=True, data_range=1.0)
    diff = 1 - diff
    return (diff > threshold).astype(np.uint8)

def save_mask(mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img.save(save_path)

def find_pairs_from_csv(csv_path, base_image_dir):
    import os
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"Spalten im CSV: {df.columns.tolist()}")
    df = df.sort_values("Unnamed: 0").reset_index(drop=True)

    pairs = []
    for i in range(0, len(df) - 1, 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        print(f"Verarbeite Zeilen {i} und {i+1}: labels {row1['label']}, {row2['label']}")
        if row1["label"] == 1 and row2["label"] == 0:
            # row1: AI, row2: real
            real_path = os.path.join(base_image_dir, os.path.basename(row2["file_name"]))
            ai_path = os.path.join(base_image_dir, os.path.basename(row1["file_name"]))
            base_id = os.path.splitext(os.path.basename(row2["file_name"]))[0]
            pairs.append((real_path, ai_path, base_id))
            print(f"Pair gefunden: Real: {row2['file_name']} | AI: {row1['file_name']}")
        elif row1["label"] == 0 and row2["label"] == 1:
            # row1: real, row2: AI
            real_path = os.path.join(base_image_dir, os.path.basename(row1["file_name"]))
            ai_path = os.path.join(base_image_dir, os.path.basename(row2["file_name"]))
            base_id = os.path.splitext(os.path.basename(row1["file_name"]))[0]
            pairs.append((real_path, ai_path, base_id))
            print(f"Pair gefunden: Real: {row1['file_name']} | AI: {row2['file_name']}")
        else:
            print(f"Ungültiges Paar an Index {i} und {i+1}: {row1['label']}, {row2['label']}")
    print(f"Insgesamt gefundene Paare: {len(pairs)}")
    return pairs

# --- Hauptfunktion ---
def generate_all_masks():
    print(f"📂 Lade Bildpaare aus: {SUBSET_CSV}")
    pairs = find_pairs_from_csv(SUBSET_CSV, TRAIN_IMAGE_DIR)
    print(f"🔍 Gefundene Paare: {len(pairs)}")

    for real_path, ai_path, base_id in pairs:
        real = load_and_resize(real_path)
        ai = load_and_resize(ai_path)
        mask = generate_difference_mask(real, ai)
        save_path = os.path.join(MASK_DIR, f"{base_id}_mask.png")
        save_mask(mask, save_path)

    print(f"✅ Alle Masken gespeichert unter: {MASK_DIR}")

# --- Nur beim direkten Ausführen ---
if __name__ == "__main__":
    generate_all_masks()
