import os
import sys
import subprocess
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import pandas as pd
from datetime import datetime

# Lokales src-Modul zugänglich machen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from config import BASE_DIR, SUBSET_CSV, TRAIN_IMAGE_DIR, OUTPUT_DIR
from preprocessing import create_paired_subset
from data_loader import load_dataset
from model import build_unet, dice_loss, dice_score, iou_score


MODE = os.environ.get("MODE", "baseline").lower()
print(f"📌 Trainingsmodus: {MODE}")

def sort_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Unnamed: 0").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"CSV sortiert: {csv_path}")
    
# --- 1. Subset erstellen ---
subset_output_path = SUBSET_CSV
csv_source_path = os.path.join("data", "train.csv")

print("📦 Erstelle gepaarte Subset-Datei...")
# Hier wird immer ein neues Subset mit 1000 Paaren erstellt
create_paired_subset(csv_path=csv_source_path, output_path=subset_output_path, num_pairs=1000)
sort_csv(subset_output_path)

# --- 2. Masken automatisch erzeugen, falls nicht vorhanden ---
masks_dir = os.path.join(BASE_DIR, "masks")
existing_masks = [f for f in os.listdir(masks_dir) if f.endswith(".png")] if os.path.exists(masks_dir) else []

if len(existing_masks) == 0:
    print("🖼️ Keine Masken gefunden. Starte automatische Maskenerzeugung...")
    subprocess.run(["python", "src/mask_generator.py"], check=True)
else:
    print(f"✅ {len(existing_masks)} Masken gefunden – keine Generierung nötig.")

# --- 3. Daten laden ---
print("📥 Lade Trainingsdaten...")
X, y = load_dataset(csv_path=subset_output_path, image_dir=TRAIN_IMAGE_DIR, target_size=(256, 256))

# --- 4. Modell aufbauen ---
print(f"🧠 Erstelle U-Net Modell im Modus: {MODE}")

if MODE == "baseline":
    model = build_unet(input_shape=(256, 256, 3), use_dropout=False)
    optimizer = tf.keras.optimizers.Adam()

elif MODE == "dropout":
    model = build_unet(input_shape=(256, 256, 3), use_dropout=True, dropout_rate=0.3)
    optimizer = tf.keras.optimizers.Adam()

elif MODE == "lowlr":
    model = build_unet(input_shape=(256, 256, 3), use_dropout=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
elif MODE == "diceloss":
    model = build_unet(input_shape=(256, 256, 3), use_dropout=False)
    loss = dice_loss
    optimizer = tf.keras.optimizers.Adam()

else:
    raise ValueError(f"Unbekannter MODE: {MODE}")

if MODE != "diceloss":
    loss = "binary_crossentropy"

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy", dice_score, iou_score]
)


# --- 5. Training konfigurieren ---
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(OUTPUT_DIR, "logs", f"run_{MODE}_{timestamp}")
checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{MODE}.keras")
model_path = os.path.join(OUTPUT_DIR, f"unet_model_{MODE}.keras")

callbacks = [
    TensorBoard(log_dir=log_dir),
    ModelCheckpoint(filepath=checkpoint_path, save_best_only=True),
    EarlyStopping(patience=3, restore_best_weights=True)
]

print("🚀 Starte Training...")
model.fit(
    X, y,
    validation_split=0.2,
    epochs=5,
    batch_size=8,
    callbacks=callbacks
)

# --- 6. Modell speichern ---
model.save(model_path)
print(f"✅ Modell gespeichert unter: {model_path}")

# --- 7. Trainingsdaten speichern (für spätere Notebook-Nutzung) ---
import numpy as np

print("💾 Speichere Trainingsdaten für das Notebook...")

# Dateinamen je nach Modus
np.save(os.path.join(OUTPUT_DIR, f"train_images_{MODE}.npy"), X)
np.save(os.path.join(OUTPUT_DIR, f"train_masks_{MODE}.npy"), y)

df = pd.read_csv(SUBSET_CSV)
df.to_pickle(os.path.join(OUTPUT_DIR, f"train_df_{MODE}.pkl"))

print(f"✅ Trainingsdaten gespeichert unter /output mit Suffix _{MODE}")
