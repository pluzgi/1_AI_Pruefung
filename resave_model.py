import os
from tensorflow.keras.models import load_model
from src.config import OUTPUT_DIR
from src.model import dice_score, dice_loss, iou_score  # ensure all are imported

# Register them manually for loading
custom_objects = {
    "dice_score": dice_score,
    "dice_loss": dice_loss,
    "iou_score": iou_score
}

# Alle .keras-Dateien im OUTPUT_DIR finden
keras_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".keras")]

if not keras_files:
    print("⚠️ Keine .keras-Dateien im Output-Verzeichnis gefunden.")
else:
    for keras_file in keras_files:
        input_path = os.path.join(OUTPUT_DIR, keras_file)
        h5_file = keras_file.replace(".keras", "_resaved.h5")
        output_path = os.path.join(OUTPUT_DIR, h5_file)

        print(f"🔄 Konvertiere: {keras_file} → {h5_file}")
        model = load_model(input_path, custom_objects=custom_objects)
        model.save(output_path)
        print(f"✅ Gespeichert unter: {output_path}")
