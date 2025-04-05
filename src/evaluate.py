import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.config import SUBSET_CSV, TRAIN_IMAGE_DIR, OUTPUT_DIR
from src.data_loader import load_dataset
from src.model import dice_score, iou_score, dice_loss


def evaluate_model(model_path=os.path.join(OUTPUT_DIR, "unet_model.keras"),
                   csv_path=SUBSET_CSV,
                   image_dir=TRAIN_IMAGE_DIR,
                   target_size=(256, 256),
                   n_examples=5):
    # --- 1. Modell laden ---
    print("📦 Lade Modell...")
    model = load_model(
    model_path,
    custom_objects={
        "dice_score": dice_score,
        "iou_score": iou_score
    }
)

    # --- 2. Daten laden ---
    print("📥 Lade Testdaten...")
    X, y_true = load_dataset(csv_path=csv_path, image_dir=image_dir, target_size=target_size)

    # --- 3. Vorhersagen ---
    print("🔮 Berechne Vorhersagen...")
    y_pred = model.predict(X)
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)

    # --- 4. Metriken berechnen ---
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred_bin.reshape(-1)

    iou = jaccard_score(y_true_flat, y_pred_flat)
    dice = f1_score(y_true_flat, y_pred_flat)

    print(f"📊 Evaluationsergebnisse:")
    print(f"   - IoU (Jaccard Index): {iou:.4f}")
    print(f"   - Dice Score:          {dice:.4f}")

    # --- 5. Visualisierung ---
    show_examples(X, y_true, y_pred_bin, n=n_examples)
    
    # --- 6. Bilder für TensorBoard loggen ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(OUTPUT_DIR, "logs", "eval", f"eval_{timestamp}")
    log_images_to_tensorboard(log_dir, X, y_true, y_pred_bin, n=3, step=0)
    print(f"🖼️ Bilder geloggt unter: {log_dir}")


def show_examples(X, y_true, y_pred_bin, n=5):
    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(X[i])
        axes[0].set_title("Eingabebild")
        axes[1].imshow(y_true[i].squeeze(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(y_pred_bin[i].squeeze(), cmap="gray")
        axes[2].set_title("Vorhersage")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

def log_images_to_tensorboard(log_dir, X, y_true, y_pred, n=3, step=0):
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        for i in range(n):
            image = X[i]
            gt_mask = y_true[i].squeeze()
            pred_mask = y_pred[i].squeeze()

            # --- Overlays erzeugen ---
            overlay_gt = overlay_mask_on_image(image, gt_mask, color=(0, 255, 0))  # Grün
            overlay_pred = overlay_mask_on_image(image, pred_mask, color=(255, 0, 0))  # Rot

            # Batch-Dimension ergänzen
            overlay_gt = tf.expand_dims(overlay_gt, axis=0)
            overlay_pred = tf.expand_dims(overlay_pred, axis=0)

            tf.summary.image(f"Sample_{i}/GT_Overlay", overlay_gt, step=step)
            tf.summary.image(f"Sample_{i}/Pred_Overlay", overlay_pred, step=step)

            # kombiniertes Overlay
            overlay_combined = overlay_comparison_on_image(image, gt_mask, pred_mask)
            overlay_combined = tf.expand_dims(overlay_combined, axis=0)
            tf.summary.image(f"Sample_{i}/Comparison_Overlay", overlay_combined, step=step)
            
def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlay eine binäre Maske über ein RGB-Bild.
    - image: (H, W, 3), float32 [0, 1]
    - mask: (H, W), float32 [0, 1]
    - color: Overlay-Farbe
    """
    import numpy as np

    image = (image * 255).astype(np.uint8)
    mask = (mask > 0.5).astype(np.uint8)

    overlay = image.copy()
    color_array = np.array(color).reshape(1, 1, 3)
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    overlay = np.where(mask_rgb, overlay * (1 - alpha) + color_array * alpha, overlay)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def overlay_comparison_on_image(image, gt_mask, pred_mask, alpha=0.5):
    """
    Overlay GT vs. Prediction:
    - Grün = TP (GT & Pred)
    - Rot = FP (Pred ohne GT)
    - Blau = FN (GT ohne Pred)
    """
    import numpy as np

    image = (image * 255).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)
    pred = (pred_mask > 0.5).astype(np.uint8)

    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    overlay = image.copy()
    color_overlay = np.zeros_like(overlay)

    color_overlay[tp] = [0, 255, 0]   # Green for TP
    color_overlay[fp] = [255, 0, 0]   # Red for FP
    color_overlay[fn] = [0, 0, 255]   # Blue for FN

    overlay = overlay * (1 - alpha) + color_overlay * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

# ============================================================
# 🚀 Automatische Evaluation aller best_model_*.keras Varianten
# ============================================================

def evaluate_all_best_models(n_examples=3, save_to="evaluation_results.csv"):
    results = []

    for mode in ["baseline", "dropout", "lowlr", "diceloss"]:
        print(f"\n📊 Starte Evaluation für BEST-MODEL: {mode.upper()}")
        model_path = os.path.join(OUTPUT_DIR, f"best_model_{mode}_resaved.h5")

        try:
            custom_objects = {
                "dice_score": dice_score,
                "iou_score": iou_score
            }
            if mode == "diceloss":
                custom_objects["dice_loss"] = dice_loss

            model = load_model(model_path, custom_objects=custom_objects)
            X, y_true = load_dataset(SUBSET_CSV, TRAIN_IMAGE_DIR, target_size=(256, 256))
            y_pred = model.predict(X)
            y_pred_bin = (y_pred > 0.5).astype(np.uint8)

            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred_bin.reshape(-1)

            iou = jaccard_score(y_true_flat, y_pred_flat)
            dice = f1_score(y_true_flat, y_pred_flat)

            # Log- und Beispielbilder (optional)
            show_examples(X, y_true, y_pred_bin, n=n_examples)
            log_dir = os.path.join(OUTPUT_DIR, "logs", "eval", f"{mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            log_images_to_tensorboard(log_dir, X, y_true, y_pred_bin, n=n_examples)

            results.append({
                "Modus": mode,
                "IoU": round(iou, 4),
                "Dice Score": round(dice, 4)
            })

        except Exception as e:
            results.append({
                "Modus": mode,
                "IoU": None,
                "Dice Score": None,
                "Fehler": str(e)
            })

    # ✅ SAFE EXPORT TO CSV
    if results:
        try:
            df = pd.DataFrame(results)
            csv_path = os.path.join(OUTPUT_DIR, save_to)
            df.to_csv(csv_path, index=False)
            print(f"\n💾 Evaluation gespeichert unter: {csv_path}")
        except Exception as export_error:
            print(f"❌ Fehler beim Speichern der CSV: {export_error}")
    else:
        print("⚠️ Keine Ergebnisse vorhanden.")
        
    
    if __name__ == "__main__":
        evaluate_all_best_models()