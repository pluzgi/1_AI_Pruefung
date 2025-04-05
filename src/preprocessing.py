import os
import pandas as pd
from src.config import TRAIN_CSV, SUBSET_CSV

def create_paired_subset(csv_path=TRAIN_CSV, output_path=SUBSET_CSV, num_pairs=500):
    """
    Erstellt ein Subset mit num_pairs echten Real/AI-Bildpaaren (je 1x Label 0, 1x Label 1),
    basierend auf aufeinanderfolgenden Zeilen mit unterschiedlichen Labels.
    Dabei wird die Originalreihenfolge des CSV beibehalten.
    """
    df = pd.read_csv(csv_path)

    # Keine zusätzliche Sortierung – wir verlassen uns auf die im CSV vorgegebene Reihenfolge!
    pairs = []
    i = 0

    while i < len(df) - 1 and len(pairs) < num_pairs:
        l1, l2 = df.loc[i, "label"], df.loc[i + 1, "label"]

        if l1 != l2:
            pairs.append(df.loc[[i, i + 1]])
            i += 2
        else:
            i += 1

    if len(pairs) < num_pairs:
        print(f"⚠️ Achtung: Nur {len(pairs)} Paare gefunden (gewünscht: {num_pairs})")

    df_subset = pd.concat(pairs).reset_index(drop=True)
    df_subset.to_csv(output_path, index=False)

    print(f"✅ {len(pairs)} Paare gespeichert unter: {output_path}")
    print("📊 Verteilung:")
    print(df_subset["label"].value_counts())

# Direkt ausführen
if __name__ == "__main__":
    create_paired_subset()
