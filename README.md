# Fake Image Segmentation – Studienarbeit

Dieses Projekt dient dem Aufbau einer vollständigen Machine Learning Pipeline zur Segmentierung und Detektion von KI-generierten Bildern.

## 🔍 Projektstruktur

```bash
.
├── data/                # train.csv, test.csv, Subset-CSV
├── input/               # train_data, test_data_v2 (Bilder)
├── masks/               # generierte Differenzmasken
├── notebooks/           # explorative Analysen
├── output/              # Logs, Modelle, TensorBoard
├── src/                 # Modularer Python-Code (model, data_loader, preprocessing, etc.)
├── pipeline.py          # Zentrales Einstiegs-Skript
└── requirements.txt     # Abhängigkeiten


⚙️ Setup

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

🚀 Ausführen
python pipeline.py

Dies führt folgende Schritte aus:

Generierung eines 1000-Bilder Subsets (500 Paare, real vs KI)
Laden & Vorverarbeitung (Resize, Normalisierung)
Aufbau eines U-Net Modells
Training mit Callbacks (EarlyStopping, ModelCheckpoint, TensorBoard)
Speicherung als .keras


📊 Evaluation
Nach dem Training:
python src/evaluate.py
→ Nutzt Dummy-Masken oder echte Differenzmasken zur Auswertung.

📈 TensorBoard
tensorboard --logdir=output/logs

