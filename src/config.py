import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Verzeichnisse
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# CSV-Dateien
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SUBSET_CSV = os.path.join(DATA_DIR, "train_subset.csv")

# Bildverzeichnisse
TRAIN_IMAGE_DIR = os.path.join(INPUT_DIR, "train_data")
TEST_IMAGE_DIR = os.path.join(INPUT_DIR, "test_data_v2")

# Beispielausgabe
if __name__ == "__main__":
    print("TRAIN_CSV:", TRAIN_CSV)
    print("TEST_CSV:", TEST_CSV)
    print("SUBSET_CSV:", SUBSET_CSV)
    print("TRAIN_IMAGE_DIR:", TRAIN_IMAGE_DIR)
    print("TEST_IMAGE_DIR:", TEST_IMAGE_DIR)