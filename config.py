"""Konfigurasi global pipeline BaliCarving.

Struktur dataset yang diharapkan (hasil ekstraksi BaliCarving.zip):

    data/raw/
    ├── JPEGImages/                      <- seluruh citra; dipakai untuk clustering
    ├── classification_*_ukiran.csv      <- anotasi; HANYA untuk evaluasi eksternal
    └── *.pkl                            <- tidak dipakai (adjacency/word2vec, bukan fitur citra)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
IMAGE_DIR = DATA_RAW / "JPEGImages"
LABEL_DIR = DATA_RAW
DATA_PROCESSED = ROOT / "data" / "processed"
EMB_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
SAMPLES_DIR = RESULTS_DIR / "cluster_samples"
BEST_DIR = ROOT / "best_model"

SEED = 42

# Preprocessing
IMG_SIZE = 224
MIN_IMAGE_SIDE = 64
PHASH_DUP_THRESHOLD = 4          # Hamming distance pHash <= nilai ini dianggap near-duplicate
TRIM_BLACK_BORDERS = True        # foto dataset memiliki sudut hitam hasil rotasi
BLACK_PIXEL_THRESHOLD = 20       # piksel dengan max(R,G,B) <= nilai ini dianggap hitam
BLACK_ROW_COL_MIN_VALID = 0.97   # baris/kolom dipertahankan jika >= 97% piksel tidak hitam
MAX_RESIDUAL_BLACK = 0.02        # jika hasil crop masih >2% hitam, crop dibatalkan
MIN_KEEP_AREA_RATIO = 0.5        # crop tidak boleh membuang lebih dari 50% luas citra

# Pembacaan anotasi (kolom dicari tanpa membedakan huruf besar/kecil)
FILENAME_COLUMNS = ["filename", "file_name", "image", "image_name", "img", "path", "name"]
LABEL_COLUMNS = ["label", "class", "kelas", "kategori", "category", "motif"]
MOTIF_CLASSES = ["Barong", "Cina", "Punggel", "Keketusan", "Daun", "Gajah", "Goak"]

# Reduksi dimensi
PCA_VARIANCE = 0.95              # explained variance yang sama untuk semua embedding agar perbandingan adil
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)

# Grid clustering
K_RANGE = range(2, 11)
AGGLO_LINKAGES = ["ward", "complete", "average"]
HDBSCAN_GRID = {
    "min_cluster_size": [5, 10, 20],
    "min_samples": [None, 5],
    "method": ["eom", "leaf"],
}
GMM_GRID = {
    "n_components": list(range(2, 11)),
    "covariance_type": ["full", "diag"],
}

# Seleksi dan evaluasi
MAX_NOISE_RATIO = 0.30
TOP_K_STABILITY = 3
STABILITY_RUNS = 10
STABILITY_SUBSAMPLE = 0.8

# Visualisasi
N_SAMPLES_PER_CLUSTER = 6

DEEP_MODELS = ["dinov2", "clip", "vit", "resnet", "efficientnet"]