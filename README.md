# Studi Kasus 1: Pengelompokan Motif Ukiran Bali

Pipeline Unsupervised Machine Learning untuk pengelompokan (*clustering*) citra motif ukiran tradisional Bali.

## Struktur Direktori

```
studi_kasus_1/
├── config.py                  # Seluruh konfigurasi dan hiperparameter global
├── run_pipeline.py            # Orkestrator utama seluruh tahapan pipeline
├── inspect_dataset.py         # Skrip inspeksi dataset dan verifikasi anotasi
├── requirements.txt           # Dependensi pustaka Python
├── data/
│   ├── raw/
│   │   ├── JPEGImages/        # 2.364 citra ukiran Bali (.jpg)
│   │   ├── classification_*_ukiran.csv # Anotasi multi-label motif ukiran (ground truth)
│   │   └── *.pkl              # Graf adjacency & Word2Vec (bukan model; tidak dipakai di clustering)
│   └── processed/             # Cache citra setelah Quality Control & preprocessing
├── embeddings/                # Cache ekstraksi fitur visual (.npz) per extractor
├── results/
│   ├── metrics/               # Hasil evaluasi: experiments.csv, candidates.csv
│   ├── plots/                 # Visualisasi: UMAP 2D, perbandingan vs ground truth, dendrogram
│   └── cluster_samples/       # Sampel citra representatif & outlier per cluster
├── best_model/                # Artefak model terbaik (.joblib, assignments.csv, config.json)
└── src/
    ├── preprocessing.py       # QC, pHash near-duplicate, border trim, letterbox resize
    ├── feature_extraction.py  # Handcrafted (GLCM, LBP, Color) + Deep (DINOv2, CLIP, ViT, ResNet)
    ├── dimensionality_reduction.py # Pipeline L2 -> StandardScaler -> PCA (95% variance) + UMAP 2D
    ├── clustering.py          # K-Means, Agglomerative Hierarchical, HDBSCAN, GMM
    ├── evaluation.py          # Evaluasi internal (Silhouette, DBI, CH), stabilitas, dan eksternal
    ├── labels.py              # Parser anotasi CSV (multi-label motif) & struktur folder
    └── visualization.py       # Visualisasi scatter cluster, dendrogram, dan galeri sampel
```

## Dataset Motif Ukiran Bali

Dataset terdiri dari **2.364 citra ukiran Bali** dengan 7 motif utama:
1. **Barong** (figur mitologi Barong / topeng)
2. **Cina** (ornamen stilasi pengaruh Tiongkok)
3. **Punggel** (ujung sulur daun ikal khas Bali)
4. **Keketusan** (motif geometris hiasan tepi/bingkai)
5. **Daun** (patra tumbuhan/flora)
6. **Gajah** (motif ornamen gajah / asti)
7. **Goak** (motif ornamen burung gagak / goak)

> [!NOTE]
> File `.pkl` bawaan dataset (`ukiran_adj.pkl` dan `ukiranword2vec.pkl`) adalah data relasi graf & word embedding untuk model klasifikasi graf (ML-GCN). Pada tugas clustering ini, file `.pkl` **tidak digunakan**.

## Instalasi

```bash
pip install -r requirements.txt
```

> **Catatan:** GPU sangat disarankan untuk mempercepat ekstraksi deep embedding (DINOv2, CLIP, ViT, ResNet). Pada pemanggilan pertama, model pretrained akan diunduh secara otomatis.

## Menjalankan Pipeline

### 1. Inspeksi Dataset
Periksa kelengkapan file citra, format resolusi, dan kecocokan anotasi:
```bash
python inspect_dataset.py
```

### 2. Eksekusi Clustering

```bash
# Opsi A: Murni Unsupervised (Rekomendasi Utama Clustering)
# Mengelompokkan citra murni berdasarkan kemiripan visual tanpa label
python run_pipeline.py

# Opsi B: Dengan Evaluasi Eksternal Ground Truth
# Menggunakan anotasi CSV untuk mencatat metrik ARI, NMI, Homogeneity, Completeness
python run_pipeline.py --labels-csv data/raw/classification_trainval_ukiran.csv

# Opsi C: Menjalankan subset model tertentu (misal: ekstraksi cepat / tanpa internet)
python run_pipeline.py --embeddings handcrafted resnet
```

## Metodologi dan Evaluasi

1. **Preprocessing:** Pemeriksaan file korup, deteksi near-duplicate menggunakan *perceptual hash (pHash)*, pemotongan sudut hitam sisa rotasi foto, dan normalisasi ukuran 224x224.
2. **Ekstraksi Fitur:** Ekstraksi fitur visual komprehensif menggunakan kombinasi tekstur/warna klasik (*handcrafted*) dan representasi semantik mendalam (*foundation vision models*).
3. **Reduksi Dimensi:** Pipeline L2 Normalizer -> StandardScaler -> PCA dengan target *explained variance* seragam (95%) agar perbandingan antar-metode adil. UMAP 2D digunakan murni untuk visualisasi.
4. **Pemilihan Model Terbaik:**
   - **Internal Metrics:** Peringkat kombinasi *Silhouette Score* (tinggi), *Davies-Bouldin Index* (rendah), dan *Calinski-Harabasz Index* (tinggi) untuk menyaring kandidat terbaik.
   - **Uji Stabilitas:** Kandidat terbaik diuji konsistensinya terhadap variasi random seed dan subsampling (bootstrap 80%) menggunakan pairwise ARI/NMI.
5. **Artefak Output:** Model reduksi dimensi terbaik disimpan ke `best_model/` bersama dengan pemetaan cluster per citra (`cluster_assignments.csv`) dan visualisasi sampel citra di `results/cluster_samples/`.