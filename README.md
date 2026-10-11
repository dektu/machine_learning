# BaliCarving Clustering

Pipeline unsupervised clustering untuk citra ukiran Bali.

## Struktur

```
balicarving-clustering/
├── config.py                  # seluruh parameter global
├── run_pipeline.py            # orkestrator utama
├── inspect_dataset.py         # inspeksi dataset dan verifikasi anotasi
├── requirements.txt
├── data/
│   └── raw/<nama_kelas>/*.jpg # opsional: nama folder hanya untuk evaluasi eksternal
├── embeddings/                # cache fitur per extractor
├── results/
│   ├── metrics/               # experiments.csv, candidates.csv
│   ├── plots/                 # UMAP, perbandingan vs ground truth, dendrogram
│   └── cluster_samples/       # representative dan outlier per cluster
├── best_model/                # artifact model terbaik
└── src/
    ├── preprocessing.py       # QC, pHash duplicate, letterbox resize
    ├── feature_extraction.py  # handcrafted + DINOv2/CLIP/ViT/ResNet/EfficientNet
    ├── dimensionality_reduction.py
    ├── clustering.py          # K-Means, Agglomerative, HDBSCAN, GMM
    ├── evaluation.py          # internal, stabilitas, eksternal, ranking
    ├── labels.py              # pemuatan anotasi CSV & label folder
    └── visualization.py
```

## Instalasi

```bash
pip install -r requirements.txt
```

GPU sangat disarankan untuk ekstraksi deep embedding. Pada pemanggilan pertama, model pretrained akan diunduh dari Hugging Face dan torchvision, sehingga koneksi internet diperlukan.

## Menjalankan

```bash
# tanpa label (murni unsupervised)
python run_pipeline.py

# dengan evaluasi eksternal terhadap nama folder kelas
python run_pipeline.py --labels-from-folders

# subset embedding
python run_pipeline.py --embeddings dinov2 handcrafted
```

## Catatan metodologi

- Label tidak pernah masuk ke ekstraksi fitur atau clustering. Dengan `--labels-from-folders`, label hanya dipakai untuk kolom ARI, NMI, homogeneity, completeness, dan V-measure.
- PCA memakai explained variance target yang sama (`PCA_VARIANCE`) untuk semua embedding, sehingga perbandingan antar-embedding lebih adil.
- UMAP hanya untuk visualisasi, bukan input clustering.
- Metrik internal HDBSCAN dihitung pada sampel non-noise dan rasio noise selalu dilaporkan. Konfigurasi dengan noise di atas `MAX_NOISE_RATIO` tidak masuk ranking.
- Pemilihan model: peringkat rata-rata Silhouette, DBI, dan CH untuk memilih kandidat, lalu stabilitas (ARI/NMI antar-seed dan subsampling) untuk memilih model terbaik di antara kandidat tersebut.
- Algoritma deterministik (Agglomerative, HDBSCAN) memiliki skor stabilitas seed = 1.0 secara trivial. Untuk algoritma ini, perhatikan skor subsampling.
- Dendrogram ditampilkan untuk membantu memeriksa struktur hierarkis, tetapi tidak dipakai untuk memilih k secara otomatis.