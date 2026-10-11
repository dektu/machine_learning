"""Reduksi dimensi: PCA untuk input clustering dan UMAP untuk visualisasi 2D."""
import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar modul config dan src bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
try:
    from umap import UMAP
except ImportError:
    UMAP = None

import config as C


def pca_reduce(X, variance_target=C.PCA_VARIANCE, seed=C.SEED):
    """Pipeline L2 Normalizer -> StandardScaler -> PCA dengan target explained variance.

    Menghasilkan representasi dimensi rendah terstandarisasi untuk input algoritma clustering.
    Target explained variance seragam di semua embedding agar perbandingan adil.
    """
    pipe = Pipeline([
        ("l2", Normalizer(norm="l2")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=variance_target, random_state=seed, svd_solver="full")),
    ])
    Z = pipe.fit_transform(X)
    return Z, pipe


def umap_reduce(Z, n_components=2, **custom_params):
    """Reduksi ke 2D menggunakan UMAP khusus untuk visualisasi, bukan untuk clustering.
    Jika umap-learn belum terpasang, secara otomatis menggunakan PCA 2D sebagai fallback.
    """
    if UMAP is not None:
        params = {**C.UMAP_PARAMS, **custom_params}
        reducer = UMAP(n_components=n_components, **params)
        return reducer.fit_transform(Z)
    else:
        print("[Peringatan] Library 'umap-learn' belum terpasang. Menggunakan PCA 2D sebagai fallback visualisasi.")
        from sklearn.decomposition import PCA as SkPCA
        return SkPCA(n_components=n_components, random_state=C.SEED).fit_transform(Z)
