"""Algoritma clustering.

Semua fungsi menerima matriks fitur Z (hasil PCA) dan TIDAK menerima label.
"""
import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar modul config dan src bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from config import SEED


def kmeans(Z, k, seed=SEED):
    return KMeans(n_clusters=k, init="k-means++", n_init=10,
                  max_iter=300, random_state=seed).fit_predict(Z)


def agglomerative(Z, k, linkage="ward"):
    # ward hanya mendukung jarak euclidean (default)
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(Z)


def hdbscan(Z, min_cluster_size=10, min_samples=None, method="eom"):
    """Label -1 berarti noise/outlier."""
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   cluster_selection_method=method).fit_predict(Z)


def gmm(Z, n_components, covariance_type="full", seed=SEED, reg_covar=1e-3):
    try:
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                                reg_covar=reg_covar, n_init=3, random_state=seed).fit(Z)
        return model.predict(Z)
    except ValueError:
        try:
            model = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                                    reg_covar=1e-1, n_init=3, random_state=seed).fit(Z)
            return model.predict(Z)
        except Exception:
            model = GaussianMixture(n_components=n_components, covariance_type="diag",
                                    reg_covar=1e-2, n_init=3, random_state=seed).fit(Z)
            return model.predict(Z)


def gmm_bic(Z, n_components, covariance_type="full", seed=SEED, reg_covar=1e-3):
    try:
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                                reg_covar=reg_covar, n_init=3, random_state=seed).fit(Z)
        return model.bic(Z)
    except Exception:
        return float("nan")


def make_cluster_fn(algo, **params):
    """Membangun fungsi (Z, seed) -> labels yang seragam, dipakai untuk sweep dan uji stabilitas."""
    if algo == "kmeans":
        return lambda Z, seed: kmeans(Z, seed=seed, **params)
    if algo == "agglomerative":
        return lambda Z, seed: agglomerative(Z, **params)
    if algo == "hdbscan":
        return lambda Z, seed: hdbscan(Z, **params)
    if algo == "gmm":
        return lambda Z, seed: gmm(Z, seed=seed, **params)
    raise ValueError(f"Metode tidak dikenal: {algo}")