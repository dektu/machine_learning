"""Evaluasi: metrik internal, stabilitas, dan metrik eksternal.

Label ground truth hanya boleh digunakan di external_metrics().
"""
import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar modul config dan src bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)

from config import MAX_NOISE_RATIO, SEED, STABILITY_RUNS, STABILITY_SUBSAMPLE


def internal_metrics(Z, labels):
    """Metrik dihitung pada sampel non-noise (label -1). Rasio noise selalu dilaporkan."""
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = {"n_clusters": n_clusters, "noise_ratio": 1 - mask.mean(),
           "silhouette": np.nan, "dbi": np.nan, "ch": np.nan}
    if n_clusters < 2 or mask.sum() <= n_clusters:
        return out

    Zm, lm = Z[mask], labels[mask]
    out["silhouette"] = silhouette_score(Zm, lm, sample_size=min(5000, len(lm)), random_state=SEED)
    out["dbi"] = davies_bouldin_score(Zm, lm)
    out["ch"] = calinski_harabasz_score(Zm, lm)
    return out


def stability(Z, cluster_fn, n_runs=STABILITY_RUNS, subsample=STABILITY_SUBSAMPLE, seed=SEED):
    """Dua sumber ketidakpastian:

    1. Variasi seed: jalankan ulang pada seluruh data dengan seed berbeda, bandingkan ARI/NMI pairwise.
    2. Subsampling: ambil sebagian data, cluster ulang, bandingkan dengan referensi pada subset yang sama.

    Catatan: algoritma deterministik (agglomerative, HDBSCAN) akan memberi skor seed = 1.0
    secara trivial; untuk algoritma tersebut, subsampling adalah ukuran stabilitas yang bermakna.
    """
    rng = np.random.default_rng(seed)
    ref = cluster_fn(Z, seed)

    seed_runs = [cluster_fn(Z, seed + i + 1) for i in range(n_runs)]
    seed_ari, seed_nmi = [], []
    for i, a in enumerate(seed_runs):
        for b in seed_runs[i + 1:]:
            seed_ari.append(adjusted_rand_score(a, b))
            seed_nmi.append(normalized_mutual_info_score(a, b))

    sub_ari, sub_nmi = [], []
    n = len(Z)
    for _ in range(n_runs):
        idx = rng.choice(n, size=int(subsample * n), replace=False)
        lab_sub = cluster_fn(Z[idx], seed)
        sub_ari.append(adjusted_rand_score(ref[idx], lab_sub))
        sub_nmi.append(normalized_mutual_info_score(ref[idx], lab_sub))

    return {
        "ari_seed_mean": np.mean(seed_ari), "ari_seed_std": np.std(seed_ari),
        "nmi_seed_mean": np.mean(seed_nmi),
        "ari_subsample_mean": np.mean(sub_ari), "ari_subsample_std": np.std(sub_ari),
        "nmi_subsample_mean": np.mean(sub_nmi),
    }


def external_metrics(y_true, y_pred):
    """Evaluasi eksternal terhadap label. Sampel noise (-1) dikeluarkan; rasionya dilaporkan terpisah."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_pred != -1
    h, c, v = homogeneity_completeness_v_measure(y_true[mask], y_pred[mask])
    return {
        "ARI": adjusted_rand_score(y_true[mask], y_pred[mask]),
        "NMI": normalized_mutual_info_score(y_true[mask], y_pred[mask]),
        "homogeneity": h, "completeness": c, "v_measure": v,
    }


def rank_configs(df):
    """Peringkat rata-rata dari Silhouette (naik), DBI (turun), dan CH (naik).

    Konfigurasi dengan noise melebihi MAX_NOISE_RATIO dan metrik yang tidak valid dikeluarkan.
    Metrik eksternal TIDAK ikut dalam peringkat.
    """
    d = df.dropna(subset=["silhouette", "dbi", "ch"]).copy()
    d = d[d["noise_ratio"] <= MAX_NOISE_RATIO].copy()
    d["r_sil"] = d["silhouette"].rank(ascending=False)
    d["r_dbi"] = d["dbi"].rank(ascending=True)
    d["r_ch"] = d["ch"].rank(ascending=False)
    d["mean_rank"] = d[["r_sil", "r_dbi", "r_ch"]].mean(axis=1)
    return d.sort_values("mean_rank").reset_index(drop=True)