import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar modul config dan src bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402

from config import N_SAMPLES_PER_CLUSTER, SAMPLES_DIR  # noqa: E402
try:
    from src.preprocessing import load_rgb, preprocess
except ImportError:
    from preprocessing import load_rgb, preprocess


def _scatter(ax, coords, labels, title):
    for lab in np.unique(labels):
        m = labels == lab
        name = "noise" if lab == -1 else f"cluster {lab}"
        kw = {"color": "lightgray"} if lab == -1 else {}
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.7, label=name, **kw)
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)


def scatter_2d(coords, labels, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter(ax, coords, labels, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_with_truth(coords, pred, truth, path):
    """Perbandingan cluster hasil model vs ground truth. Hanya untuk evaluasi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], coords, pred, "Cluster assignment")
    _scatter(axes[1], coords, truth, "Ground truth (evaluasi saja)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grid(paths, title, out_path, cols=6):
    if not paths:
        return
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p in zip(axes.ravel(), paths):
        ax.imshow(preprocess(load_rgb(p)))
        ax.set_title(Path(p).name, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def cluster_samples(Z, paths, labels, out_dir=SAMPLES_DIR, n=N_SAMPLES_PER_CLUSTER, seed=0):
    """Per cluster: sampel terdekat centroid (representatif) dan terjauh (boundary/outlier).

    Untuk noise (-1), centroid tidak bermakna sehingga diambil sampel acak.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        name = "noise" if lab == -1 else f"cluster_{lab}"
        if lab == -1:
            rep = rng.choice(idx, size=min(n, len(idx)), replace=False)
            outl = np.array([], dtype=int)
        else:
            d = np.linalg.norm(Z[idx] - Z[idx].mean(axis=0), axis=1)
            order = np.argsort(d)
            rep = idx[order[:n]]
            outl = idx[order[-n:]]
        _grid([paths[i] for i in rep], f"{name} - representatif (dekat centroid)",
              out_dir / f"{name}_representative.png")
        _grid([paths[i] for i in outl], f"{name} - outlier / boundary",
              out_dir / f"{name}_outlier.png")


def nearest_neighbors(Z, paths, query_idx, k=5):
    """Untuk interpretabilitas: k citra terdekat dari sebuah query dalam ruang embedding."""
    d = np.linalg.norm(Z - Z[query_idx], axis=1)
    order = np.argsort(d)[1:k + 1]
    return [(paths[i], float(d[i])) for i in order]


def dendrogram_plot(Z, path, method="ward"):
    L = linkage(Z, method=method)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(L, truncate_mode="lastp", p=30, ax=ax)
    ax.set_title(f"Dendrogram ({method} linkage, truncated)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)