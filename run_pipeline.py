"""Orkestrasi pipeline BaliCarving.

Alur:
  quality control -> ekstraksi fitur -> PCA -> sweep clustering
  -> ranking internal -> uji stabilitas -> visualisasi -> simpan best model

Label dari CSV hanya dipakai untuk evaluasi eksternal.

Contoh:
  python inspect_dataset.py                                  # cek struktur dan anotasi dulu
  python run_pipeline.py                                     # unsupervised murni
  python run_pipeline.py --labels-csv data/raw/classification_trainval_ukiran.csv
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

import config as C
from src.clustering import make_cluster_fn
from src.dimensionality_reduction import pca_reduce, umap_reduce
from src.evaluation import external_metrics, internal_metrics, rank_configs, stability
from src.feature_extraction import EXTRACTOR_WEIGHTS, extract_all
from src.labels import labels_from_subfolders, load_annotations, truth_for_paths
from src.preprocessing import run_quality_control
from src.visualization import cluster_samples, compare_with_truth, dendrogram_plot, scatter_2d


def iter_configs():
    for k in C.K_RANGE:
        yield "kmeans", {"k": k}
        for link in C.AGGLO_LINKAGES:
            yield "agglomerative", {"k": k, "linkage": link}
    for mcs in C.HDBSCAN_GRID["min_cluster_size"]:
        for ms in C.HDBSCAN_GRID["min_samples"]:
            for meth in C.HDBSCAN_GRID["method"]:
                yield "hdbscan", {"min_cluster_size": mcs, "min_samples": ms, "method": meth}
    for nc in C.GMM_GRID["n_components"]:
        for cov in C.GMM_GRID["covariance_type"]:
            yield "gmm", {"n_components": nc, "covariance_type": cov}


def sweep(name, Z, truth=None):
    """Menjalankan semua konfigurasi. Metrik eksternal hanya dicatat, tidak dipakai untuk memilih."""
    rows = []
    for method, params in tqdm(list(iter_configs()), desc=f"sweep {name}"):
        try:
            labels = make_cluster_fn(method, **params)(Z, C.SEED)
            row = {"embedding": name, "method": method, "params": json.dumps(params),
                   **internal_metrics(Z, labels)}
            if truth is not None:
                y, m = truth
                row["n_labeled"] = int(m.sum())
                row.update(external_metrics(y[m], labels[m]))
            rows.append(row)
        except Exception:
            continue
    return rows


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=C.IMAGE_DIR,
                    help="Folder citra (default: data/raw/JPEGImages)")
    ap.add_argument("--embeddings", nargs="+", default=["handcrafted"] + C.DEEP_MODELS)
    ap.add_argument("--labels-from-folders", action="store_true",
                    help="Ambil label evaluasi eksternal dari nama subfolder (data/raw/<kelas>/*.jpg)")
    ap.add_argument("--labels-csv", nargs="+", type=Path, default=None,
                    help="Satu atau lebih CSV anotasi (hanya untuk evaluasi eksternal)")
    ap.add_argument("--filename-col", default=None, help="Nama kolom nama file (jika deteksi gagal)")
    ap.add_argument("--label-col", default=None, help="Nama kolom label (jika deteksi gagal)")
    args = ap.parse_args()

    for d in [C.METRICS_DIR, C.PLOTS_DIR, C.SAMPLES_DIR, C.BEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Quality control (hanya dari folder citra)
    df = run_quality_control(args.data_dir)
    paths = df["path"].tolist()

    # Label: dimuat setelah daftar citra diketahui agar bisa dicocokkan (evaluasi eksternal saja)
    truth = None
    if args.labels_from_folders:
        mapping = labels_from_subfolders(paths)
        y, m = truth_for_paths(paths, mapping)
        n_classes = len(set(y[m])) if m.any() else 0
        print(f"Label folder: {n_classes} kelas terdeteksi | citra dengan label: {int(m.sum())}/{len(paths)}")
        if m.sum() < 2 or n_classes < 2:
            raise RuntimeError("Label dari folder tidak mencukupi (minimal 2 sampel dan 2 kelas).")
        truth = (y, m)
    elif args.labels_csv:
        mapping = load_annotations(args.labels_csv, args.filename_col, args.label_col)
        y, m = truth_for_paths(paths, mapping)
        print(f"Anotasi CSV: {len(mapping)} entri | citra dengan label: {int(m.sum())}/{len(paths)}")
        if m.sum() < 2:
            raise RuntimeError("Hampir tidak ada citra yang cocok dengan anotasi. "
                               "Periksa nama file di CSV (lihat inspect_dataset.py).")
        truth = (y, m)

    # 2-3. Ekstraksi fitur, PCA, dan sweep
    rows, reduced, pipes = [], {}, {}
    for name in args.embeddings:
        X = extract_all(paths, name)
        Z, pipe = pca_reduce(X)
        reduced[name], pipes[name] = Z, pipe
        print(f"[{name}] dimensi asli={X.shape[1]} -> PCA={Z.shape[1]} "
              f"(var={pipe.named_steps['pca'].explained_variance_ratio_.sum():.3f})")
        rows += sweep(name, Z, truth)

    exp = pd.DataFrame(rows)
    exp.to_csv(C.METRICS_DIR / "experiments.csv", index=False)

    # 4. Ranking internal multi-kriteria
    ranked = rank_configs(exp)
    if ranked.empty:
        raise RuntimeError("Tidak ada konfigurasi valid. Periksa data dan parameter.")

    # 5. Uji stabilitas pada kandidat teratas
    cand = []
    for _, r in tqdm(ranked.head(C.TOP_K_STABILITY).iterrows(),
                     total=min(C.TOP_K_STABILITY, len(ranked)), desc="stabilitas"):
        Z = reduced[r["embedding"]]
        fn = make_cluster_fn(r["method"], **json.loads(r["params"]))
        rec = r.to_dict()
        rec.update(stability(Z, fn))
        cand.append(rec)
    cand = pd.DataFrame(cand)
    cand["stability"] = (cand["ari_seed_mean"] + cand["ari_subsample_mean"]) / 2
    cand = cand.sort_values("stability", ascending=False).reset_index(drop=True)
    cand.to_csv(C.METRICS_DIR / "candidates.csv", index=False)

    # 6. Best model: stabilitas tertinggi di antara kandidat internal teratas
    best = cand.iloc[0]
    emb = best["embedding"]
    Z, pipe = reduced[emb], pipes[emb]
    labels = make_cluster_fn(best["method"], **json.loads(best["params"]))(Z, C.SEED)
    coords = umap_reduce(Z, n_components=2)

    # 7. Visualisasi dan inspeksi
    scatter_2d(coords, labels, f"{emb} + {best['method']}", C.PLOTS_DIR / "umap_clusters.png")
    if truth is not None:
        y, m = truth
        compare_with_truth(coords[m], labels[m], y[m], C.PLOTS_DIR / "umap_cluster_vs_truth.png")
    dendrogram_plot(Z, C.PLOTS_DIR / f"dendrogram_{emb}.png")
    cluster_samples(Z, paths, labels)

    # 8. Simpan artifact best model
    BEST = C.BEST_DIR
    joblib.dump(pipe, BEST / "dimensionality_reduction_model.joblib")
    assign = pd.DataFrame({"path": paths, "cluster": labels,
                           "umap_x": coords[:, 0], "umap_y": coords[:, 1]})
    if truth is not None:
        assign["label_eval_only"] = truth[0]
        assign["has_label"] = truth[1]
    assign.to_csv(BEST / "cluster_assignments.csv", index=False)

    config = {
        "feature_extractor": {"name": emb, "weights": EXTRACTOR_WEIGHTS[emb]},
        "dimensionality_reduction": {
            "method": "L2 -> StandardScaler -> PCA",
            "explained_variance_target": C.PCA_VARIANCE,
            "n_components": int(pipe.named_steps["pca"].n_components_),
        },
        "clustering": {"method": best["method"], "params": json.loads(best["params"])},
        "seed": C.SEED,
        "n_images": len(paths),
        "metrics": {k: best[k] for k in ["n_clusters", "noise_ratio", "silhouette", "dbi", "ch",
                                         "stability", "ari_seed_mean", "ari_subsample_mean"]},
        "external_evaluation": (
            {"n_labeled": int(best["n_labeled"]),
             **{k: best[k] for k in ["ARI", "NMI", "homogeneity", "completeness", "v_measure"]}}
            if truth is not None else None),
        "labels_from_folders": bool(args.labels_from_folders),
        "labels_csv": [str(p) for p in args.labels_csv] if args.labels_csv else None,
        "library_versions": {p: version(p) for p in
                             ["numpy", "scikit-learn", "scikit-image", "umap-learn", "torch"]},
    }
    with open(BEST / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=_json_default)

    print("\n=== Model terbaik ===")
    print(json.dumps(config, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()