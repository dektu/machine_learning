"""Skrip inspeksi dataset dan anotasi BaliCarving.

Fungsi:
1. Memeriksa keberadaan folder citra (JPEGImages atau subfolder kelas).
2. Menghitung distribusi format gambar dan resolusi.
3. Mendeteksi struktur kelas jika gambar tersusun dalam subfolder.
4. Mendeteksi dan menguji kecocokan CSV anotasi dengan file citra di disk.
5. Memberikan panduan perintah CLI run_pipeline.py yang paling sesuai.

Contoh:
  python inspect_dataset.py
  python inspect_dataset.py --data-dir data/raw
  python inspect_dataset.py --labels-csv data/raw/classification_trainval_ukiran.csv
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

import config as C
from src.labels import _detect_column, load_annotations
from src.preprocessing import IMG_EXT, check_quality, list_images


def inspect_images(data_dir: Path):
    print(f"\n[1] Memeriksa direktori citra: {data_dir.resolve()}")
    if not data_dir.exists():
        print(f"  [PERINGATAN] Folder {data_dir} belum ditemukan!")
        print(f"  Pastikan Anda telah mengekstrak BaliCarving ke folder 'data/raw/'.")
        return []

    try:
        paths = list_images(data_dir)
    except FileNotFoundError:
        paths = []

    print(f"  Total citra ditemukan: {len(paths)}")
    if not paths:
        return []

    # Cek ekstensi
    ext_counts = Counter(p.suffix.lower() for p in paths)
    print(f"  Format ekstensi: {dict(ext_counts)}")

    # Cek hierarki folder (apakah ada folder kelas)
    subfolders = Counter(p.parent.name for p in paths)
    if len(subfolders) > 1 or (len(subfolders) == 1 and list(subfolders.keys())[0] != data_dir.name):
        print(f"  Terdeteksi struktur subfolder kelas ({len(subfolders)} kelas):")
        for folder_name, count in sorted(subfolders.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    - {folder_name}: {count} citra")
        if len(subfolders) > 10:
            print(f"    ... dan {len(subfolders) - 10} kelas lainnya.")

    # Cek kualitas sampel (hingga 50 citra pertama)
    sample_paths = paths[:50]
    corrupt = 0
    small = 0
    sizes = []
    for p in sample_paths:
        q = check_quality(p)
        if not q["ok"]:
            if "corrupt" in q["reason"]:
                corrupt += 1
            elif q["reason"] == "too_small":
                small += 1
        elif q["w"] and q["h"]:
            sizes.append((q["w"], q["h"]))

    print(f"  Sampel QC (50 citra pertama):")
    print(f"    - Rusak/Corrupt: {corrupt}")
    print(f"    - Terlalu kecil (< {C.MIN_IMAGE_SIDE}px): {small}")
    if sizes:
        widths, heights = zip(*sizes)
        print(f"    - Rata-rata resolusi: {int(sum(widths)/len(widths))} x {int(sum(heights)/len(heights))} px")

    return paths


def inspect_annotations(paths: list, labels_csv: list = None, filename_col: str = None, label_col: str = None):
    print(f"\n[2] Memeriksa file anotasi CSV:")
    csv_candidates = []
    if labels_csv:
        csv_candidates = [Path(p) for p in labels_csv]
    else:
        # Otomatis cari CSV di data/raw/
        if C.DATA_RAW.exists():
            csv_candidates = list(C.DATA_RAW.glob("*.csv"))

    if not csv_candidates:
        print("  Tidak ada file CSV anotasi yang ditentukan atau ditemukan di data/raw/.")
        print("  Pipeline dapat dijalankan secara unsupervised murni atau menggunakan --labels-from-folders.")
        return

    path_names = {p.name for p in paths}
    path_stems = {p.stem for p in paths}

    for csv_path in csv_candidates:
        print(f"\n  File CSV: {csv_path.resolve()}")
        if not csv_path.exists():
            print(f"    [ERROR] File tidak ditemukan!")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"    [ERROR] Gagal membaca CSV: {e}")
            continue

        print(f"    Jumlah baris: {len(df)} | Kolom: {list(df.columns)}")

        detected_fcol = filename_col or _detect_column(df.columns, C.FILENAME_COLUMNS)
        detected_lcol = label_col or _detect_column(df.columns, C.LABEL_COLUMNS)
        motif_cols = [c for c in df.columns if c in getattr(C, "MOTIF_CLASSES", [])]
        is_multilabel_motif = len(motif_cols) >= 2 and not detected_lcol

        print(f"    Deteksi kolom filename: '{detected_fcol}'")
        if is_multilabel_motif:
            print(f"    Deteksi format label: Multi-Label Motif BaliCarving ({len(motif_cols)} motif: {motif_cols})")
        else:
            print(f"    Deteksi kolom label: '{detected_lcol}'")

        if not detected_fcol or (not detected_lcol and not is_multilabel_motif):
            print("    [PERINGATAN] Gagal mendeteksi kolom filename atau label secara otomatis.")
            print(f"    Gunakan --filename-col dan --label-col secara spesifik.")
            continue

        # Cek kecocokan dengan file citra
        try:
            mapping = load_annotations([csv_path], detected_fcol, detected_lcol)
        except Exception as e:
            print(f"    [ERROR] Gagal memetakan anotasi: {e}")
            continue

        matches_name = sum(1 for p in paths if p.name in mapping)
        matches_stem = sum(1 for p in paths if p.stem in mapping)
        total_matches = max(matches_name, matches_stem)

        print(f"    Kecocokan dengan citra di disk: {total_matches}/{len(paths)} citra")
        unique_labels = set(mapping.values())
        print(f"    Jumlah kelas unik di CSV: {len(unique_labels)}")

        if paths and total_matches == 0:
            sample_csv_files = [str(x) for x in df[detected_fcol].head(3).tolist()]
            sample_disk_files = [p.name for p in paths[:3]]
            print(f"    [MISMATCH DETAIL]")
            print(f"      Contoh nama file di CSV: {sample_csv_files}")
            print(f"      Contoh nama file di disk: {sample_disk_files}")


def main():
    ap = argparse.ArgumentParser(description="Inspeksi dataset dan anotasi BaliCarving")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Folder citra (default: C.IMAGE_DIR atau C.DATA_RAW)")
    ap.add_argument("--labels-csv", nargs="+", type=Path, default=None,
                    help="File CSV anotasi untuk diuji kecocokannya")
    ap.add_argument("--filename-col", default=None, help="Nama kolom nama file di CSV")
    ap.add_argument("--label-col", default=None, help="Nama kolom label di CSV")
    args = ap.parse_args()

    # Prioritaskan data-dir yang ada
    target_dir = args.data_dir
    if target_dir is None:
        if C.IMAGE_DIR.exists():
            target_dir = C.IMAGE_DIR
        else:
            target_dir = C.DATA_RAW

    paths = inspect_images(target_dir)
    inspect_annotations(paths, args.labels_csv, args.filename_col, args.label_col)

    print("\n[3] Rekomendasi Eksekusi Pipeline:")
    print("  1. Unsupervised murni (tanpa label ground truth):")
    print("     python run_pipeline.py\n")
    print("  2. Dengan evaluasi eksternal dari folder kelas:")
    print("     python run_pipeline.py --labels-from-folders\n")
    print("  3. Dengan evaluasi eksternal dari CSV anotasi:")
    print("     python run_pipeline.py --labels-csv data/raw/nama_file_anotasi.csv\n")


if __name__ == "__main__":
    main()