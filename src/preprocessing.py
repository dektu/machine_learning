"""Quality control dan preprocessing citra.

Tidak ada label yang digunakan di modul ini.
"""
import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar modul config dan src bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from config import (
    BLACK_PIXEL_THRESHOLD,
    BLACK_ROW_COL_MIN_VALID,
    DATA_PROCESSED,
    IMAGE_DIR,
    IMG_SIZE,
    MAX_RESIDUAL_BLACK,
    MIN_IMAGE_SIDE,
    MIN_KEEP_AREA_RATIO,
    PHASH_DUP_THRESHOLD,
    TRIM_BLACK_BORDERS,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root=IMAGE_DIR):
    """Hanya file gambar. File tersembunyi dan artefak macOS (._*) dilewati."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Folder citra tidak ditemukan: {root}")
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in IMG_EXT:
            out.append(p)
    return sorted(out)


def load_rgb(path):
    """Memuat citra sebagai RGB. EXIF orientation diperbaiki agar foto HP tidak terputar."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def check_quality(path):
    """Cek apakah citra dapat dibaca dan cukup besar. Mengembalikan dict dengan alasan kegagalan."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:  # file rusak, terpotong, atau format tidak valid
        return {"ok": False, "reason": f"corrupt:{type(e).__name__}", "w": None, "h": None}
    if min(w, h) < MIN_IMAGE_SIDE:
        return {"ok": False, "reason": "too_small", "w": w, "h": h}
    return {"ok": True, "reason": "ok", "w": w, "h": h}


def trim_black_borders(im):
    """Memangkas sudut hitam akibat rotasi/perspektif pada foto dataset.

    Heuristik konservatif: hanya baris dan kolom yang hampir seluruhnya tidak hitam yang
    dipertahankan. Crop dibatalkan jika sisa area masih hitam atau terlalu banyak yang terbuang.
    """
    if not TRIM_BLACK_BORDERS:
        return im
    arr = np.asarray(im)
    valid = arr.max(axis=2) > BLACK_PIXEL_THRESHOLD
    rows = np.where(valid.mean(axis=1) >= BLACK_ROW_COL_MIN_VALID)[0]
    cols = np.where(valid.mean(axis=0) >= BLACK_ROW_COL_MIN_VALID)[0]
    if rows.size == 0 or cols.size == 0:
        return im

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    h, w = valid.shape
    if (y1 - y0) * (x1 - x0) < MIN_KEEP_AREA_RATIO * h * w:
        return im

    residual = 1 - valid[y0:y1, x0:x1].mean()
    if residual > MAX_RESIDUAL_BLACK:
        return im
    return im.crop((x0, y0, x1, y1))


def preprocess(im):
    """Trim border hitam, lalu letterbox ke IMG_SIZE x IMG_SIZE tanpa distorsi rasio."""
    im = trim_black_borders(im)
    return ImageOps.pad(im, (IMG_SIZE, IMG_SIZE), color=(0, 0, 0), centering=(0.5, 0.5))


def remove_near_duplicates(hashes):
    """Greedy: simpan kemunculan pertama, tandai yang mirip sebagai duplicate.

    Mengembalikan indeks yang dipertahankan dan dict {indeks_duplicate: indeks_asal}.
    """
    kept, dup_of = [], {}
    for i, h in enumerate(hashes):
        if h is None:
            continue
        match = next((j for j in kept if hashes[j] - h <= PHASH_DUP_THRESHOLD), None)
        if match is None:
            kept.append(i)
        else:
            dup_of[i] = match
    return kept, dup_of


def run_quality_control(root=IMAGE_DIR):
    """Menjalankan QC dan menyimpan manifest. Mengembalikan DataFrame citra yang dipakai."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    paths = list_images(root)
    if not paths:
        raise FileNotFoundError(f"Tidak ada citra gambar di {root}")

    rows, hashes = [], []
    for p in tqdm(paths, desc="quality check"):
        q = check_quality(p)
        h = None
        if q["ok"]:
            try:
                h = imagehash.phash(load_rgb(p))
            except Exception as e:
                q = {**q, "ok": False, "reason": f"load_error:{type(e).__name__}"}
        rows.append({"path": str(p), **q, "phash": str(h) if h is not None else ""})
        hashes.append(h)

    kept, dup_of = remove_near_duplicates(hashes)
    kept_set = set(kept)
    for i, row in enumerate(rows):
        row["near_duplicate_of"] = str(paths[dup_of[i]]) if i in dup_of else ""
        row["use"] = i in kept_set

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PROCESSED / "manifest.csv", index=False)
    print(f"Total: {len(df)} | dipakai: {int(df['use'].sum())} | "
          f"gagal QC: {int((~df['ok']).sum())} | near-duplicate: {len(dup_of)}")
    if df["use"].sum() < 2:
        raise RuntimeError("Citra yang lolos QC kurang dari 2. Periksa manifest.csv.")
    return df[df["use"]].reset_index(drop=True)