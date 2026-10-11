"""Ekstraksi fitur: handcrafted (warna, tekstur, bentuk) dan deep embedding pretrained.

Deep embedding diambil dari layer sebelum classification head.
Tidak ada fine-tuning dan tidak ada label yang digunakan.
"""
import hashlib
import json
import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar modul config dan src bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from tqdm import tqdm

from config import EMB_DIR, IMG_SIZE, TRIM_BLACK_BORDERS
try:
    from src.preprocessing import load_rgb, preprocess
except ImportError:
    from preprocessing import load_rgb, preprocess

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
FEATURE_VERSION = "v2"  # naikkan jika logika ekstraksi berubah, agar cache otomatis dibuat ulang

EXTRACTOR_WEIGHTS = {
    "handcrafted": "skimage: RGB/HSV/LAB histogram + LBP + GLCM + HOG",
    "dinov2": "facebook/dinov2-base",
    "clip": "openai/clip-vit-base-patch32",
    "vit": "google/vit-base-patch16-224-in21k",
    "resnet": "torchvision ResNet50 IMAGENET1K_V2",
    "efficientnet": "torchvision EfficientNet-B0 IMAGENET1K_V1",
}


# ---------------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------------
def color_features(arr):
    """Histogram RGB, HSV, dan LAB (16 bin per channel), dinormalisasi per channel."""
    spaces = [
        (arr.astype(np.float32), [(0, 255)] * 3),
        (rgb2hsv(arr), [(0, 1)] * 3),
        (rgb2lab(arr), [(0, 100), (-128, 127), (-128, 127)]),
    ]
    feats = []
    for img, ranges in spaces:
        for c, rng in enumerate(ranges):
            h, _ = np.histogram(img[..., c], bins=16, range=rng)
            feats.append(h / max(h.sum(), 1))
    return np.concatenate(feats)


def texture_features(gray):
    """LBP (uniform, P=8, R=1) dan GLCM (Haralick-style) yang dirata-ratakan lintas sudut."""
    g8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(g8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    g64 = (gray * 63).round().astype(np.uint8)
    glcm = graycomatrix(
        g64,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    glcm_feats = [graycoprops(glcm, p).mean() for p in GLCM_PROPS]
    return np.concatenate([lbp_hist, glcm_feats])


def shape_features(gray):
    """HOG sebagai deskriptor bentuk dan tepi."""
    return hog(gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True)


def handcrafted_vector(arr):
    """arr: uint8 HxWx3. Setiap blok di-L2-normalisasi agar blok HOG tidak mendominasi."""
    gray = rgb2gray(arr)
    blocks = [color_features(arr), texture_features(gray), shape_features(gray)]
    return np.concatenate([b / (np.linalg.norm(b) + 1e-8) for b in blocks])


# ---------------------------------------------------------------------------
# Deep embeddings
# ---------------------------------------------------------------------------
class DeepEncoder:
    """Pretrained encoder tanpa classification head. Satu vektor per citra."""

    def __init__(self, name, device=None):
        if name not in EXTRACTOR_WEIGHTS or name == "handcrafted":
            raise ValueError(f"Encoder tidak dikenal: {name}")
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.transform = None

        if name in ("dinov2", "clip", "vit"):
            from transformers import AutoImageProcessor, AutoModel, CLIPModel
            model_id = EXTRACTOR_WEIGHTS[name]
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = (CLIPModel if name == "clip" else AutoModel).from_pretrained(model_id)
        elif name == "resnet":
            from torchvision.models import ResNet50_Weights, resnet50
            w = ResNet50_Weights.IMAGENET1K_V2
            self.model = resnet50(weights=w)
            self.model.fc = torch.nn.Identity()
            self.transform = w.transforms(resize_size=IMG_SIZE, crop_size=IMG_SIZE)
        elif name == "efficientnet":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            w = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.model = efficientnet_b0(weights=w)
            self.model.classifier = torch.nn.Identity()
            self.transform = w.transforms(resize_size=IMG_SIZE, crop_size=IMG_SIZE)

        self.model.eval().to(self.device)

    @torch.inference_mode()
    def encode(self, pils):
        if self.processor is not None:
            # Citra sudah IMG_SIZE x IMG_SIZE dari preprocess(), resize/crop tidak diperlukan.
            inputs = self.processor(images=pils, return_tensors="pt",
                                    do_resize=False, do_center_crop=False)
            pix = inputs["pixel_values"].to(self.device)
        else:
            pix = torch.stack([self.transform(im) for im in pils]).to(self.device)

        if self.name == "dinov2":
            h = self.model(pixel_values=pix).last_hidden_state
            emb = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        elif self.name == "clip":
            emb = self.model.get_image_features(pixel_values=pix)
        elif self.name == "vit":
            emb = self.model(pixel_values=pix).last_hidden_state[:, 0]
        else:  # resnet, efficientnet
            emb = self.model(pix)
        return emb.float().cpu().numpy()


def _encode_with_fallback(enc, pils):
    """Jika GPU kehabisan memori, batch dibelah dua secara rekursif hingga muat."""
    try:
        return enc.encode(pils)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(pils) == 1:
            raise
        mid = len(pils) // 2
        return np.concatenate([_encode_with_fallback(enc, pils[:mid]),
                               _encode_with_fallback(enc, pils[mid:])])


def _load_processed(p):
    try:
        return preprocess(load_rgb(p))
    except Exception as e:
        raise RuntimeError(f"Gagal memuat citra {p}: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------------------
# Driver dengan cache tervalidasi
# ---------------------------------------------------------------------------
def _cache_signature(paths, name):
    """Sidik jari cache: daftar nama file (berurutan), extractor, versi, dan setting preprocessing.
    Memakai nama file, bukan path absolut, agar cache tetap valid saat folder dipindah."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode("utf-8"))
        h.update(b"\n")
    h.update(json.dumps({
        "name": name, "weights": EXTRACTOR_WEIGHTS[name], "img_size": IMG_SIZE,
        "trim": TRIM_BLACK_BORDERS, "version": FEATURE_VERSION,
    }, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def extract_all(paths, name, batch_size=32, force=False):
    """Ekstrak fitur untuk semua path (urutan dipertahankan) dengan cache tervalidasi."""
    if name not in EXTRACTOR_WEIGHTS:
        raise ValueError(f"Extractor tidak dikenal: {name}")

    out_dir = EMB_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy, out_meta = out_dir / "embeddings.npy", out_dir / "meta.json"
    signature = _cache_signature(paths, name)

    if out_npy.exists() and out_meta.exists() and not force:
        meta = json.loads(out_meta.read_text(encoding="utf-8"))
        if meta.get("signature") == signature:
            feats = np.load(out_npy)
            if feats.shape[0] == len(paths):
                print(f"[{name}] memakai cache valid: {feats.shape}")
                return feats
        print(f"[{name}] cache tidak cocok dengan daftar citra saat ini, dihitung ulang")

    if name == "handcrafted":
        feats = np.stack([handcrafted_vector(np.asarray(_load_processed(p)))
                          for p in tqdm(paths, desc="handcrafted")])
    else:
        enc = DeepEncoder(name)
        chunks = []
        for i in tqdm(range(0, len(paths), batch_size), desc=name):
            pils = [_load_processed(p) for p in paths[i:i + batch_size]]
            chunks.append(_encode_with_fallback(enc, pils))
        feats = np.concatenate(chunks)

    if feats.shape[0] != len(paths):
        raise RuntimeError(f"[{name}] jumlah fitur {feats.shape[0]} != jumlah citra {len(paths)}")
    if not np.isfinite(feats).all():
        bad = int((~np.isfinite(feats)).any(axis=1).sum())
        raise RuntimeError(f"[{name}] {bad} citra menghasilkan NaN/inf. Periksa input.")

    const = int((feats.std(axis=0) == 0).sum())
    if const:
        print(f"[{name}] peringatan: {const} dimensi bernilai konstan (akan dinormalisasi oleh StandardScaler)")

    np.save(out_npy, feats)
    out_meta.write_text(json.dumps({"signature": signature, "n": len(paths),
                                    "dim": int(feats.shape[1]),
                                    "weights": EXTRACTOR_WEIGHTS[name]}, indent=2),
                        encoding="utf-8")
    return feats