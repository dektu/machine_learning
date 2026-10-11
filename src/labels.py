"""Manajemen label ground truth untuk evaluasi eksternal (CSV dan struktur folder).

Penting: Label TIDAK PERNAH digunakan dalam proses ekstraksi fitur, reduksi dimensi,
ataupun pemilihan model unsupervised. Label hanya digunakan untuk evaluasi eksternal
(ARI, NMI, homogeneity, completeness, v_measure).
"""
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

# Pastikan root project ada di sys.path agar modul config dan src bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C


def _detect_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    col_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def load_annotations(
    csv_paths: Union[str, Path, Iterable[Union[str, Path]]],
    filename_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> Dict[str, str]:
    """Membaca satu atau lebih CSV anotasi dan memetakan nama file ke label kelas."""
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    mapping: Dict[str, str] = {}
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"File anotasi CSV tidak ditemukan: {path}")

        df = pd.read_csv(path)
        f_col = filename_col or _detect_column(df.columns, C.FILENAME_COLUMNS)
        l_col = label_col or _detect_column(df.columns, C.LABEL_COLUMNS)

        if not f_col or not l_col:
            raise ValueError(
                f"Tidak dapat mendeteksi kolom nama file atau label pada {path.name}. "
                f"Kolom tersedia: {list(df.columns)}. "
                "Tentukan secara eksplisit menggunakan --filename-col dan --label-col."
            )

        for _, row in df.iterrows():
            raw_file = str(row[f_col]).strip()
            raw_label = str(row[l_col]).strip()
            if not raw_file or not raw_label or raw_label.lower() in ("nan", "none"):
                continue

            fname = Path(raw_file).name
            mapping[fname] = raw_label
            mapping[Path(fname).stem] = raw_label

    return mapping


def labels_from_subfolders(paths: Iterable[Union[str, Path]]) -> Dict[str, str]:
    """Mengekstrak label dari nama folder induk langsung (misal: data/raw/<nama_kelas>/*.jpg)."""
    mapping: Dict[str, str] = {}
    for p in paths:
        path = Path(p)
        parent_name = path.parent.name
        mapping[path.name] = parent_name
        mapping[path.stem] = parent_name
    return mapping


def truth_for_paths(
    paths: Iterable[Union[str, Path]],
    mapping: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mencocokkan daftar citra dengan mapping label.

    Mengembalikan:
      y: array label ground truth (str)
      m: boolean mask apakah citra memiliki label
    """
    labels = []
    mask = []
    for p in paths:
        path = Path(p)
        name = path.name
        stem = path.stem
        if name in mapping:
            labels.append(mapping[name])
            mask.append(True)
        elif stem in mapping:
            labels.append(mapping[stem])
            mask.append(True)
        else:
            labels.append("")
            mask.append(False)

    return np.array(labels, dtype=object), np.array(mask, dtype=bool)
