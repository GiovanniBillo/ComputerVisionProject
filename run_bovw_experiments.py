#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_bovw_experiments.py (refactored)

Optimized BoVW + SVM experiments with caching + parallelism.

Variants:
- BoVW (hard) + kNN
- BoVW (hard) + OneVsRest(LinearSVC)
- BoVW (hard) + chi2-kernel OneVsRest(SVC(kernel="precomputed"))
- BoVW (hard) + ECOC (MyECOC on OneVsRest(SVC))
- BoVW (soft) + OneVsRest(SVC)
- PMK (precomputed kernel) + SVC with kernel normalization

Dataset layout expected:
  data/train/<class>/*.jpg
  data/test/<class>/*.jpg

Caching (cache_dir):
  descriptors_train.pkl
  descriptors_test.pkl
  codebook_<tag>_k{K}_md{max_desc}_s{seed}.joblib
  bovw_train_k{K}_{hard|soft}.npy
  bovw_test_k{K}_{hard|soft}.npy
  pmk_hists_train_k{K}_L{L}.npy
  pmk_hists_test_k{K}_L{L}.npy
  pmk_kernel_train_k{K}_L{L}.npy  

Outputs (out_root/run_id):
  summary.csv
  comparison_test_f1_macro.png
  <variant_name>/
    best_params.json
    confusion_raw.png
    confusion_normalized.png
  roc/
    roc_<variant_name>.npz
    roc_per_class_<variant_name>.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from filelock import FileLock
from joblib import Parallel, delayed, dump, load
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics.pairwise import chi2_kernel, euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import LinearSVC, SVC

from mylogging import LOG, StepTimer, arr_info, cache_status
from ecoc import MyECOC
from results import best_models


# =============================================================================
# Constants / Specs
# =============================================================================

SUMMARY_FIELDS = [
    "run_id", "variant", "family", "assignment",
    "k", "cv_score", "test_acc", "test_f1_macro",
    "roc_auc_micro", "roc_npz", "roc_per_class_npz",
    "best_params_json", "out_dir"
]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str           # output directory name, and key in best_models
    assignment: str     # hard | soft | pmk
    family: str         # knn | linear | chi2 | ecoc | svc | pmk
    key: str            # short selector: knn|linear|chi2|ecoc|soft|pmk


EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec("1_BoVW_Hard_kNN",        "hard", "knn",    "knn"),
    ExperimentSpec("2_BoVW_Hard_LinearSVC",  "hard", "linear", "linear"),
    ExperimentSpec("3_BoVW_Hard_Chi2SVC",    "hard", "chi2",   "chi2"),
    ExperimentSpec("4_BoVW_Hard_ECOC",       "hard", "ecoc",   "ecoc"),
    ExperimentSpec("5_BoVW_Soft_SVC",        "soft", "svc",    "soft"),
    ExperimentSpec("6_PMK",                 "pmk",  "pmk",    "pmk"),
]

KEY_TO_EXPNAME = {e.key: e.name for e in EXPERIMENTS}


@dataclass
class Result:
    variant: str
    k: int
    assignment: str
    best_params: dict
    cv_score: float
    test_acc: float
    test_f1: float


# =============================================================================
# File / small utilities
# =============================================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def cache_path(cache_dir: str, name: str) -> str:
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, name)


def sha1_of_list(strings: List[str]) -> str:
    h = hashlib.sha1()
    for s in strings:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def append_row_locked(csv_path: str, fieldnames: List[str], row: dict) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    lock = FileLock(csv_path + ".lock")
    with lock:
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            w.writerow(row)


def dump_atomic(obj, path: str) -> None:
    # joblib dump is already fairly safe, but we keep a simple atomic write pattern
    tmp = path + ".tmp"
    dump(obj, tmp)
    os.replace(tmp, path)


def load_if_exists(path: str):
    return load(path) if os.path.exists(path) else None


# =============================================================================
# Data loading
# =============================================================================

def load_images_from_folder(folder: str, max_images: Optional[int] = None) -> Dict[str, dict]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    data: Dict[str, dict] = {}
    class_dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    class_dirs.sort()

    count = 0
    for label in class_dirs:
        folder_path = os.path.join(folder, label)
        for fn in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, fn)
            img = cv.imread(img_path)
            if img is None:
                continue
            data[img_path] = {"img": img, "label": label}
            count += 1
            if max_images is not None and count >= max_images:
                return data
    return data


def dict_to_lists(data: Dict[str, dict]) -> Tuple[List[str], List[np.ndarray], List[str]]:
    paths = list(data.keys())
    images = [data[p]["img"] for p in paths]
    labels = [data[p]["label"] for p in paths]
    return paths, images, labels


# =============================================================================
# SIFT descriptors (cached)
# =============================================================================

def compute_sift_for_image(img: np.ndarray, nfeatures: int = 0) -> Optional[np.ndarray]:
    sift = cv.SIFT_create(nfeatures=nfeatures) if nfeatures > 0 else cv.SIFT_create()
    _kps, desc = sift.detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return None
    return desc.astype(np.float32, copy=False)


def compute_descriptors_cached(
    images: List[np.ndarray],
    paths: List[str],
    cache_file: str,
    n_jobs: int,
    nfeatures: int = 0,
) -> List[Optional[np.ndarray]]:
    if os.path.exists(cache_file):
        return load(cache_file)

    # threading is fine: OpenCV releases GIL in many ops; joblib threads often good here
    descs = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute_sift_for_image)(img, nfeatures=nfeatures) for img in images
    )
    ensure_dir(os.path.dirname(cache_file) or ".")
    dump_atomic(descs, cache_file)
    return descs


def sample_descriptors(train_descs: List[Optional[np.ndarray]], max_total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_desc = [d for d in train_descs if d is not None and len(d) > 0]
    if not all_desc:
        raise RuntimeError("No descriptors found in training set.")

    X = np.vstack(all_desc)  # (N, 128)
    if X.shape[0] <= max_total:
        return X.astype(np.float32, copy=False)

    idx = rng.choice(X.shape[0], size=max_total, replace=False)
    return X[idx].astype(np.float32, copy=False)


def fit_codebook_cached(
    train_descs: List[Optional[np.ndarray]],
    k: int,
    max_desc: int,
    seed: int,
    cache_dir: str,
    tag: str,
    batch_size: int = 4096,
) -> MiniBatchKMeans:
    fname = f"codebook_{tag}_k{k}_md{max_desc}_s{seed}.joblib"
    path = cache_path(cache_dir, fname)

    model = load_if_exists(path)
    if model is not None:
        return model

    X = sample_descriptors(train_descs, max_total=max_desc, seed=seed)
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=batch_size,
        n_init="auto",
        max_iter=200,
    )
    kmeans.fit(X)
    dump_atomic(kmeans, path)
    return kmeans


# =============================================================================
# BoVW encoding (cached)
# =============================================================================

def normalize_rows_inplace(H: np.ndarray) -> None:
    s = H.sum(axis=1, keepdims=True)
    np.divide(H, np.maximum(s, 1e-12), out=H)


def bovw_hard_one(desc: Optional[np.ndarray], kmeans: MiniBatchKMeans) -> np.ndarray:
    k = kmeans.n_clusters
    if desc is None or len(desc) == 0:
        return np.zeros(k, dtype=np.float32)
    idx = kmeans.predict(desc)
    return np.bincount(idx, minlength=k).astype(np.float32, copy=False)


def bovw_soft_one(desc: Optional[np.ndarray], centers: np.ndarray) -> np.ndarray:
    k = centers.shape[0]
    if desc is None or len(desc) == 0:
        return np.zeros(k, dtype=np.float32)

    dist = euclidean_distances(desc, centers)  # (n_desc, k)
    sigma = float(dist.mean()) + 1e-12
    w = np.exp(-(dist ** 2) / (2.0 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)
    return w.sum(axis=0).astype(np.float32, copy=False)


def encode_bovw_cached(
    descs: List[Optional[np.ndarray]],
    kmeans: MiniBatchKMeans,
    assignment: str,
    cache_file: str,
    n_jobs: int,
    normalize: bool = True,
) -> np.ndarray:
    if os.path.exists(cache_file):
        return np.load(cache_file)

    if assignment == "hard":
        H_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(bovw_hard_one)(d, kmeans) for d in descs
        )
        H = np.vstack(H_list)
    elif assignment == "soft":
        centers = kmeans.cluster_centers_
        H_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(bovw_soft_one)(d, centers) for d in descs
        )
        H = np.vstack(H_list)
    else:
        raise ValueError("assignment must be 'hard' or 'soft'")

    if normalize:
        H = H.astype(np.float32, copy=False)
        normalize_rows_inplace(H)

    np.save(cache_file, H)
    return H


# =============================================================================
# Confusion matrices + metrics
# =============================================================================

def save_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    out_dir: str,
    title: str,
) -> None:
    ensure_dir(out_dir)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, xticks_rotation="vertical", ax=ax
    )
    ax.set_title(f"{title} – Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_raw.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=labels,
        xticks_rotation="vertical",
        normalize="true",
        values_format=".1f",
        ax=ax,
    )
    ax.set_title(f"{title} – Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_normalized.png"), dpi=200)
    plt.close(fig)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro"))


# =============================================================================
# ROC helpers
# =============================================================================

def get_score_matrix(clf, X, n_classes: int) -> np.ndarray:
    """
    Returns y_score with shape (n_samples, n_classes) for ROC.
    Uses predict_proba if available, otherwise decision_function.
    """
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X)
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X)
    else:
        raise ValueError("Classifier has neither predict_proba nor decision_function.")

    y_score = np.asarray(y_score)

    # binary special case: (n_samples,) -> (n_samples, 2)
    if y_score.ndim == 1:
        y_score = np.vstack([-y_score, y_score]).T

    if y_score.shape[1] != n_classes:
        raise ValueError(f"Expected score shape (*,{n_classes}) got {y_score.shape}")

    return y_score


def roc_per_class(y_true: np.ndarray, y_score: np.ndarray, n_classes: int) -> Dict[int, dict]:
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    out: Dict[int, dict] = {}
    for k in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, k], y_score[:, k])
        roc_auc = auc(fpr, tpr)
        out[k] = {
            "fpr": fpr.astype(np.float32),
            "tpr": tpr.astype(np.float32),
            "auc": float(roc_auc),
        }
    return out


def micro_avg_roc(y_true: np.ndarray, y_score: np.ndarray, n_classes: int):
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    fpr, tpr, _ = roc_curve(Y.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr.astype(np.float32), tpr.astype(np.float32), float(roc_auc)


def save_roc_artifacts(
    *,
    out_dir: str,
    variant_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_classes: int,
) -> Tuple[float, str, str]:
    roc_dir = os.path.join(out_dir, "roc")
    ensure_dir(roc_dir)

    fpr, tpr, roc_auc = micro_avg_roc(y_true, y_score, n_classes)
    roc_npz = os.path.join(roc_dir, f"roc_{variant_name}.npz")
    np.savez_compressed(roc_npz, fpr=fpr, tpr=tpr, auc=roc_auc)

    roc_data = roc_per_class(y_true, y_score, n_classes)
    roc_class_path = os.path.join(roc_dir, f"roc_per_class_{variant_name}.npz")
    save_dict = {}
    for k, data in roc_data.items():
        save_dict[f"fpr_{k}"] = data["fpr"]
        save_dict[f"tpr_{k}"] = data["tpr"]
        save_dict[f"auc_{k}"] = data["auc"]
    np.savez_compressed(roc_class_path, **save_dict)

    return roc_auc, roc_npz, roc_class_path


# =============================================================================
# Model tuning (fast CV on cached full-train features)
# =============================================================================

def cv_score_for_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    clf,
    cv_splits: int,
    seed: int,
    scorer: str,
) -> float:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scores: List[float] = []
    for tr, va in cv.split(X_train, y_train):
        clf.fit(X_train[tr], y_train[tr])
        pred = clf.predict(X_train[va])
        if scorer == "accuracy":
            scores.append(accuracy_score(y_train[va], pred))
        else:
            scores.append(f1_score(y_train[va], pred, average="macro"))
    return float(np.mean(scores))


def tune_vector_models_fast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scorer: str,
    cv_splits: int,
    seed: int,
    which: str,
) -> Tuple[dict, float]:
    """
    Returns (best_params, best_score) for the given family.
    This is 'fast' tuning: codebook is fixed on full train; CV only tunes classifier.
    """
    best = -1.0
    best_params: Optional[dict] = None

    if which == "knn":
        for n_neighbors in [1, 3, 5, 7, 9, 11]:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            s = cv_score_for_features(X_train, y_train, clf, cv_splits, seed, scorer)
            if s > best:
                best = s
                best_params = {"n_neighbors": n_neighbors}

    elif which == "linear":
        for C in [0.01, 0.1, 1, 10]:
            base = LinearSVC(C=C, dual=False, max_iter=8000)
            clf = OneVsRestClassifier(base)
            s = cv_score_for_features(X_train, y_train, clf, cv_splits, seed, scorer)
            if s > best:
                best = s
                best_params = {"C": C}

    elif which == "svc":
        for C in [0.1, 1, 10]:
            for ker in ["rbf", "linear"]:
                base = SVC(C=C, kernel=ker, gamma="scale")
                clf = OneVsRestClassifier(base)
                s = cv_score_for_features(X_train, y_train, clf, cv_splits, seed, scorer)
                if s > best:
                    best = s
                    best_params = {"C": C, "kernel": ker}

    elif which == "ecoc":
        for C in [1, 5, 10]:
            for ker in ["linear", "poly", "rbf"]:
                code_size = 2
                base = SVC(C=C, kernel=ker, gamma="scale")
                clf = MyECOC(estimator=OneVsRestClassifier(base), code_size=code_size, random_state=seed)
                s = cv_score_for_features(X_train, y_train, clf, cv_splits, seed, scorer)
                if s > best:
                    best = s
                    best_params = {"C": C, "kernel": ker, "code_size": code_size}

    elif which == "chi2":
        # chi2 uses precomputed kernel; tune (A,C)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        for A in [0.5, 1.0, 2.0, 4.0]:
            gamma = 1.0 / float(A)
            K = chi2_kernel(X_train, X_train, gamma=gamma)
            for C in [0.1, 1, 10]:
                fold_scores = []
                for tr, va in cv.split(K, y_train):
                    K_tr = K[np.ix_(tr, tr)]
                    K_va = K[np.ix_(va, tr)]
                    base = SVC(C=C, kernel="precomputed")
                    clf = OneVsRestClassifier(base)
                    clf.fit(K_tr, y_train[tr])
                    pred = clf.predict(K_va)
                    if scorer == "accuracy":
                        fold_scores.append(accuracy_score(y_train[va], pred))
                    else:
                        fold_scores.append(f1_score(y_train[va], pred, average="macro"))
                s = float(np.mean(fold_scores))
                if s > best:
                    best = s
                    best_params = {"A": A, "C": C}
    else:
        raise ValueError(which)

    assert best_params is not None
    return best_params, best


# =============================================================================
# Fit / predict unified
# =============================================================================

def fit_predict_and_score(
    family: str,
    best_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    *,
    chi2_train_kernel: Optional[np.ndarray] = None,
    chi2_test_kernel: Optional[np.ndarray] = None,
):
    """
    Returns: (clf, pred, y_score, X_used_for_score)
    """
    if family == "knn":
        clf = KNeighborsClassifier(n_neighbors=best_cfg["n_neighbors"])
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        try:
            y_score = get_score_matrix(clf, X_test, n_classes)
        except Exception as e:
            LOG.info(f"[get_score_matrix failed knn] {type(e).__name__}: {e}")
            y_score = None
        return clf, pred, y_score, X_test

    if family == "linear":
        base = LinearSVC(C=best_cfg["C"], dual=False, max_iter=8000)
        clf = OneVsRestClassifier(base)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        try:
            y_score = get_score_matrix(clf, X_test, n_classes)
        except Exception as e:
            LOG.info(f"[get_score_matrix failed linear] {type(e).__name__}: {e}")
            y_score = None
        return clf, pred, y_score, X_test

    if family == "svc":
        base = SVC(C=best_cfg["C"], kernel=best_cfg["kernel"], gamma="scale")
        clf = OneVsRestClassifier(base)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        y_score = get_score_matrix(clf, X_test, n_classes)
        return clf, pred, y_score, X_test

    if family == "ecoc":
        base = SVC(C=best_cfg["C"], kernel=best_cfg["kernel"], gamma="scale")
        clf = MyECOC(estimator=OneVsRestClassifier(base), code_size=best_cfg["code_size"])
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        try:
            y_score = get_score_matrix(clf, X_test, n_classes)
        except Exception as e:
            LOG.info(f"[get_score_matrix failed ecoc] {type(e).__name__}: {e}")
            y_score = None
        return clf, pred, y_score, X_test

    if family == "chi2":
        if chi2_train_kernel is None or chi2_test_kernel is None:
            raise ValueError("chi2 requires chi2_train_kernel and chi2_test_kernel")
        base = SVC(C=best_cfg["C"], kernel="precomputed")
        clf = OneVsRestClassifier(base)
        clf.fit(chi2_train_kernel, y_train)
        pred = clf.predict(chi2_test_kernel)
        y_score = get_score_matrix(clf, chi2_test_kernel, n_classes)
        return clf, pred, y_score, chi2_test_kernel

    raise ValueError(f"Unknown family: {family}")


# =============================================================================
# PMK implementation (your optimized approach)
# =============================================================================

def sift_kp_desc_for_image(img: np.ndarray, nfeatures: int = 0):
    sift = cv.SIFT_create(nfeatures=nfeatures) if nfeatures > 0 else cv.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return np.zeros((0, 2), dtype=np.float32), None
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    return pts, desc.astype(np.float32, copy=False)


def pmk_hist_tensor_one_image(
    img: np.ndarray,
    kmeans: MiniBatchKMeans,
    L: int,
    nfeatures: int = 0,
) -> np.ndarray:
    """
    Returns tensor (L, T_max, K) where T_max = (2^(L-1))^2.
    Level l has T_l = (2^l)^2 bins; bins beyond T_l are 0.
    """
    K = kmeans.n_clusters
    T_max = (2 ** (L - 1)) ** 2
    tensor = np.zeros((L, T_max, K), dtype=np.float32)

    pts, desc = sift_kp_desc_for_image(img, nfeatures=nfeatures)
    if desc is None or len(desc) == 0:
        return tensor

    vw = kmeans.predict(desc)
    h_img, w_img = img.shape[0], img.shape[1]
    xs = pts[:, 0] / max(w_img, 1e-12)
    ys = pts[:, 1] / max(h_img, 1e-12)

    for l in range(L):
        n1d = 2 ** l
        cx = np.minimum((xs * n1d).astype(np.int32), n1d - 1)
        cy = np.minimum((ys * n1d).astype(np.int32), n1d - 1)
        cell = cy * n1d + cx
        T_l = n1d * n1d

        # accumulate per cell, per visual word
        for t in range(T_l):
            idx = np.where(cell == t)[0]
            if idx.size == 0:
                continue
            hist = np.bincount(vw[idx], minlength=K).astype(np.float32, copy=False)
            tensor[l, t, :] = hist

    # per-bin L1 normalization
    s = tensor.sum(axis=-1, keepdims=True)
    tensor = tensor / np.maximum(s, 1e-12)
    return tensor


def pmk_build_hist_tensors_cached(
    images: List[np.ndarray],
    kmeans: MiniBatchKMeans,
    L: int,
    cache_file: str,
    n_jobs: int,
    nfeatures: int = 0,
) -> np.ndarray:
    if os.path.exists(cache_file):
        return np.load(cache_file)

    tensors = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(pmk_hist_tensor_one_image)(img, kmeans, L, nfeatures=nfeatures) for img in images
    )
    A = np.stack(tensors, axis=0).astype(np.float32, copy=False)  # (N, L, T, K)
    np.save(cache_file, A)
    return A


def pyramid_match_kernel_blocked(
    A: np.ndarray,
    B: np.ndarray,
    L: int,
    block_rows: int = 32,
    block_k: int = 64,
    dtype_out=np.float32,
) -> np.ndarray:
    """
    Memory-safe PMK: computes K(A,B) by blocking over rows and visual-word dimension.
    A: (N, L, T, K)
    B: (M, L, T, K)
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    N, L2, T, K = A.shape
    M = B.shape[0]
    assert L2 == L

    weights = np.array([1.0 / (2 ** (L - l)) for l in range(L)], dtype=np.float32)

    Kmat = np.zeros((N, M), dtype=np.float32)

    for i0 in range(0, N, block_rows):
        i1 = min(i0 + block_rows, N)
        Ab = A[i0:i1]  # (b, L, T, K)
        b = Ab.shape[0]

        Kb = np.zeros((b, M), dtype=np.float32)

        for l in range(L):
            w = weights[l]
            for t in range(T):
                acc_bt = np.zeros((b, M), dtype=np.float32)

                for k0 in range(0, K, block_k):
                    k1 = min(k0 + block_k, K)
                    Ab_slice = Ab[:, l, t, k0:k1]
                    B_slice = B[:, l, t, k0:k1]
                    acc_bt += np.minimum(Ab_slice[:, None, :], B_slice[None, :, :]).sum(axis=-1)

                Kb += w * acc_bt

        Kmat[i0:i1] = Kb.astype(dtype_out, copy=False)

    return Kmat


def pyramid_match_kernel(A: np.ndarray, B: np.ndarray, L: int) -> np.ndarray:
    return pyramid_match_kernel_blocked(A, B, L, block_rows=32, block_k=64)


def normalize_kernel_train(K: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.outer(np.diag(K), np.diag(K))) + 1e-12
    return K / diag


def normalize_kernel_test(K_test: np.ndarray, K_train: np.ndarray, K_self_test: np.ndarray) -> np.ndarray:
    denom = np.sqrt((K_self_test + 1e-12)[:, None] * (np.diag(K_train) + 1e-12)[None, :])
    return K_test / denom


# =============================================================================
# Unified evaluation + summary writing
# =============================================================================

def evaluate_and_record(
    *,
    out_dir: str,
    run_id: str,
    variant_name: str,
    family: str,
    assignment: str,
    k: int,
    cv_score: float,
    best_cfg: dict,
    v_dir: str,
    label_names: List[str],
    y_test: np.ndarray,
    pred: np.ndarray,
    y_score: Optional[np.ndarray],
) -> Tuple[float, float]:
    save_confusion_matrices(y_test, pred, label_names, v_dir, variant_name)
    acc, f1m = score(y_test, pred)

    roc_auc = roc_npz = roc_per_class_npz = ""
    if y_score is not None:
        n_classes = len(label_names)
        roc_auc_f, roc_npz_f, roc_pc_f = save_roc_artifacts(
            out_dir=out_dir,
            variant_name=variant_name,
            y_true=y_test,
            y_score=y_score,
            n_classes=n_classes,
        )
        roc_auc = roc_auc_f
        roc_npz = roc_npz_f
        roc_per_class_npz = roc_pc_f

    row = {
        "run_id": run_id,
        "variant": variant_name,
        "family": family,
        "assignment": assignment,
        "k": k,
        "cv_score": cv_score,
        "test_acc": acc,
        "test_f1_macro": f1m,
        "roc_auc_micro": roc_auc,
        "roc_npz": roc_npz,
        "roc_per_class_npz": roc_per_class_npz,
        "best_params_json": json.dumps(best_cfg, sort_keys=True),
        "out_dir": v_dir,
    }
    append_row_locked(os.path.join(out_dir, "summary.csv"), SUMMARY_FIELDS, row)

    LOG.info(f"[{variant_name}] ✅ FINAL: k={k} CV={cv_score:.4f} | Test acc={acc:.4f} f1={f1m:.4f}")
    return acc, f1m


# =============================================================================
# Plotting comparison
# =============================================================================

def plot_comparison(results: List[Result], out_dir: str) -> None:
    if not results:
        return
    names = [r.variant for r in results]
    vals = [r.test_f1 for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(names, vals)
    ax.set_ylabel("Test F1 Macro")
    ax.set_title("Comparison: Test F1 Macro")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_test_f1_macro.png"), dpi=200)
    plt.close(fig)


# =============================================================================
# Main experiment runners (vector vs PMK)
# =============================================================================

def run_vector_experiment(
    *,
    spec: ExperimentSpec,
    args,
    out_dir: str,
    run_id: str,
    v_dir: str,
    train_descs,
    test_descs,
    train_paths: List[str],
    X_cache_tag: str,
    ks: List[int],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
) -> Result:
    variant_name = spec.name
    assignment = spec.assignment
    family = spec.family

    given_best = args.use_best_models

    best_overall = -1.0
    best_cfg: Optional[dict] = None
    best_k: Optional[int] = None

    if not given_best:
        LOG.info(f"[{variant_name}] Vector tuning over ks={ks} family={family} assignment={assignment}")
        for ki, k in enumerate(ks, start=1):
            tag = X_cache_tag

            with StepTimer(f"{variant_name} | fit_codebook_cached | k={k} ({ki}/{len(ks)})"):
                kmeans = fit_codebook_cached(train_descs, k, args.max_desc, args.seed, args.cache_dir, tag)

            Xtr_file = cache_path(args.cache_dir, f"bovw_train_k{k}_{assignment}.npy")
            Xte_file = cache_path(args.cache_dir, f"bovw_test_k{k}_{assignment}.npy")
            LOG.info(f"[{variant_name}] k={k} Xtr={cache_status(Xtr_file)} Xte={cache_status(Xte_file)}")

            with StepTimer(f"{variant_name} | encode_bovw_cached(train) | k={k}"):
                X_train = encode_bovw_cached(train_descs, kmeans, assignment, Xtr_file, n_jobs=args.n_jobs, normalize=True)

            with StepTimer(f"{variant_name} | encode_bovw_cached(test)  | k={k}"):
                _X_test = encode_bovw_cached(test_descs, kmeans, assignment, Xte_file, n_jobs=args.n_jobs, normalize=True)

            with StepTimer(f"{variant_name} | tune_vector_models_fast | k={k} family={family}"):
                params, cvbest = tune_vector_models_fast(X_train, y_train, args.scorer, args.cv_splits, args.seed, family)

            cfg = {"k": k, **params}
            LOG.info(f"[{variant_name}] k={k} CV={cvbest:.4f} params={cfg}")

            if cvbest > best_overall:
                best_overall = cvbest
                best_cfg = cfg
                best_k = k
                LOG.info(f"[{variant_name}] ✅ NEW BEST: k={best_k} CV={best_overall:.4f} cfg={best_cfg}")

        assert best_cfg is not None and best_k is not None
        save_json({"model": variant_name, "best_params": best_cfg, "best_cv_score": best_overall}, os.path.join(v_dir, "best_params.json"))

    else:
        # load pre-picked best config
        variant_cfg = best_models[variant_name]
        best_k = int(variant_cfg["k"])
        best_overall = float(variant_cfg["score"])
        params = {k: v for k, v in variant_cfg.items() if k not in ["k", "score", "model_id", "model_name"]}
        best_cfg = {"k": best_k, **params}
        save_json({"model": variant_name, "best_params": best_cfg, "best_cv_score": best_overall}, os.path.join(v_dir, "best_params.json"))
        LOG.info(f"[{variant_name}] Using best_models: k={best_k} cfg={best_cfg} CV={best_overall:.4f}")

    assert best_cfg is not None and best_k is not None

    # final train/test with cached features for best k
    X_train_path = cache_path(args.cache_dir, f"bovw_train_k{best_k}_{assignment}.npy")
    X_test_path = cache_path(args.cache_dir, f"bovw_test_k{best_k}_{assignment}.npy")

    with StepTimer(f"{variant_name} | load cached X | best_k={best_k}"):
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)

    LOG.info(arr_info(X_train, f"X_train(best) k={best_k}"))
    LOG.info(arr_info(X_test, f"X_test(best) k={best_k}"))

    n_classes = len(label_names)

    with StepTimer(f"{variant_name} | final fit/predict | family={family}"):
        if family == "chi2":
            gamma = 1.0 / float(best_cfg["A"])
            K_tr = chi2_kernel(X_train, X_train, gamma=gamma)
            K_te = chi2_kernel(X_test, X_train, gamma=gamma)
            _clf, pred, y_score, _ = fit_predict_and_score(
                family, best_cfg, X_train, y_train, X_test, n_classes,
                chi2_train_kernel=K_tr, chi2_test_kernel=K_te
            )
        else:
            _clf, pred, y_score, _ = fit_predict_and_score(family, best_cfg, X_train, y_train, X_test, n_classes)

    acc, f1m = evaluate_and_record(
        out_dir=out_dir,
        run_id=run_id,
        variant_name=variant_name,
        family=family,
        assignment=assignment,
        k=int(best_k),
        cv_score=float(best_overall),
        best_cfg=best_cfg,
        v_dir=v_dir,
        label_names=label_names,
        y_test=y_test,
        pred=pred,
        y_score=y_score,
    )

    return Result(variant_name, int(best_k), assignment, best_cfg, float(best_overall), acc, f1m)


def run_pmk_experiment(
    *,
    spec: ExperimentSpec,
    args,
    out_dir: str,
    run_id: str,
    v_dir: str,
    train_descs,
    train_images: List[np.ndarray],
    test_images: List[np.ndarray],
    train_paths: List[str],
    ks: List[int],
    pmk_levels: List[int],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
) -> Result:
    variant_name = spec.name
    family = spec.family
    assignment = spec.assignment
    assert family == "pmk"

    given_best = args.use_best_models
    X_cache_tag = f"trainfull_{sha1_of_list(train_paths)}"

    best_overall = -1.0
    best_params: Optional[dict] = None
    best_k: Optional[int] = None
    best_L: Optional[int] = None

    if not given_best:
        LOG.info(f"[{variant_name}] PMK tuning grid: ks={ks} x L={pmk_levels}")

        for ki, k in enumerate(ks, start=1):
            with StepTimer(f"{variant_name} | fit_codebook_cached | k={k} ({ki}/{len(ks)})"):
                kmeans = fit_codebook_cached(train_descs, k, args.max_desc, args.seed, args.cache_dir, X_cache_tag)

            for li, L in enumerate(pmk_levels, start=1):
                htr_file = cache_path(args.cache_dir, f"pmk_hists_train_k{k}_L{L}.npy")
                K_train_file = cache_path(args.cache_dir, f"pmk_kernel_train_k{k}_L{L}.npy")

                LOG.info(f"[{variant_name}] (k={k} {ki}/{len(ks)}) (L={L} {li}/{len(pmk_levels)}) "
                         f"htr={cache_status(htr_file)} Ktr={cache_status(K_train_file)}")

                with StepTimer(f"{variant_name} | pmk_build_hist_tensors_cached | train | k={k} L={L}"):
                    A = pmk_build_hist_tensors_cached(
                        train_images, kmeans, L, htr_file,
                        n_jobs=args.n_jobs, nfeatures=args.sift_nfeatures
                    )

                with StepTimer(f"{variant_name} | build K_tr + normalize | k={k} L={L}"):
                    K_tr = pyramid_match_kernel(A, A, L=L)
                    K_tr_n = normalize_kernel_train(K_tr)

                # CV on precomputed kernel
                cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
                fold_scores = []

                with StepTimer(f"{variant_name} | CV precomputed kernel | k={k} L={L}"):
                    for tr, va in cv.split(K_tr_n, y_train):
                        K_fold_tr = K_tr_n[np.ix_(tr, tr)]
                        K_fold_va = K_tr_n[np.ix_(va, tr)]

                        clf = SVC(kernel="precomputed", C=1.0)
                        clf.fit(K_fold_tr, y_train[tr])
                        pred = clf.predict(K_fold_va)

                        if args.scorer == "accuracy":
                            fold_scores.append(accuracy_score(y_train[va], pred))
                        else:
                            fold_scores.append(f1_score(y_train[va], pred, average="macro"))

                s = float(np.mean(fold_scores))
                LOG.info(f"[{variant_name}] score(k={k}, L={L}) CV={s:.4f} folds={np.round(fold_scores, 4).tolist()}")

                if s > best_overall:
                    best_overall = s
                    best_params = {"k": k, "L": L, "C": 1.0}
                    best_k, best_L = k, L
                    LOG.info(f"[{variant_name}] ✅ NEW BEST: CV={best_overall:.4f} params={best_params}")

        assert best_params is not None and best_k is not None and best_L is not None
        save_json({"model": variant_name, "best_params": best_params, "best_cv_score": best_overall}, os.path.join(v_dir, "best_params.json"))

    else:
        variant_cfg = best_models[variant_name]
        best_k = int(variant_cfg["k"])
        best_L = int(variant_cfg.get("L", variant_cfg.get("pmk_L", 2)))  # tolerate different keys
        best_overall = float(variant_cfg["score"])
        best_params = {"k": best_k, "L": best_L, "C": float(variant_cfg.get("C", 1.0))}
        save_json({"model": variant_name, "best_params": best_params, "best_cv_score": best_overall}, os.path.join(v_dir, "best_params.json"))
        LOG.info(f"[{variant_name}] Using best_models: {best_params} CV={best_overall:.4f}")

    assert best_params is not None and best_k is not None and best_L is not None

    # final train/test
    with StepTimer(f"{variant_name} | fit codebook | k={best_k}"):
        kmeans = fit_codebook_cached(train_descs, best_k, args.max_desc, args.seed, args.cache_dir, X_cache_tag)

    htr_file = cache_path(args.cache_dir, f"pmk_hists_train_k{best_k}_L{best_L}.npy")
    hte_file = cache_path(args.cache_dir, f"pmk_hists_test_k{best_k}_L{best_L}.npy")

    with StepTimer(f"{variant_name} | build final A/B hists | k={best_k} L={best_L}"):
        A = pmk_build_hist_tensors_cached(train_images, kmeans, best_L, htr_file, n_jobs=args.n_jobs, nfeatures=args.sift_nfeatures)
        B = pmk_build_hist_tensors_cached(test_images,  kmeans, best_L, hte_file, n_jobs=args.n_jobs, nfeatures=args.sift_nfeatures)

    with StepTimer(f"{variant_name} | final kernels + normalize | k={best_k} L={best_L}"):
        K_tr = pyramid_match_kernel(A, A, L=best_L)
        K_te = pyramid_match_kernel(B, A, L=best_L)

        K_tr_n = normalize_kernel_train(K_tr)
        K_te_self = np.diag(pyramid_match_kernel(B, B, L=best_L))
        K_te_n = normalize_kernel_test(K_te, K_tr, K_te_self)

    with StepTimer(f"{variant_name} | fit/predict SVC(precomputed)"):
        clf = SVC(kernel="precomputed", C=float(best_params.get("C", 1.0)))
        clf.fit(K_tr_n, y_train)
        pred = clf.predict(K_te_n)

    n_classes = len(label_names)
    y_score = get_score_matrix(clf, K_te_n, n_classes)

    acc, f1m = evaluate_and_record(
        out_dir=out_dir,
        run_id=run_id,
        variant_name=variant_name,
        family=family,
        assignment=assignment,
        k=int(best_k),
        cv_score=float(best_overall),
        best_cfg=best_params,
        v_dir=v_dir,
        label_names=label_names,
        y_test=y_test,
        pred=pred,
        y_score=y_score,
    )

    return Result(variant_name, int(best_k), assignment, best_params, float(best_overall), acc, f1m)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_test", type=int, default=None)

    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--out_root", type=str, default="results")
    ap.add_argument("--run_id", type=str, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--scorer", type=str, default="f1_macro", choices=["f1_macro", "accuracy"])

    ap.add_argument("--ks", type=str, default="32,64,128,256,512")
    ap.add_argument("--max_desc", type=int, default=200_000)
    ap.add_argument("--sift_nfeatures", type=int, default=0)

    ap.add_argument("--pmk_levels", type=str, default="2,3")

    ap.add_argument("--use_best_models", action="store_true",
                    help="Use precomputed best configs from results.best_models instead of tuning.")

    ap.add_argument(
        "--variant",
        type=str,
        default="all",
        help="all OR one of: knn, linear, chi2, ecoc, soft, pmk OR full variant name like 1_BoVW_Hard_kNN",
    )

    return ap.parse_args()


def select_experiments(variant_arg: str) -> List[ExperimentSpec]:
    if variant_arg == "all":
        return EXPERIMENTS

    # allow short key (knn/linear/...) or full experiment name
    if variant_arg in KEY_TO_EXPNAME:
        name = KEY_TO_EXPNAME[variant_arg]
        return [e for e in EXPERIMENTS if e.name == name]

    # full name
    return [e for e in EXPERIMENTS if e.name == variant_arg]


def main():
    args = parse_args()
    set_seeds(args.seed)

    run_id = args.run_id or now_stamp()
    out_dir = os.path.join(args.out_root, run_id)
    ensure_dir(out_dir)

    train_folder = os.path.join(args.data_root, "train")
    test_folder = os.path.join(args.data_root, "test")

    train_data = load_images_from_folder(train_folder, max_images=args.max_train)
    test_data = load_images_from_folder(test_folder, max_images=args.max_test)

    train_paths, train_images, train_labels_str = dict_to_lists(train_data)
    _test_paths, test_images, test_labels_str = dict_to_lists(test_data)

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels_str)
    y_test = le.transform(test_labels_str)
    label_names = list(le.classes_)

    print(f"Train: {len(train_images)} | Test: {len(test_images)} | Classes: {len(label_names)}")

    # descriptors cached
    desc_train_file = cache_path(args.cache_dir, "descriptors_train.pkl")
    desc_test_file = cache_path(args.cache_dir, "descriptors_test.pkl")

    print("Loading/computing SIFT descriptors (cached)...")
    train_descs = compute_descriptors_cached(train_images, train_paths, desc_train_file, n_jobs=args.n_jobs, nfeatures=args.sift_nfeatures)
    test_descs = compute_descriptors_cached(test_images, _test_paths, desc_test_file, n_jobs=args.n_jobs, nfeatures=args.sift_nfeatures)

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    pmk_levels = [int(x.strip()) for x in args.pmk_levels.split(",") if x.strip()]

    exp_list = select_experiments(args.variant)
    print(f"Experiment order: {[e.name for e in exp_list]}")
    print(f"use_best_models={args.use_best_models} | ks={ks} | pmk_levels={pmk_levels}")

    LOG.info("Starting Experiments...")
    LOG.info(arr_info(train_paths, "train_paths"))
    LOG.info(arr_info(y_train, "y_train"))
    LOG.info(arr_info(y_test, "y_test"))
    LOG.info(arr_info(ks, "ks"))
    LOG.info(arr_info(pmk_levels, "pmk_levels"))

    results: List[Result] = []

    X_cache_tag = f"trainfull_{sha1_of_list(train_paths)}"

    for i, spec in enumerate(exp_list, start=1):
        variant_name = spec.name
        v_dir = os.path.join(out_dir, variant_name)
        ensure_dir(v_dir)

        LOG.info("------------------------------------------------------------")
        LOG.info(f"[VARIANT {i}/{len(exp_list)}] {variant_name} | family={spec.family} | assignment={spec.assignment}")
        LOG.info(f"Output dir: {v_dir}")

        if spec.family == "pmk":
            r = run_pmk_experiment(
                spec=spec,
                args=args,
                out_dir=out_dir,
                run_id=run_id,
                v_dir=v_dir,
                train_descs=train_descs,
                train_images=train_images,
                test_images=test_images,
                train_paths=train_paths,
                ks=ks,
                pmk_levels=pmk_levels,
                y_train=y_train,
                y_test=y_test,
                label_names=label_names,
            )
        else:
            r = run_vector_experiment(
                spec=spec,
                args=args,
                out_dir=out_dir,
                run_id=run_id,
                v_dir=v_dir,
                train_descs=train_descs,
                test_descs=test_descs,
                train_paths=train_paths,
                X_cache_tag=X_cache_tag,
                ks=ks,
                y_train=y_train,
                y_test=y_test,
                label_names=label_names,
            )

        results.append(r)

    plot_comparison(results, out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()

